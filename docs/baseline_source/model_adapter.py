"""Causal frame-append harness around unmodified pretrained SmolVLM2.

This is not ReKV retrieval or an upstream streaming benchmark. Visual tokens
are passed once; the same Transformers Cache survives frame and QA calls.
"""
import os
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, SmolVLMForConditionalGeneration
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.runtime/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

def decode_video(path, count=4):
    cap = cv2.VideoCapture(str(ROOT / path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open {path}')
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if n <= 0 or fps <= 0:
        cap.release()
        raise ValueError('Invalid video metadata')
    indices = np.unique(np.linspace(0, n - 1, min(count, n), dtype=int))
    frames = []
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f'Missing frame {index}; benchmark sample rejected')
            frames.append(dict(index=int(index), seconds=float(index / fps), bgr=bgr))
    finally:
        cap.release()
    return frames

def native_cache_bytes(cache):
    return sum(t.numel() * t.element_size() for layer in cache for t in layer)

class VideoModel:
    def __init__(self, config):
        self.config = config
        torch.set_num_threads(config['threads'])
        torch.manual_seed(config['seed'])
        self.device = torch.device(config['device'])
        self.dtype = getattr(torch, config['dtype'])
        path = str(ROOT / config['model_path'])
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        # 4.51's processor passes these properties explicitly to image_processor;
        # passing size in __call__ would supply it twice.
        self.processor.do_image_splitting = False
        self.processor.image_size = {'longest_edge': 512}
        self.model = SmolVLMForConditionalGeneration.from_pretrained(
            path, torch_dtype=self.dtype, attn_implementation='sdpa',
            local_files_only=True).to(self.device).eval()
        self.vision_calls = 0
        self._vision_hook = self.model.model.vision_model.register_forward_hook(self._count_vision)

    def _count_vision(self, *args):
        self.vision_calls += 1

    def text_inputs(self, text):
        return self.processor.tokenizer(text, add_special_tokens=False,
                                        return_tensors='pt')

    def frame_inputs(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame['bgr'], cv2.COLOR_BGR2RGB))
        text = f'\nFrame from {frame["seconds"]:.2f} seconds:<image>\n'
        return self.processor(text=text, images=[image], return_tensors='pt',
                              add_special_tokens=False)

    @torch.inference_mode()
    def forward(self, inputs, cache):
        length = inputs['input_ids'].shape[1]
        past = cache.get_seq_length()
        x = {key: value.to(device=self.device,
                          dtype=self.dtype if value.is_floating_point() else value.dtype)
             for key, value in inputs.items() if key != 'attention_mask'}
        x['attention_mask'] = torch.ones((1, past + length), device=self.device, dtype=torch.long)
        # Explicit positions ensure RoPE positions do not restart at frame boundaries.
        x['cache_position'] = torch.arange(past, past + length, device=self.device)
        return self.model(**x, past_key_values=cache, use_cache=True, logits_to_keep=1)

    def begin(self, cache=None):
        cache = DynamicCache() if cache is None else cache
        prefix = '<|im_start|>User: You are provided frames from a video in chronological order.\n'
        self.forward(self.text_inputs(prefix), cache)
        return cache

    def append(self, frame, cache):
        before = cache.get_seq_length()
        inputs = self.frame_inputs(frame)
        n_visual = int((inputs['input_ids'] == self.model.config.image_token_id).sum())
        self.forward(inputs, cache)
        return dict(start=before, end=cache.get_seq_length(), visual_tokens=n_visual,
                    source_index=frame['index'], seconds=frame['seconds'])

    def question_text(self, sample):
        options = '\n'.join(f'{chr(65+i)}. {s}' for i, s in enumerate(sample['candidates']))
        return ('\n' + sample['question'] + '\n' + options +
                '\nAnswer with only the option letter.<end_of_utterance>\nAssistant:')

    def answer(self, sample, cache):
        output = self.forward(self.text_inputs(self.question_text(sample)), cache)
        labels = [chr(65+i) for i in range(len(sample['candidates']))]
        tokens = [self.processor.tokenizer.encode(s, add_special_tokens=False) for s in labels]
        if any(len(x) != 1 for x in tokens):
            raise ValueError('This scoring protocol requires single-token labels')
        scores = output.logits[0, -1, [x[0] for x in tokens]].float()
        prediction = int(scores.argmax())
        return dict(prediction=prediction, label=labels[prediction],
                    predicted_text=sample['candidates'][prediction], logits=scores.tolist(),
                    correct=sample['candidates'][prediction] == sample['answer'])

    def describe(self, cache, max_tokens=32):
        x = self.text_inputs('\nDescribe the action in this video.<end_of_utterance>\nAssistant:')
        ids = []
        for _ in range(max_tokens):
            output = self.forward(x, cache)
            token = int(output.logits[0, -1].argmax())
            if token in {self.processor.tokenizer.eos_token_id,
                         self.processor.tokenizer.convert_tokens_to_ids('<end_of_utterance>')}:
                break
            ids.append(token)
            x = {'input_ids': torch.tensor([[token]])}
        return self.processor.tokenizer.decode(ids, skip_special_tokens=True)
