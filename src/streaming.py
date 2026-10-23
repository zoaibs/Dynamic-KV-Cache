"""Shared GPU streaming engine, visual feature reuse and unbiased QA protocols.

All compared cache modes share this engine and unchanged pretrained weights.
Native mode uses upstream DynamicCache/SDPA. Fused modes also combine decoder
projections and change normalization execution, with BF16 as a separate control.
Each QA branch starts at the identical video-prefix cache state.
"""
from dataclasses import dataclass
import json
import math
import time
import cv2
import numpy as np
import torch
from transformers.cache_utils import DynamicCache
from src.model_adapter import VideoModel,decode_video,native_cache_bytes
from src.information_score import InformationScorer
from src.gpu_cache import GPUCache,install_packed_attention
from src.gpu_ops import install_fused_ops
from src.selective_features import SelectiveScorer
from scripts.resources import ROOT

@dataclass
class PreparedVideo:
    sample: dict
    frames: list
    inputs: list
    features: list
    prepare_seconds: float
    feature_seconds:float=0.
    decode_seconds:float=0.
    preprocessing_seconds:float=0.

def score_prepared(prepared,config,weights=None):
    """Independent CPU work; safe to overlap with CUDA vision on the main thread."""
    active=[w!=0 for w in weights] if weights is not None else [True]*3
    scorer=SelectiveScorer(config,active)
    start=time.perf_counter()
    features=[scorer(frame['bgr']).to_dict() for frame in prepared.frames]
    return features,time.perf_counter()-start

def prepare_video(adapter,sample,frames,compute_features=True,feature_weights=None):
    start=time.perf_counter()
    # WebM frame-count headers and random seeks are unreliable for several
    # public clips. Decode to EOF, then sample real available frames only.
    if not 1<=frames<=96:raise ValueError('Frame cap is 96')
    cap=cv2.VideoCapture(str(ROOT/sample['path']))
    fps=cap.get(cv2.CAP_PROP_FPS)
    available=[]
    try:
        for _ in range(1024):
            ok,bgr=cap.read()
            if not ok:break
            available.append(bgr)
        else:raise ValueError('Video exceeds bounded sequential decode cap')
    finally:cap.release()
    if not available or fps<=0:raise ValueError('No valid decoded video frames')
    indices=np.unique(np.linspace(0,len(available)-1,min(frames,len(available)),dtype=int))
    decoded=[dict(index=int(i),seconds=float(i/fps),bgr=available[i],
                  available_frames=len(available)) for i in indices]
    decode_seconds=time.perf_counter()-start
    active=[w!=0 for w in feature_weights] if feature_weights is not None else [True]*3
    scorer=SelectiveScorer(adapter.config['score'],active)
    inputs=[];features=[]
    feature_seconds=preprocessing_seconds=0.
    for frame in decoded:
        t=time.perf_counter()
        inputs.append(adapter.frame_inputs(frame))
        preprocessing_seconds+=time.perf_counter()-t;t=time.perf_counter()
        features.append(scorer(frame['bgr']).to_dict() if compute_features else None)
        feature_seconds+=time.perf_counter()-t
    return PreparedVideo(sample,decoded,inputs,features,time.perf_counter()-start,
                         feature_seconds,decode_seconds,preprocessing_seconds)

def rotations(sample):
    for shift in range(len(sample['candidates'])):
        s=dict(sample)
        s['candidates']=sample['candidates'][shift:]+sample['candidates'][:shift]
        s['rotation']=shift
        yield s

def fork_cache(cache):
    if isinstance(cache,GPUCache):return cache.fork()
    # DynamicCache.update concatenates new tensors instead of mutating history.
    other=DynamicCache.from_legacy_cache(tuple((k,v) for k,v in cache))
    other.fast_gqa=getattr(cache,'fast_gqa',False)
    return other

class StreamingModel:
    def __init__(self,model_path='models/smolvlm2',vision_batch=8):
        config=json.loads((ROOT/'configs/default.json').read_text())
        config.update(device='cuda',model_path=model_path)
        cv2.setNumThreads(1)
        self.adapter=VideoModel(config)
        self.model=self.adapter.model
        self.core=self.model.model
        self.text=self.core.text_model
        self.tokenizer=self.adapter.processor.tokenizer
        self.dtype=self.adapter.dtype
        self.device=self.adapter.device
        self.vision_batch=vision_batch
        self.priors={}
        self.optimized_ops={'enabled':False}
        install_fused_ops(self.text,self.optimized_ops)
        self.prefix='<|im_start|>User: You are provided frames from a video in chronological order.\n'
        self.prefix_ids=self.encode(self.prefix)
        self.label_ids=torch.tensor([self.tokenizer.encode(x,add_special_tokens=False)[0]
                                     for x in ['A','B','C','D']],device='cuda')
        self.leading_label_ids=torch.tensor([self.tokenizer.encode(x,add_special_tokens=False)[0]
                                            for x in [' A',' B',' C',' D']],device='cuda')
        install_packed_attention(self.model)
        # A fused cache applies its own explicit bottom-right causal mask. Avoid
        # allocating a redundant dense mask that its attention never consumes.
        original=self.text._update_causal_mask
        def causal(attention_mask,input_tensor,cache_position,past_key_values,output_attentions):
            if isinstance(past_key_values,GPUCache) and past_key_values.backend=='fused':return None
            return original(attention_mask,input_tensor,cache_position,past_key_values,output_attentions)
        self.text._update_causal_mask=causal
        torch.cuda.empty_cache()

    def encode(self,text):
        return self.tokenizer(text,add_special_tokens=False,return_tensors='pt')['input_ids'].to('cuda')

    def empty_cache(self):
        self.optimized_ops['enabled']=False
        cache=DynamicCache();self.forward(self.prefix_ids,cache)
        return cache

    @torch.inference_mode()
    def encode_video(self,prepared):
        output=[]
        for start in range(0,len(prepared.inputs),self.vision_batch):
            chunk=prepared.inputs[start:start+self.vision_batch]
            pixels=torch.cat([x['pixel_values'] for x in chunk],dim=1).flatten(0,1).to('cuda',self.dtype)
            mask=torch.cat([x['pixel_attention_mask'] for x in chunk],dim=1).flatten(0,1).to('cuda')
            patch=self.model.config.vision_config.patch_size
            patch_mask=mask.unfold(1,patch,patch).unfold(2,patch,patch).sum((-1,-2))>0
            encoded=self.core.vision_model(pixel_values=pixels,patch_attention_mask=patch_mask).last_hidden_state
            encoded=self.core.connector(encoded)
            output.extend(encoded[i:i+1] for i in range(len(chunk)))
        return output

    @torch.inference_mode()
    def forward(self,ids,cache,image_features=None,visual_positions=None,logits=False):
        ids=ids.to('cuda')
        embeddings=self.text.get_input_embeddings()(ids)
        if image_features is not None:
            embeddings=embeddings.index_copy(1,visual_positions,image_features)
        past=cache.get_seq_length()
        out=self.text(inputs_embeds=embeddings,past_key_values=cache,use_cache=True,
                      cache_position=torch.arange(past,past+ids.shape[1],device='cuda'),
                      attention_mask=None,return_dict=True)
        return self.model.lm_head(out.last_hidden_state[:,-int(logits):]).float() if logits else None

    def create_cache(self,prepared,bits,mode,qa_reserve=192,**kernel):
        if mode in ('native','native_gqa'):
            cache=DynamicCache();cache.fast_gqa=mode=='native_gqa';return cache
        c=self.model.config.text_config
        plan=[(16,self.prefix_ids.shape[1])]
        plan.extend((bit,x['input_ids'].shape[1]) for bit,x in zip(bits,prepared.inputs))
        plan.append((16,qa_reserve))
        if sum(n for b,n in plan)>c.max_position_embeddings:
            raise ValueError('Context exceeds the checkpoint limit')
        return GPUCache(None,layers=c.num_hidden_layers,heads=c.num_key_value_heads,
                        dim=c.head_dim,dtype=self.dtype,backend='sdpa' if mode=='reconstruct' else 'fused',**kernel)

    @torch.inference_mode()
    def ingest(self,prepared,bits,mode,encoded=None,qa_reserve=None,**kernel):
        self.optimized_ops['enabled']=mode not in ('native','native_gqa','reconstruct')
        if qa_reserve is None:
            prompt=len(self.tokenizer.encode(self.question(prepared.sample,'candidate'),add_special_tokens=False))
            qa_reserve=prompt+max(len(self.tokenizer.encode(' '+c,add_special_tokens=False))
                                  for c in prepared.sample['candidates'])
        cache=self.create_cache(prepared,bits,mode,qa_reserve,**kernel)
        if encoded is None:encoded=self.encode_video(prepared)
        self.forward(self.prefix_ids,cache)
        for i,(inputs,embedding,bit) in enumerate(zip(prepared.inputs,encoded,bits)):
            if isinstance(cache,GPUCache):cache.select(bit,i)
            ids=inputs['input_ids']
            positions=(ids[0]==self.model.config.image_token_id).nonzero().flatten().to('cuda')
            if len(positions)!=embedding.shape[1]:raise ValueError('Visual-token mapping mismatch')
            self.forward(ids,cache,embedding,positions)
        if isinstance(cache,GPUCache):cache.select(16,-1)
        if isinstance(cache,GPUCache):cache.reserve(16,cache.host_counts[16]+qa_reserve)
        if isinstance(cache,GPUCache) and len(set(cache.seen))!=1:
            raise RuntimeError('Decoder layers disagree about the cache length')
        return cache

    def question(self,sample,protocol='label'):
        options='\n'.join(f'{chr(65+i)}. {x}' for i,x in enumerate(sample['candidates']))
        if protocol.startswith('candidate'):
            return ('\n'+sample['question']+'\nAnswer briefly with the action.'+
                    '<end_of_utterance>\nAssistant:')
        suffix=' ' if protocol=='label_space' else ''
        return ('\n'+sample['question']+'\n'+options+
                '\nAnswer with only the option letter.<end_of_utterance>\nAssistant:'+suffix)

    @torch.inference_mode()
    def answer(self,sample,base_cache,protocol='label'):
        self.optimized_ops['enabled']=isinstance(base_cache,GPUCache) and base_cache.backend=='fused'
        qids=self.encode(self.question(sample,protocol))
        if protocol.startswith('label'):
            native=not isinstance(base_cache,GPUCache)
            original_length=base_cache.get_seq_length()
            cache=base_cache if native else fork_cache(base_cache)
            labels=(self.leading_label_ids if protocol=='label_leading' else self.label_ids)[:len(sample['candidates'])]
            scores=self.forward(qids,cache,logits=True)[0,-1,labels]
            if native:base_cache.crop(original_length)
        else:
            scores=[]
            # Average conditional token log-likelihood of each candidate string.
            # Candidate scoring has no answer-letter position to exploit.
            for candidate in sample['candidates']:
                answer_ids=self.encode(' '+candidate)
                ids=torch.cat((qids,answer_ids),dim=1)
                native=not isinstance(base_cache,GPUCache)
                original_length=base_cache.get_seq_length()
                cache=base_cache if native else fork_cache(base_cache)
                logits=self.forward(ids,cache,logits=answer_ids.shape[1]+1)
                if native:base_cache.crop(original_length)
                selected=logits[0,:-1]
                targets=answer_ids[0]
                ll=selected.log_softmax(-1).gather(1,targets[:,None]).mean()
                scores.append(ll)
            scores=torch.stack(scores)
            if protocol=='candidate_calibrated':
                keys=[(sample['question'],candidate) for candidate in sample['candidates']]
                if any(key not in self.priors for key in keys):
                    prior=self.empty_cache()
                    values=self.answer(sample,prior,'candidate')['scores']
                    self.priors.update(zip(keys,values))
                scores=scores-torch.tensor([self.priors[key] for key in keys],device='cuda')
        values=scores.tolist()
        predicted=max(range(len(values)),key=values.__getitem__)
        return dict(prediction=predicted,predicted_text=sample['candidates'][predicted],
                    correct=sample['candidates'][predicted]==sample['answer'],
                    scores=values,rotation=sample.get('rotation',0),protocol=protocol)

    def cache_accounting(self,cache):
        if isinstance(cache,GPUCache):return cache.accounting()
        size=native_cache_bytes(cache)
        storages={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for layer in cache for t in layer}
        return dict(gpu_cache_used_bytes=size,gpu_cache_allocated_bytes=sum(storages.values()),
                    cpu_cache_payload_bytes=0,payload_bytes=size,
                    quantization_metadata_bytes=0,token_mapping_bytes=0,
                    used_tokens=cache.get_seq_length())
