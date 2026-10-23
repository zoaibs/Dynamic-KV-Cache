"""Controlled real-checkpoint/video experiments; JSONL written after each sample."""
import argparse
import gc
import json
import math
import statistics
import threading
import time
import psutil
import torch
from scripts.resources import ROOT
from src.model_adapter import VideoModel, decode_video, native_cache_bytes
from src.cache_manager import MixedPrecisionCache
from src.information_score import InformationScorer
from src.precision_scheduler import precision_for_score

MODES = ['native', '8', '4', 'dynamic']

class MemorySampler:
    """10 ms process RSS sample; includes weights, interpreter and intermediates."""
    def __init__(self):
        self.process = psutil.Process()
        self.start_rss = self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
    def run(self):
        while not self.stop.wait(.01):
            self.peak = max(self.peak, self.process.memory_info().rss)
    def __enter__(self):
        self.thread.start()
        return self
    def __exit__(self,*args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join(timeout=2)

def sync(model):
    if model.device.type == 'cuda':
        torch.cuda.synchronize(model.device)

def run_sample(model, sample, mode, config):
    gc.collect()
    sync(model)
    if model.device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(model.device)
    with MemorySampler() as memory:
        t0=time.perf_counter()
        frames=decode_video(sample['path'],config['frames'])
        decode_seconds=time.perf_counter()-t0
        cache = None if mode=='native' else MixedPrecisionCache(config['group_size'])
        cache=model.begin(cache)
        scorer=InformationScorer(config['score']) if mode=='dynamic' else None
        score_seconds=0.0
        trace=[]
        frame_spans=[]
        for i, frame in enumerate(frames):
            if scorer:
                t=time.perf_counter()
                info=scorer(frame['bgr'])
                bits=precision_for_score(info.score,config['score']['thresholds'])
                score_seconds+=time.perf_counter()-t
            else:
                info=None
                bits=16 if mode=='native' else int(mode)
            if mode!='native':
                cache.select(bits,i)
            mapping=model.append(frame,cache)
            frame_spans.append(mapping)
            trace.append(dict(frame_id=i,bits=bits,**mapping,
                              **(info.to_dict() if info else {})))
        if mode=='native':
            size=native_cache_bytes(cache)
            accounting=dict(cpu_cache_bytes=size,cpu_payload_bytes=size,cpu_metadata_bytes=0,
                            gpu_cache_bytes=0,disk_cache_bytes=0)
            frame_tokens=sum(x['end']-x['start'] for x in frame_spans)
            frame_bytes=size*frame_tokens/cache.get_seq_length()
        else:
            accounting=cache.accounting()
            frame_bytes=cache.accounting(frame_only=True)['cpu_cache_bytes']
            cache.select(16,-1)
        vision_before=model.vision_calls
        answer=model.answer(sample,cache)
        assert model.vision_calls==vision_before
        sync(model)
        elapsed=time.perf_counter()-t0
        pack_seconds=0.0 if mode=='native' else cache.pack_seconds
        unpack_seconds=0.0 if mode=='native' else cache.unpack_seconds
        retrievals=None if mode=='native' else cache.retrievals
    return dict(mode=mode,sample_id=sample['id'],frames=len(frames),
                latency_seconds=elapsed,qa_per_second=1/elapsed,frames_per_second=len(frames)/elapsed,
                decode_seconds=decode_seconds,score_seconds=score_seconds,
                pack_seconds=pack_seconds,unpack_seconds=unpack_seconds,
                process_rss_start_bytes=memory.start_rss,process_peak_rss_bytes=memory.peak,
                gpu_peak_allocated_bytes=(torch.cuda.max_memory_allocated(model.device)
                                          if model.device.type=='cuda' else None),
                gpu_peak_reserved_bytes=(torch.cuda.max_memory_reserved(model.device)
                                         if model.device.type=='cuda' else None),
                frame_cache_bytes=int(frame_bytes),cache_boundary='after last frame, before QA',
                **accounting,answer=answer,trace=trace,retrievals=retrievals)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/default.json')
    p.add_argument('--samples',type=int)
    p.add_argument('--repeats',type=int)
    p.add_argument('--output',default='results/raw/benchmark.jsonl')
    args=p.parse_args()
    config=json.loads((ROOT/args.config).read_text())
    if args.samples is not None: config['samples']=args.samples
    if args.repeats is not None: config['repeats']=args.repeats
    if not 1<=config['samples']<=8 or not 1<=config['repeats']<=5:
        raise ValueError('Local sample/repeat caps: 8 samples, 5 repeats')
    samples=json.loads((ROOT/'data/manifest.json').read_text())['samples'][:config['samples']]
    model=VideoModel(config)
    print('Warmup: one complete request per configuration',flush=True)
    for mode in MODES:
        run_sample(model,samples[0],mode,config)
    (ROOT/'results/raw/benchmark_config.json').write_text(json.dumps(config,indent=2))
    output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT/'results'):
        raise ValueError('Benchmark logs must be under results')
    n=0
    with output.open('w') as log:
        for repeat in range(config['repeats']):
            for i,sample in enumerate(samples):
                offset=(repeat+i)%len(MODES)
                order=MODES[offset:]+MODES[:offset]
                for mode in order:
                    record=run_sample(model,sample,mode,config)
                    record['repeat']=repeat
                    log.write(json.dumps(record)+'\n')
                    log.flush()
                    n+=1
                    print(f'{n}/{len(samples)*config["repeats"]*4}: '
                          f'{mode} {sample["id"]}, {record["latency_seconds"]:.3f}s '
                          f'correct={record["answer"]["correct"]}',flush=True)

if __name__=='__main__':
    main()
