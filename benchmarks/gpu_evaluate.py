"""Frozen-policy held-out evaluation and synchronized real-video GPU timings."""
import argparse,json,time,random,gc,hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from src.streaming import StreamingModel,prepare_video,rotations,score_prepared
from src.policies import Policy
from scripts.resources import ROOT,sha256

MODES=['native','fused16','4','8','dynamic','original','random']

def frame_map(model,prepared,bits):
    cursor=model.prefix_ids.shape[1];records=[]
    for i,(frame,inputs,bit) in enumerate(zip(prepared.frames,prepared.inputs,bits)):
        n=inputs['input_ids'].shape[1]
        records.append(dict(frame_id=i,start=cursor,end=cursor+n,bits=bit,
            source_index=frame['index'],seconds=frame['seconds'],
            visual_tokens=int((inputs['input_ids']==model.model.config.image_token_id).sum())))
        cursor+=n
    return records

def bits_for(mode,prepared,selected):
    n=len(prepared.frames)
    if mode in ('native','fused16'):return [16]*n,[]
    if mode in ('4','8'):return [int(mode)]*n,[]
    if mode=='random':
        seed=20261022+int.from_bytes(hashlib.sha256(prepared.sample['id'].encode()).digest()[:8],'little')
        return random.Random(seed).choices([4,8,16],weights=selected['random_probabilities'],k=n),[]
    policy=Policy(**selected['policy']) if mode=='dynamic' else Policy('original')
    return policy.assign(prepared.features)

def warmup(model,selected,sample,frames,modes):
    prep=prepare_video(model.adapter,sample,frames,True)
    images=model.encode_video(prep)
    for mode in modes:
        bits,_=bits_for(mode,prep,selected)
        cache=model.ingest(prep,bits,mode,images,block_m=selected['block_m'],block_n=selected['block_n'])
        model.answer(sample,cache,selected['protocol'])
    torch.cuda.synchronize()

def evaluate_quality(model,selected):
    if not selected.get('frozen_before_test'):raise RuntimeError('Freeze policy before opening held-out cases')
    path=ROOT/'data/test_v3.json'
    if sha256(path)!=selected['test_manifest_sha256']:raise RuntimeError('Held-out manifest changed')
    samples=json.loads(path.read_text())['samples']
    warm_sample=json.loads((ROOT/'data/validation_v3.json').read_text())['samples'][0]
    warmup(model,selected,warm_sample,selected['quality_frames'],MODES)
    with (ROOT/'results/raw/gpu_test_quality.jsonl').open('w') as log:
        for i,sample in enumerate(samples):
            prepared=prepare_video(model.adapter,sample,selected['quality_frames'],True)
            images=model.encode_video(prepared)
            empty=model.empty_cache()
            no_video=[model.answer(s,empty,selected['protocol']) for s in rotations(sample)]
            for mode in MODES:
                bits,scores=bits_for(mode,prepared,selected)
                cache=model.ingest(prepared,bits,mode,images,block_m=selected['block_m'],block_n=selected['block_n'])
                accounting=model.cache_accounting(cache)
                answers=[model.answer(s,cache,selected['protocol']) for s in rotations(sample)]
                row=dict(sample_id=sample['id'],task=sample['task'],mode=mode,
                         actual_frames=len(prepared.frames),answers=answers,no_video=no_video,
                         video_mean_accuracy=float(np.mean([a['correct'] for a in answers])),
                         cache=accounting,bits=bits,information_scores=scores)
                row['raw_features']=prepared.features
                row['frame_map']=frame_map(model,prepared,bits)
                log.write(json.dumps(row)+'\n');log.flush()
            print(f'Held-out video {i+1}/{len(samples)}',flush=True)

def timed_request(model,selected,sample,frames,mode):
    gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    weights=selected['policy']['weights'] if mode=='dynamic' else None
    prepared=prepare_video(model.adapter,sample,frames,False)
    prepare_seconds=time.perf_counter()-start
    pool=ThreadPoolExecutor(max_workers=1) if mode in ('dynamic','original') else None
    future=pool.submit(score_prepared,prepared,model.adapter.config['score'],weights) if pool else None
    try:
        t=time.perf_counter();images=model.encode_video(prepared);torch.cuda.synchronize()
        vision_seconds=time.perf_counter()-t
        t=time.perf_counter()
        if future:prepared.features,prepared.feature_seconds=future.result()
        feature_wait_seconds=time.perf_counter()-t if future else 0.
    finally:
        if pool:pool.shutdown(wait=True)
    # Feature runtime overlaps vision. Only its residual wait is additive.
    prepare_seconds+=feature_wait_seconds
    bits,scores=bits_for(mode,prepared,selected)
    vision_peak_allocated=torch.cuda.max_memory_allocated()
    vision_peak_reserved=torch.cuda.max_memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    t=time.perf_counter()
    cache=model.ingest(prepared,bits,mode,images,block_m=selected['block_m'],block_n=selected['block_n'])
    torch.cuda.synchronize();decoder_seconds=time.perf_counter()-t
    accounting=model.cache_accounting(cache)
    t=time.perf_counter();answer=model.answer(sample,cache,selected['protocol']);torch.cuda.synchronize()
    qa_seconds=time.perf_counter()-t
    elapsed=time.perf_counter()-start
    return dict(sample_id=sample['id'],task=sample['task'],mode=mode,
        requested_frames=frames,actual_frames=len(prepared.frames),latency_seconds=elapsed,
        prepare_seconds=prepare_seconds,vision_seconds=vision_seconds,
        feature_seconds=prepared.feature_seconds,decode_seconds=prepared.decode_seconds,
        feature_wait_seconds=feature_wait_seconds,features_overlap_vision=bool(future),
        preprocessing_seconds=prepared.preprocessing_seconds,
        decoder_seconds=decoder_seconds,qa_seconds=qa_seconds,
        frames_per_second=len(prepared.frames)/elapsed,qa_per_second=1/elapsed,
        gpu_peak_allocated_bytes=max(vision_peak_allocated,torch.cuda.max_memory_allocated()),
        gpu_peak_reserved_bytes=max(vision_peak_reserved,torch.cuda.max_memory_reserved()),
        gpu_decoder_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu_decoder_peak_reserved_bytes=torch.cuda.max_memory_reserved(),cache=accounting,
        answer=answer,bits=bits,information_scores=scores,computed_features=weights,
        raw_features=prepared.features,frame_map=frame_map(model,prepared,bits))

def evaluate_timing(model,selected):
    samples=json.loads((ROOT/'data/validation_v3.json').read_text())['samples']
    # Chosen by clip availability and task, never by correctness. Three clips
    # include two human-action clips and a distinct visual action task.
    chosen=[next(s for s in samples if s['id']==x) for x in ['action_antonym_107','action_antonym_021']]
    chosen.append(next(s for s in samples if s['task']=='fine_grained_action'))
    modes=['native','fused16','4','8','dynamic']
    n=0
    with (ROOT/'results/raw/gpu_timing.jsonl').open('w') as log:
        for frames in selected['performance_frames']:
            warmup(model,selected,chosen[0],frames,modes)
            for repeat in range(3):
                for i,sample in enumerate(chosen):
                    offset=(repeat*len(chosen)+i)%len(modes)
                    order=modes[offset:]+modes[:offset]
                    for mode in order:
                        row=timed_request(model,selected,sample,frames,mode)
                        row['repeat']=repeat
                        log.write(json.dumps(row)+'\n');log.flush();n+=1
                        print(f'Timing {n}/180: frames={frames} {mode} {row["latency_seconds"]:.3f}s',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['quality','timing'],required=True)
    args=parser.parse_args()
    selected=json.loads((ROOT/'configs/gpu_selected.json').read_text())
    snapshot=dict(selected=selected,selected_sha256=sha256(ROOT/'configs/gpu_selected.json'),
                  phase=args.phase,scope='GPU, identical checkpoint and inputs within each comparison',
                  feature_scheduling='CPU features overlap CUDA vision; residual wait is preparation time' if args.phase=='timing' else 'Synchronous shared quality preparation',
                  numerical_control='fused16 shares attention, combined projections, and normalization execution, without low-bit KV storage')
    (ROOT/f'results/raw/gpu_{args.phase}_config.json').write_text(json.dumps(snapshot,indent=2))
    model=StreamingModel(selected['model_path'])
    (evaluate_quality if args.phase=='quality' else evaluate_timing)(model,selected)

if __name__=='__main__':main()
