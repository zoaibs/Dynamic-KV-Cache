"""Bounded validation search using actual causal packed-cache inference."""
import json,time,statistics
import numpy as np
import torch
from datetime import datetime,timezone
from src.streaming import StreamingModel,prepare_video,rotations
from src.policies import Policy,candidates,fit_scales
from scripts.resources import ROOT,sha256

def main():
    selected=json.loads((ROOT/'configs/gpu_selected.json').read_text())
    model=StreamingModel(selected['model_path'])
    samples=json.loads((ROOT/'data/validation_v3.json').read_text())['samples']
    prepared=[prepare_video(model.adapter,s,selected['quality_frames']) for s in samples]
    scales=fit_scales([p.features for p in prepared])
    policies=[Policy('uniform4',fixed=4),Policy('uniform8',fixed=8),Policy('fused16',fixed=16),*candidates(scales)]
    encoded=[model.encode_video(p) for p in prepared]
    protocol=selected['protocol']
    records=[]
    # All candidates get an excluded warmup on the same validation sample.
    for policy in policies:
        bits,_=policy.assign(prepared[0].features)
        cache=model.ingest(prepared[0],bits,'dynamic',encoded[0])
        model.answer(samples[0],cache,protocol)
    torch.cuda.synchronize()
    with (ROOT/'results/raw/scheduler_validation.jsonl').open('w') as log:
        for i,(sample,prep,images) in enumerate(zip(samples,prepared,encoded)):
            # Native reference for fidelity and native task quality, same inputs.
            baseline=model.ingest(prep,[16]*len(prep.frames),'native',images)
            base=model.answer(sample,baseline,protocol)
            base_storage=model.cache_accounting(baseline)['gpu_cache_used_bytes']
            for policy in policies:
                bits,scores=policy.assign(prep.features)
                torch.cuda.synchronize();start=time.perf_counter()
                cache=model.ingest(prep,bits,'dynamic',images,
                                   block_m=selected['block_m'],block_n=selected['block_n'])
                result=model.answer(sample,cache,protocol)
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                a=model.cache_accounting(cache)
                row=dict(policy=policy.name,sample_id=sample['id'],task=sample['task'],
                         correct=result['correct'],native_correct=base['correct'],
                         prediction=result['predicted_text'],native_prediction=base['predicted_text'],
                         score_delta_l2=float(np.linalg.norm(np.array(result['scores'])-np.array(base['scores']))),
                         gpu_cache_used_bytes=a['gpu_cache_used_bytes'],native_cache_bytes=base_storage,
                         gpu_cache_allocated_bytes=a['gpu_cache_allocated_bytes'],
                         decoder_and_qa_seconds=seconds,bits=bits,information_scores=scores)
                records.append(row);log.write(json.dumps(row)+'\n');log.flush()
            print(f'Validation policies: video {i+1}/{len(samples)}',flush=True)
    summary=[]
    for policy in policies:
        rs=[r for r in records if r['policy']==policy.name]
        summary.append(dict(policy=policy.to_dict(),accuracy=statistics.mean(r['correct'] for r in rs),
            native_accuracy=statistics.mean(r['native_correct'] for r in rs),
            agreement=statistics.mean(r['prediction']==r['native_prediction'] for r in rs),
            cache_reduction=1-sum(r['gpu_cache_used_bytes'] for r in rs)/sum(r['native_cache_bytes'] for r in rs),
            allocated_cache_reduction=1-sum(r['gpu_cache_allocated_bytes'] for r in rs)/sum(r['native_cache_bytes'] for r in rs),
            mean_decoder_qa_seconds=statistics.mean(r['decoder_and_qa_seconds'] for r in rs),
            score_delta_l2=statistics.mean(r['score_delta_l2'] for r in rs)))
    eligible=[r for r in summary if not r['policy']['fixed'] and r['cache_reduction']>=.25
              and r['allocated_cache_reduction']>=.25]
    chosen=max(eligible,key=lambda r:(r['accuracy'],r['cache_reduction'],-r['score_delta_l2']))
    output=dict(validation_only=True,unique_videos=len(samples),tested_policies=len(policies),
                normalization_scales_fitted_on_validation=scales,summary=summary,selected=chosen,
                rule='Among dynamic policies with >=25% used and allocated cache reduction: maximize validation accuracy, then cache reduction, then native-score fidelity')
    (ROOT/'results/scheduler_validation.json').write_text(json.dumps(output,indent=2))
    selected['policy']=chosen['policy']
    selected['policy_selection']=chosen
    chosen_rows=[r for r in records if r['policy']==chosen['policy']['name']]
    bit_counts={b:sum(row['bits'].count(b) for row in chosen_rows) for b in [4,8,16]}
    selected['random_probabilities']=[bit_counts[b]/sum(bit_counts.values()) for b in [4,8,16]]
    selected['test_manifest_sha256']=sha256(ROOT/'data/test_v3.json')
    selected['frozen_before_test']=True
    selected['policy_frozen_at_utc']=datetime.now(timezone.utc).isoformat()
    (ROOT/'configs/gpu_selected.json').write_text(json.dumps(selected,indent=2))
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
