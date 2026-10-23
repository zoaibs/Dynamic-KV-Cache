"""Summarize held-out quality and matched GPU end-to-end measurements."""
import csv,json,statistics as st
import numpy as np
from scripts.resources import ROOT

def bootstrap(values,tasks,seed=1729):
    rng=np.random.default_rng(seed);means=np.zeros(10000)
    for task in sorted(set(tasks)):
        group=np.array([v for v,t in zip(values,tasks) if t==task],dtype=float)
        means+=rng.choice(group,(10000,len(group)),replace=True).mean(1)*len(group)/len(values)
    return [float(x) for x in np.quantile(means,[.025,.975])]

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    quality=[json.loads(s) for s in (ROOT/'results/raw/gpu_test_quality.jsonl').read_text().splitlines()]
    timing=[json.loads(s) for s in (ROOT/'results/raw/gpu_timing.jsonl').read_text().splitlines()]
    assert len(quality)==72*7 and len(timing)==180,'Incomplete experiment'
    modes=['native','fused16','4','8','dynamic','original','random']
    native={r['sample_id']:r for r in quality if r['mode']=='native'}
    requested=json.loads((ROOT/'configs/gpu_selected.json').read_text())['quality_frames']
    rows=[]
    for mode in modes:
        rs=[r for r in quality if r['mode']==mode]
        acc=[r['video_mean_accuracy'] for r in rs];tasks=[r['task'] for r in rs]
        delta=[r['video_mean_accuracy']-native[r['sample_id']]['video_mean_accuracy'] for r in rs]
        ci=bootstrap(acc,tasks);dci=bootstrap(delta,tasks)
        used=st.mean(r['cache']['gpu_cache_used_bytes'] for r in rs)
        allocated=st.mean(r['cache']['gpu_cache_allocated_bytes'] for r in rs)
        baseline=st.mean(r['cache']['gpu_cache_used_bytes'] for r in native.values())
        score_errors=[x-y for r in rs for a,b in zip(r['answers'],native[r['sample_id']]['answers'])
                      for x,y in zip(a['scores'],b['scores'])]
        row=dict(mode=mode,unique_videos=len(rs),rotated_questions=sum(len(r['answers']) for r in rs),
                 requested_frames=requested,actual_frames_min=min(r['actual_frames'] for r in rs),
                 actual_frames_max=max(r['actual_frames'] for r in rs),
                 actual_frames_mean=st.mean(r['actual_frames'] for r in rs),
                 accuracy_percent=100*st.mean(acc),accuracy_ci_low=100*ci[0],accuracy_ci_high=100*ci[1],
                 native_delta_pp=100*st.mean(delta),delta_ci_low=100*dci[0],delta_ci_high=100*dci[1],
                 no_video_accuracy_percent=100*st.mean(st.mean(a['correct'] for a in r['no_video']) for r in rs),
                 gpu_cache_used_bytes_mean=used,gpu_cache_allocated_bytes_mean=allocated,
                 used_cache_reduction_percent=100*(1-used/baseline),
                 allocated_cache_reduction_percent=100*(1-allocated/baseline),
                 candidate_score_rmse=float(np.sqrt(np.mean(np.square(score_errors)))),
                 candidate_score_max_abs_delta=max(abs(x) for x in score_errors),
                 answer_agreement_percent=100*st.mean(st.mean(a['predicted_text']==b['predicted_text']
                        for a,b in zip(r['answers'],native[r['sample_id']]['answers'])) for r in rs))
        rows.append(row)
    trows=[]
    for frames in [4,16,32,64]:
        for mode in modes[:5]:
            rs=[r for r in timing if r['mode']==mode and r['requested_frames']==frames]
            means=[st.mean(r['latency_seconds'] for r in rs if r['repeat']==rep) for rep in range(3)]
            fps=[sum(r['actual_frames'] for r in rs if r['repeat']==rep)/
                 sum(r['latency_seconds'] for r in rs if r['repeat']==rep) for rep in range(3)]
            trows.append(dict(mode=mode,requested_frames=frames,timed_requests=len(rs),
                actual_frames_mean=st.mean(r['actual_frames'] for r in rs),
                cached_tokens_mean=st.mean(r['cache']['used_tokens'] for r in rs),
                latency_seconds=st.mean(means),latency_repeat_std=st.stdev(means),
                frames_per_second=st.mean(fps),frames_per_second_repeat_std=st.stdev(fps),
                qa_per_second=st.mean(1/x for x in means),
                decoder_seconds=st.mean(r['decoder_seconds'] for r in rs),
                vision_seconds=st.mean(r['vision_seconds'] for r in rs),
                preparation_seconds=st.mean(r['prepare_seconds'] for r in rs),
                feature_ms=1000*st.mean(r['feature_seconds'] for r in rs),
                feature_wait_ms=1000*st.mean(r.get('feature_wait_seconds',0) for r in rs),
                qa_seconds=st.mean(r['qa_seconds'] for r in rs),
                gpu_peak_allocated_bytes_mean=st.mean(r['gpu_peak_allocated_bytes'] for r in rs),
                gpu_peak_reserved_bytes_mean=st.mean(r['gpu_peak_reserved_bytes'] for r in rs),
                gpu_decoder_peak_allocated_bytes_mean=st.mean(r['gpu_decoder_peak_allocated_bytes'] for r in rs),
                gpu_cache_used_bytes_mean=st.mean(r['cache']['gpu_cache_used_bytes'] for r in rs),
                gpu_cache_allocated_bytes_mean=st.mean(r['cache']['gpu_cache_allocated_bytes'] for r in rs)))
    for r in trows:
        baseline=next(t for t in trows if t['mode']=='native' and t['requested_frames']==r['requested_frames'])
        r['throughput_change_percent']=100*(r['frames_per_second']/baseline['frames_per_second']-1)
    result=dict(quality=rows,timing=trows,quality_unique_videos=72,
                short_quality_videos=[dict(sample_id=r['sample_id'],actual_frames=r['actual_frames'])
                                      for r in native.values() if r['actual_frames']<requested],
                statistical_unit='Video; answer rotations are clustered within a video',
                accuracy_uncertainty='95% task-stratified video bootstrap, 10000 draws',
                timing_scope='Decode + preprocessing + vision encoding + decoder frame appends + one complete candidate-scored QA; model loading and JIT warmups excluded',
                timing_uncertainty='Sample SD of three repetition-level means over the same three validation clips',
                feature_scheduling='CPU feature work overlaps CUDA vision; residual feature wait is included in preparation time. Feature runtime is not additive to total latency.',
                attribution='Fused16 controls for combined projections, normalization execution and packed-attention implementation independently of low-bit storage',
                cpu_v1_comparable=False)
    (ROOT/'results/gpu_metrics.json').write_text(json.dumps(result,indent=2))
    write_csv(ROOT/'results/gpu_quality.csv',rows);write_csv(ROOT/'results/gpu_timing.csv',trows)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
