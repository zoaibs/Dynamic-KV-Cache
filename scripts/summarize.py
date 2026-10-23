"""Aggregate observed data, with uncertainty across three complete repetitions."""
import csv
import json
import math
import statistics as stats
from scripts.resources import ROOT

MODES=['native','8','4','dynamic']

def wilson(correct,n):
    z=1.95996398454
    phat=correct/n
    denominator=1+z*z/n
    center=(phat+z*z/(2*n))/denominator
    half=z*math.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/denominator
    return [100*(center-half),100*(center+half)]

def main():
    records=[json.loads(x) for x in (ROOT/'results/raw/benchmark.jsonl').read_text().splitlines()]
    expected=json.loads((ROOT/'results/raw/benchmark_config.json').read_text())
    if len(records)!=4*expected['samples']*expected['repeats']:
        raise ValueError('Incomplete experiment; refusing to create final summary')
    rows=[]
    predictions=[]
    for mode in MODES:
        rs=[r for r in records if r['mode']==mode]
        unique=[r for r in rs if r['repeat']==0]
        for sample in unique:
            other=[r for r in rs if r['sample_id']==sample['sample_id']]
            if len({tuple(r['answer']['logits']) for r in other}) != 1:
                raise ValueError('Nondeterministic predictions; inspect before aggregating accuracy')
        correct=sum(r['answer']['correct'] for r in unique)
        times=[]
        throughputs=[]
        for repeat in sorted({r['repeat'] for r in rs}):
            group=[r for r in rs if r['repeat']==repeat]
            total=sum(r['latency_seconds'] for r in group)
            times.append(total/len(group))
            throughputs.append(sum(r['frames'] for r in group)/total)
        row=dict(mode=mode,unique_samples=len(unique),timed_requests=len(rs),
                 correct=correct,accuracy_percent=100*correct/len(unique),
                 accuracy_wilson95_low=wilson(correct,len(unique))[0],
                 accuracy_wilson95_high=wilson(correct,len(unique))[1],
                 cpu_cache_bytes_mean=stats.mean(r['cpu_cache_bytes'] for r in unique),
                 cpu_payload_bytes_mean=stats.mean(r['cpu_payload_bytes'] for r in unique),
                 cpu_metadata_bytes_mean=stats.mean(r['cpu_metadata_bytes'] for r in unique),
                 frame_cache_bytes_mean=stats.mean(r['frame_cache_bytes'] for r in unique),
                 latency_seconds_mean=stats.mean(times),
                 latency_repeat_std=stats.stdev(times) if len(times)>1 else 0,
                 frames_per_second=stats.mean(throughputs),
                 frames_per_second_repeat_std=stats.stdev(throughputs) if len(throughputs)>1 else 0,
                 qa_per_second=stats.mean(1/t for t in times),
                 process_peak_rss_bytes_mean=stats.mean(r['process_peak_rss_bytes'] for r in rs),
                 gpu_peak_allocated_bytes=None,gpu_peak_reserved_bytes=None,
                 gpu_cache_bytes=0,disk_cache_bytes=0,
                 score_ms_mean=1000*stats.mean(r['score_seconds'] for r in rs),
                 pack_ms_mean=1000*stats.mean(r['pack_seconds'] for r in rs),
                 unpack_ms_mean=1000*stats.mean(r['unpack_seconds'] for r in rs))
        rows.append(row)
        for r in unique:
            predictions.append(dict(mode=mode,sample_id=r['sample_id'],**r['answer']))
    base=rows[0]
    for row in rows:
        row['cache_reduction_percent']=100*(1-row['cpu_cache_bytes_mean']/base['cpu_cache_bytes_mean'])
        row['accuracy_delta_pp']=row['accuracy_percent']-base['accuracy_percent']
        row['throughput_change_percent']=100*(row['frames_per_second']/base['frames_per_second']-1)
    with (ROOT/'results/metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    timeline=[]
    for r in records:
        if r['mode']=='dynamic' and r['repeat']==0:
            timeline.extend(dict(sample_id=r['sample_id'],**frame) for frame in r['trace'])
    with (ROOT/'results/timeline.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(timeline[0]))
        writer.writeheader();writer.writerows(timeline)
    (ROOT/'results/metrics.json').write_text(json.dumps(dict(metrics=rows,
        accuracy_unit='8 unique videos, not 24 independent observations',
        timing_uncertainty='sample standard deviation of three repetition-level means',
        cache_boundary='after last frame before QA; packed payload plus binary metadata; excludes Python bookkeeping',
        gpu_measurement_status='unavailable: CPU-only experiment',
        memory_scope='process RSS includes weights/interpreter/transient attention buffers; 10 ms sampling',
        dynamic_precision_counts={str(b):sum(t['bits']==b for t in timeline) for b in [4,8,16]},
        predictions=predictions),indent=2))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':
    main()
