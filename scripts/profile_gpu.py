"""Aggregated own-process operator profile; no large trace or private data."""
import csv,json
import torch
from src.streaming import StreamingModel,prepare_video
from src.policies import Policy
from scripts.resources import ROOT

def main():
    selected=json.loads((ROOT/'configs/gpu_selected.json').read_text())
    model=StreamingModel(selected['model_path'])
    sample=next(s for s in json.loads((ROOT/'data/validation_v3.json').read_text())['samples']
                if s['id']=='action_antonym_107')
    prep=prepare_video(model.adapter,sample,64,True)
    images=model.encode_video(prep)
    output=ROOT/'results/profiles';output.mkdir(exist_ok=True)
    report={}
    for mode in ['native','dynamic']:
        bits=([16]*len(prep.frames) if mode=='native' else Policy(**selected['policy']).assign(prep.features)[0])
        def run():
            cache=model.ingest(prep,bits,mode,images)
            return model.answer(sample,cache,selected['protocol'])
        run();torch.cuda.synchronize()
        activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
        try:
            with torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=True) as prof:
                run();torch.cuda.synchronize()
            rows=[dict(operator=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total,
                       self_device_us=getattr(e,'self_device_time_total',None)) for e in prof.key_averages()]
            with (output/f'{mode}_operators.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            device_observed=any(r['self_device_us'] and r['self_device_us']>0 for r in rows)
            report[mode]=dict(status='success',device_timing_available=device_observed,
                aten_cat_count=sum(r['count'] for r in rows if r['operator']=='aten::cat'),
                aten_mm_count=sum(r['count'] for r in rows if r['operator']=='aten::mm'),
                aten_mean_count=sum(r['count'] for r in rows if r['operator']=='aten::mean'),
                pack_device_us=(sum(r['self_device_us'] or 0 for r in rows if 'pack_rows' in r['operator']) if device_observed else None),
                packed_attention_device_us=(sum(r['self_device_us'] or 0 for r in rows if 'packed_attention' in r['operator']) if device_observed else None))
        except Exception as error:report[mode]=dict(status='unavailable',error=str(error))
    report['scope']='One real 64-frame validation clip, decoder + one candidate-scored QA; profiling is separate from timing'
    (ROOT/'results/profiles/summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
