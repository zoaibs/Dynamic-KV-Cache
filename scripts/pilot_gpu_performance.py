"""Real-model decoder timing pilot: images preencoded and shared across modes."""
import json,time,statistics
import torch
from src.streaming import StreamingModel,prepare_video
from src.policies import Policy
from scripts.resources import ROOT

def main():
    model=StreamingModel('models/smolvlm2_500m')
    samples=json.loads((ROOT/'data/validation_v2.json').read_text())['samples']
    sample=next(s for s in samples if s['id']=='action_antonym_107')
    records=[]
    for frames in [4,16,64]:
        prep=prepare_video(model.adapter,sample,frames)
        encoded=model.encode_video(prep)
        for mode in ['native','native_gqa','fused16','4','8','dynamic']:
            bits=Policy('original').assign(prep.features)[0] if mode=='dynamic' else [int(mode) if mode in ('4','8') else 16]*len(prep.frames)
            for repeat in range(4):
                torch.cuda.synchronize();t=time.perf_counter()
                cache=model.ingest(prep,bits,mode,encoded)
                torch.cuda.synchronize();prefill=time.perf_counter()-t
                t=time.perf_counter();answer=model.answer(sample,cache,'candidate')
                torch.cuda.synchronize();qa=time.perf_counter()-t
                if repeat:
                    row=dict(mode=mode,requested_frames=frames,actual_frames=len(prep.frames),
                             repeat=repeat,prefill_seconds=prefill,qa_seconds=qa,
                             total_seconds=prefill+qa,cache=model.cache_accounting(cache),answer=answer)
                    records.append(row)
                    print(json.dumps(row),flush=True)
    (ROOT/'results/raw/gpu_performance_pilot.json').write_text(json.dumps(dict(
        scope='Real 500M model, decoder prefill plus candidate QA; preencoded video features, no preprocessing/vision time',
        records=records),indent=2))

if __name__=='__main__':main()
