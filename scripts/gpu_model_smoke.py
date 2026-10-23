"""Same-checkpoint real-video GPU correctness and initial timing."""
import json,time
import torch
from src.streaming import StreamingModel,prepare_video,rotations
from scripts.resources import ROOT

def main():
    model=StreamingModel()
    s=json.loads((ROOT/'data/manifest.json').read_text())['samples'][0]
    prepared=prepare_video(model.adapter,s,4)
    encoded=model.encode_video(prepared)
    records=[]
    for mode,bits in [('native',[16]*4),('native_gqa',[16]*4),('fused16',[16]*4),('8',[8]*4),('4',[4]*4),('dynamic',[4,8,16,4])]:
        torch.cuda.synchronize();start=time.perf_counter()
        cache=model.ingest(prepared,bits,mode,encoded=encoded)
        for protocol in ['label','label_space','candidate']:
            result=model.answer(s,cache,protocol)
            records.append(dict(mode=mode,**result))
            print(mode,protocol,result,flush=True)
        torch.cuda.synchronize()
        print('mode seconds',time.perf_counter()-start,model.cache_accounting(cache),flush=True)
    by={r['mode']:r for r in records if r['protocol']=='label'}
    diff=max(abs(a-b) for a,b in zip(by['native']['scores'],by['fused16']['scores']))
    assert diff<.3,('BF16 fused/native real model mismatch',diff)
    out=dict(status='passed',native_fused16_max_score_delta=diff,records=records)
    (ROOT/'results/raw/gpu_model_smoke.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__':main()
