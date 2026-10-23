"""Validation-only protocol audit, with all answer-order rotations and no-video control."""
import argparse,json,time
import torch
from transformers.cache_utils import DynamicCache
from src.streaming import StreamingModel,prepare_video,rotations
from scripts.resources import ROOT

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='models/smolvlm2')
    parser.add_argument('--frames',type=int,default=8)
    parser.add_argument('--samples',type=int,default=24)
    parser.add_argument('--name',default='256m')
    parser.add_argument('--manifest',default='data/validation_v2.json')
    args=parser.parse_args()
    model=StreamingModel(args.model)
    if 'test' in args.manifest:raise ValueError('Protocol selection may use validation only')
    samples=json.loads((ROOT/args.manifest).read_text())['samples'][:args.samples]
    records=[]
    out=ROOT/f'results/raw/accuracy_audit_{args.name}.jsonl'
    with out.open('w') as log:
        for i,s in enumerate(samples):
            prepared=prepare_video(model.adapter,s,args.frames,False)
            cache=model.ingest(prepared,[16]*len(prepared.frames),'native')
            empty=DynamicCache();model.forward(model.prefix_ids,empty)
            for protocol in ['label','label_space','label_leading','candidate','candidate_calibrated']:
                orderings=list(rotations(s))
                for variant in orderings:
                    answer=model.answer(variant,cache,protocol)
                    prior=model.answer(variant,empty,protocol)
                    record=dict(sample_id=s['id'],protocol=protocol,**{k:v for k,v in answer.items() if k!='protocol'},
                                no_video=prior)
                    records.append(record);log.write(json.dumps(record)+'\n');log.flush()
            print(f'{args.name} audit {i+1}/{len(samples)}',flush=True)
    summary={}
    for protocol in ['label','label_space','label_leading','candidate','candidate_calibrated']:
        rs=[r for r in records if r['protocol']==protocol]
        grouped=[[r for r in rs if r['sample_id']==s['id']] for s in samples]
        summary[protocol]=dict(accuracy=sum(sum(r['correct'] for r in g)/len(g) for g in grouped)/len(grouped),
                              no_video_accuracy=sum(sum(r['no_video']['correct'] for r in g)/len(g) for g in grouped)/len(grouped),
                              observations=len(rs),unique_videos=len(samples))
    (ROOT/f'results/accuracy_audit_{args.name}.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
