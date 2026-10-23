"""One bounded local-video QA request using the same measured inference path."""
import argparse
import json
import torch
from benchmarks.evaluate import run_sample, VideoModel
from scripts.resources import ROOT

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sample',type=int,default=0)
    p.add_argument('--mode',choices=['native','fused16','8','4','dynamic','original','random'],default='dynamic')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--frames',type=int)
    args=p.parse_args()
    if args.frames is not None and not 1<=args.frames<=96:p.error('--frames must be between 1 and 96')
    samples=json.loads((ROOT/'data/manifest.json').read_text())['samples']
    if not 0<=args.sample<len(samples):p.error(f'--sample must be between 0 and {len(samples)-1}')
    sample=samples[args.sample]
    if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()):
        from src.streaming import StreamingModel
        from benchmarks.gpu_evaluate import timed_request
        selected=json.loads((ROOT/'configs/gpu_selected.json').read_text())
        if not selected.get('frozen_before_test'):raise RuntimeError('GPU policy selection has not finished')
        model=StreamingModel(selected['model_path'])
        print(json.dumps(timed_request(model,selected,sample,args.frames or selected['quality_frames'],args.mode),indent=2))
        return
    if args.mode not in ('native','4','8','dynamic'):raise ValueError('This mode requires the GPU runtime')
    config=json.loads((ROOT/'configs/default.json').read_text())
    if args.frames:config['frames']=args.frames
    print(json.dumps(run_sample(VideoModel(config),sample,args.mode,config),indent=2))

if __name__=='__main__':
    main()
