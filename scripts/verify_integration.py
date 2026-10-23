"""Integration evidence from the real local checkpoint, never mock outputs."""
import json
import torch
from transformers.cache_utils import DynamicCache
from scripts.resources import ROOT
from src.model_adapter import VideoModel, decode_video
from src.cache_manager import MixedPrecisionCache
from src.information_score import InformationScorer
from src.precision_scheduler import precision_for_score

def main():
    config=json.loads((ROOT/'configs/default.json').read_text())
    sample=json.loads((ROOT/'data/manifest.json').read_text())['samples'][0]
    model=VideoModel(config)
    frames=decode_video(sample['path'],config['frames'])
    evidence={}
    native=model.begin()
    for frame in frames:
        model.append(frame,native)
    native_output=model.answer(sample,native)
    evidence['native']=native_output
    for mode in ['16','8','4','dynamic','mixed_integration_test']:
        cache=MixedPrecisionCache(config['group_size'])
        scorer=InformationScorer(config['score'])
        model.begin(cache)
        trace=[]
        for i,frame in enumerate(frames):
            info=scorer(frame['bgr'])
            if mode=='mixed_integration_test':
                bits=[4,8,16,4][i]
            else:
                bits=precision_for_score(info.score,config['score']['thresholds']) if mode=='dynamic' else int(mode)
            cache.select(bits,i)
            mapping=model.append(frame,cache)
            trace.append(dict(bits=bits,**info.to_dict(),**mapping))
        calls=model.vision_calls
        cache.select(16,-1)
        answer=model.answer(sample,cache)
        assert model.vision_calls==calls
        assert all(torch.isfinite(torch.tensor(answer['logits'])))
        if mode=='16':
            assert answer['logits']==native_output['logits'], 'Native BF16 mismatch'
        evidence[mode]=dict(answer=answer,cache=cache.accounting(),trace=trace,
                            retrievals=cache.retrievals,qa_vision_calls=0)
    (ROOT/'results/raw/integration.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence,indent=2))

if __name__=='__main__':
    main()
