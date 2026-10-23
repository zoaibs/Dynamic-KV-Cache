"""Choose the checkpoint and answer protocol on validation, never held-out data."""
import json
from datetime import datetime,timezone
from scripts.resources import ROOT,sha256

def main():
    candidates=[]
    for name,path,api in [('256m_v3','models/smolvlm2','docs/model_api.json'),
                          ('500m_v3','models/smolvlm2_500m','docs/model_500m_api.json')]:
        metrics=json.loads((ROOT/f'results/accuracy_audit_{name}.json').read_text())
        for protocol,values in metrics.items():
            candidates.append(dict(model_path=path,api=api,protocol=protocol,**values,
                visual_gain=values['accuracy']-values['no_video_accuracy']))
    # Accuracy first; break ties with video contribution over no-video control.
    chosen=max(candidates,key=lambda r:(r['accuracy'],r['visual_gain']))
    api=json.loads((ROOT/chosen['api']).read_text(encoding='utf-8-sig'))
    selected=dict(model_path=chosen['model_path'],checkpoint=api['id'],revision=api['sha'],
        protocol=chosen['protocol'],quality_frames=16,performance_frames=[4,16,32,64],
        validation_videos=36,test_videos=72,validation_manifest_sha256=sha256(ROOT/'data/validation_v3.json'),
        selection_rule='Maximum validation video-mean accuracy; ties: improvement over no-video control',
        validation_choice=chosen,candidates=candidates,
        selected_at_utc=datetime.now(timezone.utc).isoformat(),block_m=16,block_n=64)
    (ROOT/'configs/gpu_selected.json').write_text(json.dumps(selected,indent=2))
    print(json.dumps(selected,indent=2))

if __name__=='__main__':main()
