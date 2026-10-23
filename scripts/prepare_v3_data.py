"""Add an independent MVBench visual action task; keep video-level splits disjoint."""
import json,random,zipfile
from pathlib import Path
from scripts.resources import ROOT,sha256

def main():
    qs=json.loads((ROOT/'data/fine_grained_action.json').read_text())
    qs=[dict(q,source_index=i) for i,q in enumerate(qs)]
    random.Random(20261023).shuffle(qs)
    selected={'validation':qs[:12],'test':qs[12:36]}
    with zipfile.ZipFile(ROOT/'data/moments.zip') as z:
        for split,questions in selected.items():
            output=json.loads((ROOT/f'data/{split}_v2.json').read_text())
            for q in questions:
                matches=[n for n in z.namelist() if n.endswith(q['video'])]
                if len(matches)!=1:raise RuntimeError(f'Unmatched public clip: {q["video"]}, {matches}')
                dest=ROOT/'data/videos/moments'/q['video']
                if not dest.resolve().is_relative_to((ROOT/'data/videos').resolve()):
                    raise ValueError('Unsafe public annotation path')
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(z.read(matches[0]))
                output['samples'].append(dict(id=f'fine_grained_{q["source_index"]:03d}',
                    path=str(dest.relative_to(ROOT)),sha256=sha256(dest),task='fine_grained_action',**q))
            for s in output['samples']:s.setdefault('task','action_antonym')
            output['tasks']=['action_antonym','fine_grained_action']
            output['answer_position_protocol']='All cyclic rotations of the three or four options'
            (ROOT/f'data/{split}_v3.json').write_text(json.dumps(output,indent=2))
    a=json.loads((ROOT/'data/validation_v3.json').read_text())['samples']
    b=json.loads((ROOT/'data/test_v3.json').read_text())['samples']
    assert not ({s['path'] for s in a}&{s['path'] for s in b})
    assert not ({s['sha256'] for s in a}&{s['sha256'] for s in b})
    print('Prepared',len(a),'validation and',len(b),'held-out test videos from two tasks.')

if __name__=='__main__':main()
