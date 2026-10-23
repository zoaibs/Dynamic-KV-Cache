"""Disjoint video splits, with exactly balanced answer-position rotations.

The eight previously inspected videos are excluded from both new splits.
No held-out labels are used to choose a policy or answer-scoring protocol.
"""
import json
import zipfile
import random
from pathlib import Path
from scripts.resources import ROOT,sha256,workspace_bytes,CAP

def main():
    all_questions=json.loads((ROOT/'data/action_antonym.json').read_text())
    old=json.loads((ROOT/'data/manifest.json').read_text())
    excluded={s['video'] for s in old['samples']}
    questions=[dict(q,source_index=i) for i,q in enumerate(all_questions) if q['video'] not in excluded]
    random.Random(20261022).shuffle(questions)
    selected={'validation':questions[:24],'test':questions[24:72]}
    archive=ROOT/'data/ssv2_video.zip'
    with zipfile.ZipFile(archive) as z:
        names={Path(n).name:n for n in z.namelist() if n.endswith('.webm')}
        for split,qs in selected.items():
            records=[]
            for q in qs:
                dest=ROOT/'data/videos'/q['video']
                if not dest.exists():dest.write_bytes(z.read(names[q['video']]))
                records.append(dict(id=f'action_antonym_{q["source_index"]:03d}',
                                    path=str(dest.relative_to(ROOT)),sha256=sha256(dest),**q))
            output=dict(dataset='OpenGVLab/MVBench',revision=old['revision'],split=split,
                        seed=20261022,excluded_previously_inspected=sorted(excluded),
                        task='action_antonym',samples=records,
                        answer_position_protocol='All three cyclic option rotations per video; one correct label at each position')
            (ROOT/f'data/{split}_v2.json').write_text(json.dumps(output,indent=2))
    assert not ({q['video'] for q in selected['validation']} & {q['video'] for q in selected['test']})
    print('Prepared 24 validation and 48 held-out test videos, excluding original eight.')

if __name__=='__main__':main()
