"""Deterministic small MVBench subset; extracts only requested video files."""
import json
import zipfile
from pathlib import Path
from scripts.resources import ROOT, sha256, workspace_bytes, CAP

def main(count=8):
    questions = json.loads((ROOT / 'data/action_antonym.json').read_text())[:count]
    archive = ROOT / 'data/ssv2_video.zip'
    records = []
    with zipfile.ZipFile(archive) as z:
        for i, q in enumerate(questions):
            matches = [x for x in z.infolist() if Path(x.filename).name == q['video']]
            if len(matches) != 1:
                raise RuntimeError(f'Video not uniquely found: {q["video"]}')
            info = matches[0]
            if workspace_bytes() + info.file_size > CAP or info.file_size > 20_000_000:
                raise RuntimeError('Dataset storage cap')
            dest = ROOT / 'data/videos' / Path(q['video']).name
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(z.read(info))
            records.append(dict(id=f'action_antonym_{i:03d}', source_index=i,
                                task='action_antonym', path=str(dest.relative_to(ROOT)),
                                sha256=sha256(dest), **q))
    api_path = ROOT / 'docs/dataset_api.json'
    revision = json.loads(api_path.read_text())['sha'] if api_path.exists() else None
    manifest = dict(dataset='OpenGVLab/MVBench', revision=revision, task='action_antonym',
                    protocol='First eight published annotation records, original order; not official MVBench',
                    annotation_sha256=sha256(ROOT / 'data/action_antonym.json'),
                    archive_sha256=sha256(archive), samples=records)
    (ROOT / 'data/manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Prepared', len(records), 'real benchmark videos')

if __name__ == '__main__':
    main()
