"""Hash executable sources, checkpoint, data and measured outputs for review."""
import importlib.metadata as md
import json
from pathlib import Path
import shutil
from scripts.resources import ROOT, sha256, workspace_bytes

def main():
    model=json.loads((ROOT/'docs/model_api.json').read_text(encoding='utf-8-sig'))
    expected=next(x for x in model['siblings'] if x['rfilename']=='model.safetensors')['lfs']['sha256']
    checkpoint_hash=sha256(ROOT/'models/smolvlm2/model.safetensors')
    assert checkpoint_hash==expected, 'Checkpoint SHA256 does not match public LFS metadata'
    sources={}
    for folder in ['src','tests','scripts','benchmarks','configs']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                sources[str(p.relative_to(ROOT))]=sha256(p)
    installed={d.metadata['Name']:d.version for d in md.distributions()}
    # Keep an exact copy of the baseline executable and configuration.
    destination=ROOT/'docs/baseline_source'
    destination.mkdir(exist_ok=True)
    for f in ['src/model_adapter.py','benchmarks/baseline.py','configs/default.json']:
        shutil.copyfile(ROOT/f,destination/Path(f).name)
    smoke=json.loads((ROOT/'results/raw/baseline_smoke.json').read_text())
    assert sha256(ROOT/'src/model_adapter.py')==smoke['adapter_sha256']
    data=dict(checkpoint_sha256=checkpoint_hash,checkpoint_public_lfs_sha256=expected,
              checkpoint_revision=model['sha'],source_sha256=sources,
              transformers_source_commit=json.loads((ROOT/'docs/transformers_commit.json').read_text())['sha'],
              rekv_inspected_commit='1fd9a3dbf5dbff7f27069ae2f4463674c495e830',
              installed_packages=installed,workspace_bytes=workspace_bytes(),
              baseline_adapter_unchanged_since_smoke=True)
    (ROOT/'results/provenance.json').write_text(json.dumps(data,indent=2,sort_keys=True))
    print('Verified public checkpoint hash and unchanged baseline adapter; bytes:',workspace_bytes())

if __name__=='__main__':
    main()
