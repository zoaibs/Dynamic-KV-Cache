"""Workspace-scoped downloads with a hard total storage cap; never uses accounts."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import urllib.error
import time

ROOT = Path(__file__).resolve().parents[1]
CAP = 15_000_000_000

def workspace_bytes():
    # DirEntry caches Windows file metadata. Avoid two slow stat calls per
    # file under OneDrive, while counting the same regular files and file links.
    total=0;pending=[ROOT]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):pending.append(entry.path)
                elif entry.is_file():total+=entry.stat().st_size
    return total

def download(url, target, expected=None):
    target = ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (expected is None or target.stat().st_size == expected):
        return
    if expected and workspace_bytes() + expected > CAP:
        raise RuntimeError('Download would exceed the 15 GB workspace cap')
    req = urllib.request.Request(url, headers={'User-Agent': 'DynamicKVResearch/0.1'})
    for attempt in range(4):
        try:
            response=urllib.request.urlopen(req,timeout=60)
            break
        except urllib.error.HTTPError as error:
            if error.code not in (429,503) or attempt==3:raise
            delay=int(error.headers.get('Retry-After',30*(attempt+1)))
            print(f'Public server requested retry; waiting {delay}s',flush=True)
            time.sleep(delay)
    with response:
        size = int(response.headers.get('Content-Length', expected or 0))
        if workspace_bytes() + size > CAP:
            raise RuntimeError('Response would exceed the 15 GB workspace cap')
        with target.open('wb') as output:
            total = 0
            while block := response.read(4 * 1024 * 1024):
                total += len(block)
                if total > (expected or size or 1_200_000_000):
                    raise RuntimeError('Unexpected response size')
                output.write(block)
        if expected and total != expected:
            raise RuntimeError(f'Incomplete download: {total} != {expected}')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

if __name__ == '__main__':
    metadata = json.loads((ROOT / 'docs/model_api.json').read_text(encoding='utf-8-sig'))
    revision = metadata['sha']
    for entry in metadata['siblings']:
        name = entry['rfilename']
        if '/' not in name and (name.endswith(('.json', '.txt')) or name == 'model.safetensors'):
            print('Acquiring', name, entry['size'], flush=True)
            download(f'https://huggingface.co/{metadata["id"]}/resolve/{revision}/{name}',
                     f'models/smolvlm2/{name}', entry['size'])
    manifest = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                for p in (ROOT / 'models/smolvlm2').glob('*') if p.is_file()}
    (ROOT / 'docs/checkpoint_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Workspace bytes:', workspace_bytes(), flush=True)
