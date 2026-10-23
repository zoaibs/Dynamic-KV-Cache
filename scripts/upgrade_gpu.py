"""Preserve the CPU environment and build a separate project-local CUDA runtime."""
from pathlib import Path
import json
import shutil
import zipfile
from scripts.resources import ROOT, CAP, workspace_bytes

def main():
    archive=ROOT/'archives/cpu_v1.zip'
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for folder in ['src','tests','configs','scripts','benchmarks','results','paper','docs']:
                for p in (ROOT/folder).rglob('*'):
                    if p.is_file() and '__pycache__' not in p.parts:
                        z.write(p,p.relative_to(ROOT))
            for f in ['README.md','FINAL_SUMMARY.md','requirements.txt','.gitignore']:
                z.write(ROOT/f,f)
    used=workspace_bytes()
    if (ROOT/'.runtime/gpu/Lib/site-packages/torch-2.7.1+cu128.dist-info').exists():
        print('GPU runtime already installed; CPU environment remains preserved.',flush=True)
        return
    # Allow the 5.79 GB installed wheel + 3.28 GB temporary download + ~0.8 GB
    # non-torch runtime, while retaining the complete original CPU environment.
    if used+10_100_000_000>CAP:
        raise RuntimeError('Insufficient 15 GB budget for a separate GPU environment')
    src=ROOT/'.runtime/python'
    dest=ROOT/'.runtime/gpu'
    def ignore(directory,names):
        return [n for n in names if n in {'torch','torchgen','functorch','__pycache__'}
                or n.startswith('torch-')]
    shutil.copytree(src,dest,ignore=ignore,dirs_exist_ok=True)
    (ROOT/'results/gpu_upgrade.json').write_text(json.dumps(dict(
        original_environment_preserved=str(src.relative_to(ROOT)),
        new_environment=str(dest.relative_to(ROOT)),cap_bytes=CAP,
        original_results_archive=str(archive.relative_to(ROOT))),indent=2))
    print('Isolated GPU runtime prepared; original CPU installation preserved.',flush=True)

if __name__=='__main__':
    main()
