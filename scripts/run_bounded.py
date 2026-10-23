"""Run a project module with a watchdog; timeout affects only its own process."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.resources import ROOT, workspace_bytes, CAP

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seconds', type=int, default=900)
    p.add_argument('--log', required=True)
    p.add_argument('module')
    args, extra = p.parse_known_args()
    if workspace_bytes() > CAP:
        raise RuntimeError('Workspace exceeds storage cap')
    env = dict(os.environ, HF_HOME=str(ROOT / '.runtime/huggingface'),
               MPLCONFIGDIR=str(ROOT / '.runtime/matplotlib'),
               TEMP=str(ROOT / '.runtime'), TMP=str(ROOT / '.runtime'),
               TRITON_CACHE_DIR=str(ROOT / '.runtime/triton_cache'),
               TORCHINDUCTOR_CACHE_DIR=str(ROOT / '.runtime/inductor_cache'),
               CUDA_CACHE_PATH=str(ROOT / '.runtime/cuda_cache'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    log = (ROOT / args.log).resolve()
    if not log.is_relative_to(ROOT):
        raise ValueError('Log must be within the workspace')
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    status = {}
    with log.open('w', encoding='utf-8') as output:
        process = subprocess.Popen([sys.executable, '-u', '-m', args.module, *extra],
                                   cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=args.seconds)
            status = dict(status='success' if code == 0 else 'failed', exit_code=code)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            status = dict(status='timeout', timeout_seconds=args.seconds)
    status.update(elapsed_seconds=time.perf_counter()-start,
                  command=[args.module, *extra], workspace_bytes=workspace_bytes())
    log.with_suffix('.status.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)
    if status['status'] != 'success':
        print(log.read_text(encoding='utf-8')[-5000:])
        sys.exit(1)

if __name__ == '__main__':
    main()
