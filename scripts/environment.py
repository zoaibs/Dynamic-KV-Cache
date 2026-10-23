"""Read development hardware facts only; no system configuration changes."""
import ctypes
import json
import platform
import shutil
import subprocess
import sys
import re
import winreg
from pathlib import Path
import psutil
import torch
from scripts.resources import ROOT, workspace_bytes

def main():
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                          '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
    smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=15)
    cuda_max = re.search(r'CUDA Version:\s*([\d.]+)', smi.stdout)
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        cpu_name=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    data = dict(os=platform.platform(), python=sys.version, executable=sys.executable,
                processor=platform.processor(), cpu_name=cpu_name,cpu_count=psutil.cpu_count(),
                physical_cores=psutil.cpu_count(logical=False),
                ram_total_bytes=psutil.virtual_memory().total,
                ram_available_bytes=psutil.virtual_memory().available,
                disk_free_bytes=shutil.disk_usage(ROOT).free,
                workspace_bytes=workspace_bytes(), workspace_cap_bytes=5_000_000_000,
                nvidia_smi=gpu.stdout.strip(), torch=torch.__version__,
                driver_supported_cuda_max=cuda_max.group(1) if cuda_max else None,
                torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                execution_device='cpu', cuda_wheel_download_bytes=3273066072,
                cuda_not_installed_reason='5 GB total project budget; CUDA wheel alone 3.27 GB',
                git_available=shutil.which('git') is not None,
                notes=['CIM system query denied; RAM/disk obtained with psutil.',
                       'Embedded CPython isolated in workspace; no system installation.',
                       'GPU allocated/reserved metrics are unavailable in this CPU build.'])
    (ROOT / 'results/environment.json').write_text(json.dumps(data, indent=2))
    print(json.dumps(data, indent=2))

if __name__ == '__main__':
    main()
