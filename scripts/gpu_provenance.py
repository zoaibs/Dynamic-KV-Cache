"""Record the exact tested GPU environment and source/checkpoint identities."""
import json,platform,importlib.metadata as md,sys
import psutil
import torch,triton
from scripts.resources import ROOT,sha256,workspace_bytes,CAP

def main():
    selected=json.loads((ROOT/'configs/gpu_selected.json').read_text())
    api=json.loads((ROOT/('docs/model_500m_api.json' if '500m' in selected['model_path'] else 'docs/model_api.json')).read_text(encoding='utf-8-sig'))
    entry=next(x for x in api['siblings'] if x['rfilename']=='model.safetensors')
    actual=sha256(ROOT/selected['model_path']/'model.safetensors')
    assert actual==entry['lfs']['sha256']
    sources={str(p.relative_to(ROOT)):sha256(p) for folder in ['src','scripts','tests','benchmarks','configs']
             for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    props=torch.cuda.get_device_properties(0)
    result=dict(python=sys.version,executable=sys.executable,os=platform.platform(),
        torch=torch.__version__,cuda_runtime=torch.version.cuda,triton=triton.__version__,
        gpu=props.name,capability=torch.cuda.get_device_capability(),gpu_total_bytes=props.total_memory,
        ram_total_bytes=psutil.virtual_memory().total,ram_available_bytes=psutil.virtual_memory().available,
        selected=selected,selected_sha256=sha256(ROOT/'configs/gpu_selected.json'),
        checkpoint_sha256=actual,source_sha256=sources,
        installed_packages={d.metadata['Name']:d.version for d in md.distributions()},
        workspace_bytes=workspace_bytes(),workspace_cap_bytes=CAP,
        cpu_reference_archive='archives/cpu_v1.zip',security_settings_changed=False,
        notes=['GPU kernel prototype with additional launcher arguments was blocked by Windows Application Control.',
               'The previously tested Triton launcher interface and PyTorch CUDA operations are used instead.',
               'Final timing overlaps independent CPU feature scoring with CUDA vision; synchronous/background feature parity is tested.',
               'Projection parameters are consolidated into shared storage with exact value checks; no weight values are trained or quantized.'])
    (ROOT/'results/gpu_provenance.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['torch','cuda_runtime','gpu','workspace_bytes','checkpoint_sha256']},indent=2))

if __name__=='__main__':main()
