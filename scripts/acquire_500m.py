"""Optional same-family checkpoint candidate; selected on validation only."""
import json
from scripts.resources import ROOT,download,sha256,workspace_bytes

def main():
    metadata=json.loads((ROOT/'docs/model_500m_api.json').read_text())
    for entry in metadata['siblings']:
        name=entry['rfilename']
        if '/' not in name and (name.endswith(('.json','.txt')) or name=='model.safetensors'):
            print('Acquiring 500M',name,entry['size'],flush=True)
            download(f'https://huggingface.co/{metadata["id"]}/resolve/{metadata["sha"]}/{name}',
                     f'models/smolvlm2_500m/{name}',entry['size'])
    weight=next(x for x in metadata['siblings'] if x['rfilename']=='model.safetensors')
    actual=sha256(ROOT/'models/smolvlm2_500m/model.safetensors')
    assert actual==weight['lfs']['sha256']
    print('500M checkpoint verified',actual,'workspace bytes',workspace_bytes())

if __name__=='__main__':main()
