"""GPU kernel correctness against independently materialized PyTorch attention."""
import json
import torch
import torch.nn.functional as F
from src.gpu_cache import GPUCache
from scripts.resources import ROOT

def reference(q,k,v,past):
    mask=torch.arange(k.shape[-2],device=q.device)[None,:]<=past+torch.arange(q.shape[-2],device=q.device)[:,None]
    return F.scaled_dot_product_attention(q,k.repeat_interleave(3,1),v.repeat_interleave(3,1),attn_mask=mask).transpose(1,2)

def main():
    torch.manual_seed(12)
    results=[]
    for bits_list in [[16,16,16],[4,8,16],[8,4,8]]:
        lengths=[15,67,23]
        cache=GPUCache(list(zip(bits_list,lengths)),layers=1)
        full_k=[];full_v=[]
        for bits,n in zip(bits_list,lengths):
            q=torch.randn((1,9,n,64),device='cuda',dtype=torch.bfloat16)
            k=torch.randn((1,3,n,64),device='cuda',dtype=torch.bfloat16)
            v=torch.randn_like(k)
            cache.select(bits)
            past=cache.append(k,v,0)
            output=cache.attend(q,k,v,0,past)
            # Compare dequantized HISTORY plus native CURRENT, not compressed current.
            if past:
                hist_k=torch.cat(full_k,dim=2);hist_v=torch.cat(full_v,dim=2)
                rk=torch.cat([hist_k,k],dim=2);rv=torch.cat([hist_v,v],dim=2)
            else:rk,rv=k,v
            expected=reference(q,rk,rv,past)
            diff=(output.float()-expected.float()).abs()
            assert diff.max().item()<.035,(bits_list,n,diff.max().item())
            materialized_k,materialized_v=cache.materialize(0)
            full_k=[materialized_k];full_v=[materialized_v]
            if bits==16:
                assert torch.equal(materialized_k[:,:,-n:],k)
            results.append(dict(bits=bits,n=n,past=past,max_abs_error=diff.max().item(),
                                mean_abs_error=diff.mean().item()))
    torch.cuda.synchronize()
    output=dict(status='passed',gpu=torch.cuda.get_device_name(),tests=results)
    (ROOT/'results/raw/gpu_kernel_correctness.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
