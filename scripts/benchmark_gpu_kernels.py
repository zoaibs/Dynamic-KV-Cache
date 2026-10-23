"""Measured tensor-kernel microbenchmarks; not model QA or task accuracy."""
import argparse,json,time
import torch
import torch.nn.functional as F
from src.gpu_cache import GPUCache
from scripts.resources import ROOT

def benchmark(fn,reps=40):
    for _ in range(5):fn()
    torch.cuda.synchronize()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):fn()
    end.record();end.synchronize()
    return start.elapsed_time(end)/reps

def main():
    torch.manual_seed(4)
    rows=[]
    for n in [256,2048,4096,6144]:
        nq=78
        k=torch.randn(1,3,n,64,device='cuda',dtype=torch.bfloat16)
        v=torch.randn_like(k)
        q=torch.randn(1,9,nq,64,device='cuda',dtype=torch.bfloat16)
        kn=torch.randn(1,3,nq,64,device='cuda',dtype=torch.bfloat16)
        vn=torch.randn_like(kn)
        mask=torch.arange(n+nq,device='cuda')[None,:]<=n+torch.arange(nq,device='cuda')[:,None]
        def native():
            return F.scaled_dot_product_attention(q,torch.cat([k,kn],2).repeat_interleave(3,1),
                torch.cat([v,vn],2).repeat_interleave(3,1),attn_mask=mask)
        native_ms=benchmark(native)
        def native_gqa():
            return F.scaled_dot_product_attention(q,torch.cat([k,kn],2),
                torch.cat([v,vn],2),attn_mask=mask,enable_gqa=True)
        gqa_ms=benchmark(native_gqa)
        for bm,bn in [(16,32),(16,64),(16,128),(32,64),(32,128)]:
            for bits in [16,4]:
                cache=GPUCache([(bits,n),(16,nq)],layers=1,block_m=bm,block_n=bn)
                cache.select(bits);cache.append(k,v,0)
                call=lambda:cache.attend(q,kn,vn,0,n)
                try:ms=benchmark(call)
                except Exception as error:
                    rows.append(dict(past_tokens=n,bits=bits,block_m=bm,block_n=bn,
                                     status='unsupported',error=str(error)))
                    print('Unsupported kernel',bm,bn,str(error),flush=True)
                    continue
                row=dict(past_tokens=n,query_tokens=nq,bits=bits,block_m=bm,block_n=bn,
                         native_cat_attention_ms=native_ms,fused_attention_ms=ms,
                         native_gqa_ms=gqa_ms,speedup_vs_gqa=gqa_ms/ms,
                         speedup=native_ms/ms)
                rows.append(row);print(json.dumps(row),flush=True)
    out=dict(scope='Synthetic tensor attention microbenchmark, one layer; not real-model end-to-end',
             cuda_synchronized=True,records=rows)
    (ROOT/'results/raw/gpu_kernel_benchmark.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__':main()
