"""Triton pack and fused causal attention over physically packed KV history.

The online-softmax recurrence follows the standard FlashAttention formulation.
This is inference-only, batch=1, head_dim=64, with grouped-query attention.
No reconstructed full-history K/V tensor is allocated in the fused path.
"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def pack_rows(K, V, P4, M4, P8, M8, P16,
              kh, kt, vh, vt, layer, row_start, n,
              cap4, cap8, cap16, H: tl.constexpr, D: tl.constexpr,
              BITS: tl.constexpr):
    row=tl.program_id(0)
    kv=tl.program_id(1)
    head=row//n
    token=row%n
    d=tl.arange(0,D)
    ptr=tl.where(kv==0,K+head*kh+token*kt,V+head*vh+token*vt)
    x=tl.load(ptr+d).to(tl.float32)
    out_row=row_start+token
    if BITS==16:
        offset=((layer*2+kv)*H+head)*cap16*D+out_row*D+d
        tl.store(P16+offset,x)
    else:
        a=tl.min(x,0)
        span=tl.max(x,0)-a
        s=tl.where(span>0,span/((1<<BITS)-1),1.)
        q=libdevice.nearbyint((x-a)/s).to(tl.int32)
        q=tl.minimum(tl.maximum(q,0),(1<<BITS)-1)
        if BITS==4:
            half=tl.arange(0,D//2)
            lo=tl.gather(q,2*half,0)
            hi=tl.gather(q,2*half+1,0)
            packed=lo|(hi<<4)
            offset=(((layer*2+kv)*H+head)*cap4+out_row)*(D//2)+half
            tl.store(P4+offset,packed.to(tl.uint8))
            mo=(((layer*2+kv)*H+head)*cap4+out_row)*2
            tl.store(M4+mo,a);tl.store(M4+mo+1,s)
        else:
            offset=(((layer*2+kv)*H+head)*cap8+out_row)*D+d
            tl.store(P8+offset,q.to(tl.uint8))
            mo=(((layer*2+kv)*H+head)*cap8+out_row)*2
            tl.store(M8+mo,a);tl.store(M8+mo+1,s)

@triton.jit
def load_history(P4,M4,P8,M8,P16,Rows,Codes,layer,head,t,d,
                 cap4,cap8,cap16,H:tl.constexpr,D:tl.constexpr,
                 kv:tl.constexpr,past):
    valid=t<past
    row=tl.load(Rows+t,valid,other=0)
    code=tl.load(Codes+t,valid,other=16)
    base=((layer*2+kv)*H+head)
    b4=tl.load(P4+(base*cap4+row[:,None])*(D//2)+d[None,:]//2,
               valid[:,None]&(code[:,None]==4),other=0).to(tl.int32)
    q4=(b4>>((d[None,:]%2)*4))&15
    a4=tl.load(M4+(base*cap4+row)*2,valid&(code==4),other=0)
    s4=tl.load(M4+(base*cap4+row)*2+1,valid&(code==4),other=0)
    q8=tl.load(P8+(base*cap8+row[:,None])*D+d[None,:],
               valid[:,None]&(code[:,None]==8),other=0).to(tl.float32)
    a8=tl.load(M8+(base*cap8+row)*2,valid&(code==8),other=0)
    s8=tl.load(M8+(base*cap8+row)*2+1,valid&(code==8),other=0)
    x16=tl.load(P16+(base*cap16+row[:,None])*D+d[None,:],
               valid[:,None]&(code[:,None]==16),other=0).to(tl.float32)
    x4=q4.to(tl.float32)*s4[:,None]+a4[:,None]
    x8=q8*s8[:,None]+a8[:,None]
    return tl.where(code[:,None]==4,x4,tl.where(code[:,None]==8,x8,x16))

@triton.jit
def packed_attention(Q,Knew,Vnew,P4,M4,P8,M8,P16,Rows,Codes,Out,
                     qh,qt,kh,kt,vh,vt,layer,past,nq,cap4,cap8,cap16,
                     H:tl.constexpr,GROUP:tl.constexpr,D:tl.constexpr,
                     BM:tl.constexpr,BN:tl.constexpr):
    query_head=tl.program_id(0)
    head=query_head//GROUP
    m=tl.program_id(1)*BM+tl.arange(0,BM)
    d=tl.arange(0,D)
    q=tl.load(Q+query_head*qh+m[:,None]*qt+d[None,:],m[:,None]<nq,other=0)
    maximum=tl.full((BM,),-float('inf'),tl.float32)
    denominator=tl.zeros((BM,),tl.float32)
    acc=tl.zeros((BM,D),tl.float32)
    for block in range(tl.cdiv(past+nq,BN)):
        t=block*BN+tl.arange(0,BN)
        k=load_history(P4,M4,P8,M8,P16,Rows,Codes,layer,head,t,d,
                       cap4,cap8,cap16,H,D,0,past)
        v=load_history(P4,M4,P8,M8,P16,Rows,Codes,layer,head,t,d,
                       cap4,cap8,cap16,H,D,1,past)
        current=(t>=past)&(t<past+nq)
        kn=tl.load(Knew+head*kh+(t[:,None]-past)*kt+d[None,:],current[:,None],other=0)
        vn=tl.load(Vnew+head*vh+(t[:,None]-past)*vt+d[None,:],current[:,None],other=0)
        k=tl.where(current[:,None],kn,k).to(q.dtype)
        v=tl.where(current[:,None],vn,v).to(q.dtype)
        scores=tl.dot(q,tl.trans(k))*(D**-0.5)
        valid=(t[None,:]<past+nq)&(t[None,:]<=past+m[:,None])
        scores=tl.where(valid,scores,-float('inf'))
        new_max=tl.maximum(maximum,tl.max(scores,1))
        alpha=tl.exp(maximum-new_max)
        p=tl.exp(scores-new_max[:,None])
        denominator=denominator*alpha+tl.sum(p,1)
        acc=acc*alpha[:,None]+tl.dot(p.to(q.dtype),v)
        maximum=new_max
    output=acc/denominator[:,None]
    tl.store(Out+(m[:,None]*(H*GROUP)+query_head)*D+d[None,:],output,m[:,None]<nq)

@triton.jit
def reconstruct(P4,M4,P8,M8,P16,Rows,Codes,Out,
                layer,n,cap4,cap8,cap16,H:tl.constexpr,D:tl.constexpr,BN:tl.constexpr):
    head=tl.program_id(0)
    kv=tl.program_id(1)
    t=tl.program_id(2)*BN+tl.arange(0,BN)
    d=tl.arange(0,D)
    # kv needs a compile-time value for the shared helper; specialize by branch.
    if kv==0:
        x=load_history(P4,M4,P8,M8,P16,Rows,Codes,layer,head,t,d,cap4,cap8,cap16,H,D,0,n)
    else:
        x=load_history(P4,M4,P8,M8,P16,Rows,Codes,layer,head,t,d,cap4,cap8,cap16,H,D,1,n)
    tl.store(Out+((kv*H+head)*n+t[:,None])*D+d[None,:],x,t[:,None]<n)
