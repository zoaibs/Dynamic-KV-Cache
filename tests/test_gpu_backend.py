"""CUDA integration tests; skipped by the preserved CPU-only environment."""
import json
import pytest
import torch
triton=pytest.importorskip('triton')
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
from src.gpu_cache import GPUCache
from src.streaming import StreamingModel,prepare_video,score_prepared
from src.gpu_ops import cuda_rms,activation_product
from transformers.cache_utils import DynamicCache
from scripts.resources import ROOT

@pytest.mark.parametrize('bits',[4,8,16])
def test_gpu_pack_independent_error_bound(bits):
    torch.manual_seed(71)
    k=torch.randn((1,3,19,64),device='cuda',dtype=torch.bfloat16)
    v=torch.randn((1,19,3,64),device='cuda',dtype=torch.bfloat16).transpose(1,2)
    c=GPUCache([(bits,19)],layers=1);c.select(bits);c.append(k,v,0)
    rk,rv=c.materialize(0)
    for x,y in [(k,rk),(v,rv)]:
        if bits==16:assert torch.equal(x,y)
        else:
            xf=x.float()
            scale=(xf.amax(-1,keepdim=True)-xf.amin(-1,keepdim=True))/((1<<bits)-1)
            # Include the final BF16 reconstruction rounding, independently of
            # the Triton reconstruction implementation.
            assert torch.all((xf-y.float()).abs()<=scale*.501+.032)

def test_gpu_branch_does_not_advance_video_cache():
    c=GPUCache([(4,20),(16,10)],layers=1)
    k=torch.randn(1,3,20,64,device='cuda',dtype=torch.bfloat16)
    c.select(4);c.append(k,k,0)
    before=c.materialize(0)[0].clone()
    branch=c.fork();q=torch.randn(1,3,5,64,device='cuda',dtype=torch.bfloat16)
    branch.append(q,q,0)
    assert c.get_seq_length()==20 and branch.get_seq_length()==25
    assert torch.equal(c.materialize(0)[0],before)

def test_online_growth_and_mapping():
    c=GPUCache(None,layers=1,max_tokens=512)
    expected=[]
    for i,(bits,n) in enumerate([(4,79),(8,93),(4,88),(16,71)]):
        x=torch.randn(1,3,n,64,device='cuda',dtype=torch.bfloat16)
        c.select(bits,i);c.append(x,x,0)
        expected.extend([bits]*n)
    assert c.codes[:len(expected)].tolist()==expected
    assert c.caps[4]>=167 and c.caps[8]>=93 and c.caps[16]>=71
    assert c.get_seq_length()==len(expected)
    before=c.materialize(0)[0].clone()
    branch=c.fork();branch.select(16)
    z=torch.randn(1,3,90,64,device='cuda',dtype=torch.bfloat16)
    old_caps=c.caps.copy();branch.append(z,z,0)
    assert c.caps==old_caps
    assert torch.equal(c.materialize(0)[0],before)

@pytest.mark.parametrize('model_path',['models/smolvlm2','models/smolvlm2_500m'])
def test_shared_streaming_engine_matches_original_model_wrapper(model_path):
    m=StreamingModel(model_path)
    sample=json.loads((ROOT/'data/manifest.json').read_text())['samples'][0]
    prepared=prepare_video(m.adapter,sample,4,False)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        background,_=pool.submit(score_prepared,prepared,m.adapter.config['score']).result()
    synchronous=prepare_video(m.adapter,sample,4,True)
    assert background==synchronous.features
    encoded=m.encode_video(prepared)
    native=m.ingest(prepared,[16]*4,'native',encoded)
    original=DynamicCache()
    m.adapter.forward({'input_ids':m.prefix_ids},original)
    for inputs,features in zip(prepared.inputs,encoded):
        m.adapter.forward({'input_ids':inputs['input_ids'],'image_hidden_states':features},original)
    for (ak,av),(bk,bv) in zip(native,original):
        assert torch.equal(ak,bk) and torch.equal(av,bv)
    before=[(k.clone(),v.clone()) for k,v in native]
    length=native.get_seq_length()
    first=m.answer(sample,native,'candidate')
    second=m.answer(sample,native,'candidate')
    assert first['scores']==second['scores'] and native.get_seq_length()==length
    for (ak,av),(bk,bv) in zip(native,before):
        assert torch.equal(ak,bk) and torch.equal(av,bv)

@pytest.mark.parametrize('query_length',[1,77])
def test_mixed_attention_five_kv_heads_and_long_history(query_length):
    torch.manual_seed(97)
    c=GPUCache(None,layers=1,heads=5,max_tokens=2048)
    for bits,n in [(4,501),(8,498)]:
        k=torch.randn(1,5,n,64,device='cuda',dtype=torch.bfloat16)
        v=torch.randn(1,n,5,64,device='cuda',dtype=torch.bfloat16).transpose(1,2)
        c.select(bits);c.append(k,v,0)
    history_k,history_v=c.materialize(0)
    n=query_length
    q=torch.randn(1,15,n,64,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,5,n,64,device='cuda',dtype=torch.bfloat16)
    v=torch.randn(1,n,5,64,device='cuda',dtype=torch.bfloat16).transpose(1,2)
    c.select(16);past=c.append(k,v,0)
    actual=c.attend(q,k,v,0,past)
    full_k=torch.cat((history_k,k),2).repeat_interleave(3,1)
    full_v=torch.cat((history_v,v),2).repeat_interleave(3,1)
    mask=torch.arange(past+n,device='cuda')[None,:]<=past+torch.arange(n,device='cuda')[:,None]
    expected=torch.nn.functional.scaled_dot_product_attention(q,full_k,full_v,attn_mask=mask).transpose(1,2)
    assert torch.allclose(actual.float(),expected.float(),atol=.035,rtol=.015)

@pytest.mark.parametrize('dim',[576,960])
def test_cuda_rms_matches_upstream(dim):
    x=torch.randn(1,19,dim,device='cuda',dtype=torch.bfloat16)
    weight=torch.randn(dim,device='cuda',dtype=torch.bfloat16)
    reference=weight*(x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+1e-5)).bfloat16()
    actual=cuda_rms(x,weight,1e-5)
    assert (reference.float()-actual.float()).abs().max()<.063
    assert (reference.float()-actual.float()).abs().mean()<.001

def test_fused_activation_matches_upstream():
    gate=torch.randn(1,23,2560,device='cuda',dtype=torch.bfloat16)
    up=torch.randn_like(gate)
    expected=torch.nn.functional.silu(gate)*up
    actual=activation_product(gate,up)
    assert (expected.float()-actual.float()).abs().max()<.063
    assert (expected.float()-actual.float()).abs().mean()<.001
    joined=torch.randn(1,23,5120,device='cuda',dtype=torch.bfloat16)
    gate,up=joined.chunk(2,-1)
    expected=torch.nn.functional.silu(gate)*up
    assert torch.allclose(activation_product(gate,up),expected,atol=.063,rtol=.01)
