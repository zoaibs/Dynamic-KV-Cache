import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from src.cache_manager import MixedPrecisionCache, SEGMENT
from src.kv_quantization import unpack

def test_mixed_retrieval_shapes_positions_and_reuse():
    torch.manual_seed(7)
    c = MixedPrecisionCache()
    expected_k, expected_v = [], []
    for frame, bits in enumerate([4,8,16]):
        c.select(bits, frame)
        k, v = torch.randn(1,3,7,64).bfloat16(), torch.randn(1,3,7,64).bfloat16()
        out_k, out_v = c.update(k,v,0)
        assert out_k.shape == out_v.shape == (1,3,(frame+1)*7,64)
        assert torch.equal(out_k[..., -7:, :],k)
        if frame:
            assert torch.equal(out_k[..., :-7, :],torch.cat(expected_k,dim=-2))
        expected_k.append(unpack(c.segments[0][-1].key))
        expected_v.append(unpack(c.segments[0][-1].value))
        record = SEGMENT.unpack(bytes(c.segments[0][-1].descriptor.tolist()))
        assert record == (0,frame,frame*7,(frame+1)*7,bits)
    assert c.get_seq_length()==21 and c.retrievals==3
    assert torch.equal(c[0][0],torch.cat(expected_k,dim=-2))
    a = c.accounting()
    assert a['cpu_cache_bytes'] == a['cpu_payload_bytes'] + a['cpu_metadata_bytes']
    assert all(a['bytes_by_precision'][b]>0 for b in [4,8,16])

def test_real_attention_api_full_precision_equivalence():
    # Random small decoder is a unit test only, never an experimental result.
    torch.manual_seed(11)
    config = LlamaConfig(vocab_size=40, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    model = LlamaForCausalLM(config).eval().to(torch.bfloat16)
    caches = [DynamicCache(), MixedPrecisionCache()]
    with torch.inference_mode():
        for tokens in [torch.tensor([[1,2,3]]),torch.tensor([[4,5]]),torch.tensor([[6]])]:
            outputs = [model(input_ids=tokens,past_key_values=c,use_cache=True).logits for c in caches]
            assert torch.equal(*outputs)
