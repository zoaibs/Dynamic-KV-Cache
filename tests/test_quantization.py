import math
import pytest
import torch
from src.kv_quantization import pack, unpack, HEADER

@pytest.mark.parametrize('bits', [4, 8, 16])
def test_payload_bytes_and_roundtrip(bits):
    torch.manual_seed(10)
    x = torch.randn(1, 3, 7, 65)
    p = pack(x, bits, 64)
    expected = math.ceil(x.numel()*bits/8)
    assert p.payload_bytes == expected
    assert p.payload.dtype == torch.uint8 and p.payload.is_contiguous()
    assert p.metadata_bytes == HEADER.size + (0 if bits==16 else 8*math.ceil(x.numel()/64))
    out = unpack(p, torch.float32)
    assert out.shape == x.shape
    if bits == 16:
        assert torch.equal(out, x.half().float())
    else:
        # Affine nearest rounding error is at most half the group's scale.
        error = (out-x).abs().flatten()
        scale = p.scales.repeat_interleave(64)[:x.numel()]
        assert torch.all(error <= scale/2 + 1e-5)

@pytest.mark.parametrize('bits', [4,8,16])
def test_constant_group_exact(bits):
    x = torch.full((1,1,3,17), -2.5)
    assert torch.equal(unpack(pack(x,bits),torch.float32),x)

def test_bfloat16_is_preserved_exactly():
    x = torch.randn(1,3,5,64).bfloat16()
    assert torch.equal(unpack(pack(x,16)),x)

def test_reject_nonfinite():
    with pytest.raises(ValueError):
        pack(torch.full((1,1,1,1),float('nan')),4)
