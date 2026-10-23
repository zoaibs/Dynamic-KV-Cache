"""Combine decoder projections and use CUDA-library normalization.

Native mode keeps upstream operations. Fused BF16 is a separate control, so
these gains are not attributed solely to KV compression.
"""
import types
import torch

def cuda_rms(x,weight,eps):
    # Prebuilt CUDA RMS operation avoids a local C-extension build. Keep the
    # BF16 rounding before multiplication, as in the upstream decoder.
    return torch.nn.functional.rms_norm(x,(x.shape[-1],),None,eps)*weight

def activation_product(gate,up):
    # Projection fusion saves the gate/up GEMM launch. These two pointwise
    # operations deliberately use the existing PyTorch CUDA binary.
    return torch.nn.functional.silu(gate)*up

def install_fused_ops(text,state):
    def rms_forward(module,x):
        return cuda_rms(x,module.weight,module.variance_epsilon) if state['enabled'] else module._original_rms(x)
    def mlp_forward(module,x):
        if not state['enabled']:return module._original_mlp(x)
        gate,up=torch.nn.functional.linear(x,module._gate_up_weight).chunk(2,dim=-1)
        return module.down_proj(activation_product(gate,up))
    def join_weights(modules):
        # Consolidate storage, retaining bit-identical Parameter views for the
        # unchanged native reference. No extra persistent weight copy remains.
        with torch.no_grad():
            joined=torch.cat([m.weight for m in modules],dim=0)
            offset=0
            for m in modules:
                n=m.weight.shape[0];view=joined[offset:offset+n]
                assert torch.equal(view,m.weight),'Projection values changed during packing'
                m.weight=torch.nn.Parameter(view,requires_grad=False)
                offset+=n
        return joined
    for module in text.modules():
        if module.__class__.__name__=='LlamaAttention' and all(m.bias is None for m in [module.q_proj,module.k_proj,module.v_proj]):
            module._qkv_weight=join_weights([module.q_proj,module.k_proj,module.v_proj])
            module._qkv_sizes=[module.q_proj.out_features,module.k_proj.out_features,module.v_proj.out_features]
        elif module.__class__.__name__=='LlamaRMSNorm':
            module._original_rms=module.forward
            module.forward=types.MethodType(rms_forward,module)
        elif module.__class__.__name__=='LlamaMLP' and text.config.hidden_act=='silu':
            module._gate_up_weight=join_weights([module.gate_proj,module.up_proj])
            module._original_mlp=module.forward
            module.forward=types.MethodType(mlp_forward,module)
