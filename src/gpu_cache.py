"""GPU-only precision pools, one pack launch per layer and direct packed attention."""
import types
import torch
import torch.nn.functional as F
import triton
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from src.gpu_kernels import pack_rows, packed_attention, reconstruct

class GPUCache(DynamicCache):
    def __init__(self,plan=None,layers=30,heads=3,dim=64,device='cuda',dtype=torch.bfloat16,
                 backend='fused',block_m=16,block_n=64,max_tokens=8192):
        super().__init__()
        if dim!=64: raise ValueError('GPU kernel currently supports head_dim=64')
        self.layers,self.heads,self.dim=layers,heads,dim
        self.dtype,self.device=dtype,torch.device(device)
        self.backend,self.block_m,self.block_n=backend,block_m,block_n
        self.seen=[0]*layers
        self.cursors=[0]*layers
        self.pending_bits=16
        self.frame_id=-1
        self.online=plan is None
        rows=[];codes=[];counts={4:0,8:0,16:0}
        self.plan=[]
        for bits,n in (plan or []):
            if bits not in counts or n<0: raise ValueError('Invalid GPU cache plan')
            self.plan.append((len(rows),len(rows)+n,bits,counts[bits]))
            rows.extend(range(counts[bits],counts[bits]+n));codes.extend([bits]*n)
            counts[bits]+=n
        self.capacity=max_tokens if self.online else len(rows)
        self.caps={b:64 if self.online else max(1,n) for b,n in counts.items()}
        self.rows=(torch.empty(self.capacity,dtype=torch.int32,device=self.device) if self.online
                   else torch.tensor(rows,dtype=torch.int32,device=self.device))
        self.codes=(torch.empty(self.capacity,dtype=torch.uint8,device=self.device) if self.online
                    else torch.tensor(codes,dtype=torch.uint8,device=self.device))
        common=(layers,2,heads)
        self.p4=torch.empty((*common,self.caps[4],dim//2),dtype=torch.uint8,device=self.device)
        self.p8=torch.empty((*common,self.caps[8],dim),dtype=torch.uint8,device=self.device)
        self.p16=torch.empty((*common,self.caps[16],dim),dtype=dtype,device=self.device)
        self.m4=torch.empty((*common,self.caps[4],2),dtype=torch.float32,device=self.device)
        self.m8=torch.empty((*common,self.caps[8],2),dtype=torch.float32,device=self.device)
        self.host_counts=counts

    def reserve(self,bits,needed):
        old=self.caps[bits]
        if needed<=old:return
        cap=max(needed,((int(old*1.5)+63)//64)*64)
        common=(self.layers,2,self.heads)
        attr={4:'p4',8:'p8',16:'p16'}[bits]
        source=getattr(self,attr)
        target=torch.empty((*common,cap,self.dim//2 if bits==4 else self.dim),
                           dtype=source.dtype,device=self.device)
        target[:,:,:,:old].copy_(source)
        setattr(self,attr,target)
        if bits!=16:
            attr='m4' if bits==4 else 'm8';source=getattr(self,attr)
            target=torch.empty((*common,cap,2),dtype=source.dtype,device=self.device)
            target[:,:,:,:old].copy_(source);setattr(self,attr,target)
        self.caps[bits]=cap

    def select(self,bits,frame_id=-1):
        self.pending_bits,self.frame_id=bits,frame_id

    def get_seq_length(self,layer_idx=0): return self.seen[layer_idx]
    def get_max_cache_shape(self): return None
    def __len__(self): return self.layers

    def append(self,k,v,layer):
        if k.shape[0]!=1 or k.shape[-1]!=self.dim or k.device.type!='cuda':
            raise ValueError('GPU cache requires CUDA, batch one and the configured head dimension')
        past=self.seen[layer];n=k.shape[-2]
        if past+n>self.capacity:raise ValueError('GPU context limit exceeded')
        if self.online:
            if layer==0:
                row=self.host_counts[self.pending_bits]
                self.reserve(self.pending_bits,row+n)
                self.plan.append((past,past+n,self.pending_bits,row))
                self.host_counts[self.pending_bits]+=n
                self.rows[past:past+n]=torch.arange(row,row+n,device=self.device,dtype=torch.int32)
                self.codes[past:past+n].fill_(self.pending_bits)
            segment=self.plan[-1]
        else:
            cursor=self.cursors[layer]
            while cursor<len(self.plan) and past>=self.plan[cursor][1]:cursor+=1
            self.cursors[layer]=cursor
            segment=self.plan[cursor] if cursor<len(self.plan) else None
        if segment is None or past+n>segment[1] or segment[2]!=self.pending_bits:
            raise ValueError(f'Cache append does not match plan at token {past}, length {n}, bits {self.pending_bits}')
        row=segment[3]+past-segment[0]
        pack_rows[(self.heads*n,2)](k,v,self.p4,self.m4,self.p8,self.m8,self.p16,
            k.stride(1),k.stride(2),v.stride(1),v.stride(2),layer,row,n,
            self.caps[4],self.caps[8],self.caps[16],self.heads,self.dim,self.pending_bits)
        self.seen[layer]=past+n
        return past

    def attend(self,q,k,v,layer,past):
        n=q.shape[-2]
        out=torch.empty((1,n,q.shape[1],self.dim),dtype=q.dtype,device=q.device)
        packed_attention[(q.shape[1],triton.cdiv(n,self.block_m))](q,k,v,
            self.p4,self.m4,self.p8,self.m8,self.p16,self.rows,self.codes,out,
            q.stride(1),q.stride(2),k.stride(1),k.stride(2),v.stride(1),v.stride(2),
            layer,past,n,self.caps[4],self.caps[8],self.caps[16],self.heads,
            q.shape[1]//self.heads,self.dim,self.block_m,self.block_n,num_warps=4,num_stages=1)
        return out

    def materialize(self,layer):
        n=self.seen[layer]
        out=torch.empty((2,self.heads,n,self.dim),dtype=self.dtype,device=self.device)
        reconstruct[(self.heads,2,triton.cdiv(n,32))](self.p4,self.m4,self.p8,self.m8,self.p16,
            self.rows,self.codes,out,layer,n,self.caps[4],self.caps[8],self.caps[16],
            self.heads,self.dim,32)
        return out[0:1],out[1:2]

    def update(self,key_states,value_states,layer_idx,cache_kwargs=None):
        self.append(key_states,value_states,layer_idx)
        k,v=self.materialize(layer_idx)
        k[:,:,-key_states.shape[-2]:].copy_(key_states)
        v[:,:,-value_states.shape[-2]:].copy_(value_states)
        return k,v

    def fork(self):
        # QA branches share immutable video pools; their reserved text region is
        # overwritten sequentially on one CUDA stream, never concurrently.
        other=object.__new__(GPUCache)
        other.__dict__=self.__dict__.copy()
        other.seen=self.seen.copy()
        other.cursors=self.cursors.copy()
        other.plan=self.plan.copy()
        other.host_counts=self.host_counts.copy()
        other.caps=self.caps.copy()
        other.pending_bits=16
        return other

    def accounting(self):
        tensors=[self.p4,self.p8,self.p16,self.m4,self.m8,self.rows,self.codes]
        allocated=sum(t.numel()*t.element_size() for t in tensors)
        used={4:0,8:0,16:0}
        n=self.seen[0]
        for start,end,bits,row in self.plan:
            used[bits]+=max(0,min(n,end)-start)
        payload=self.layers*2*self.heads*self.dim*sum(used[b]*b/8 for b in used)
        scales=self.layers*2*self.heads*8*(used[4]+used[8])
        mapping=n*5
        return dict(gpu_cache_used_bytes=int(payload+scales+mapping),
                    gpu_cache_allocated_bytes=allocated,cpu_cache_payload_bytes=0,
                    payload_bytes=int(payload),quantization_metadata_bytes=int(scales),
                    token_mapping_bytes=mapping,logical_index_capacity=self.capacity,
                    precision_pool_capacities=self.caps,
                    used_tokens=n,precision_token_counts=used)

def install_packed_attention(model):
    """Dispatch GPUCache through fused attention; all other caches use upstream code."""
    def forward(module,hidden_states,position_embeddings,attention_mask=None,
                past_key_value=None,cache_position=None,**kwargs):
        fused=isinstance(past_key_value,GPUCache) and past_key_value.backend=='fused'
        fast_gqa=getattr(past_key_value,'fast_gqa',False)
        if not fused and not fast_gqa:
            return module._original_forward(hidden_states=hidden_states,
                position_embeddings=position_embeddings,attention_mask=attention_mask,
                past_key_value=past_key_value,cache_position=cache_position,**kwargs)
        if kwargs.get('output_attentions',False):
            raise ValueError('Fused packed attention does not expose attention matrices')
        shape=hidden_states.shape[:-1]
        if fused and hasattr(module,'_qkv_weight'):
            q,k,v=F.linear(hidden_states,module._qkv_weight).split(module._qkv_sizes,dim=-1)
        else:q,k,v=module.q_proj(hidden_states),module.k_proj(hidden_states),module.v_proj(hidden_states)
        q=q.view(*shape,-1,module.head_dim).transpose(1,2)
        k=k.view(*shape,-1,module.head_dim).transpose(1,2)
        v=v.view(*shape,-1,module.head_dim).transpose(1,2)
        q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
        if fused:
            past=past_key_value.append(k,v,module.layer_idx)
            output=past_key_value.attend(q,k,v,module.layer_idx,past)
        else:
            k,v=past_key_value.update(k,v,module.layer_idx,{'cache_position':cache_position})
            mask=None if attention_mask is None else attention_mask[:,:,:,:k.shape[-2]]
            output=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0,
                is_causal=mask is None and q.shape[-2]>1,scale=module.scaling,enable_gqa=True).transpose(1,2)
        return module.o_proj(output.reshape(*shape,-1).contiguous()),None
    for module in model.modules():
        if module.__class__.__name__=='LlamaAttention' and not hasattr(module,'_original_forward'):
            module._original_forward=module.forward
            module.forward=types.MethodType(forward,module)
