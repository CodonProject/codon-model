import torch.nn.functional as F
import torch
import math

from dataclasses import dataclass
from typing      import Optional, Tuple

try:
    from fla.ops.linear_attn import chunk_linear_attn as _fla_chunk_linear_attn
    from fla.ops.gla import chunk_gla as _fla_chunk_gla
    import triton
    FLA_ENABLA = True
except ImportError:
    FLA_ENABLA = False

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

@dataclass
class AttentionOutput:
    '''
    Output of the attention mechanism.

    Args:
        output (torch.Tensor): The output tensor from the attention mechanism.
        attention_weights (Optional[torch.Tensor], optional): Attention weights. Defaults to None.
        past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): 
            Cached key and value tensors for autoregressive generation. Defaults to None.
    '''
    output: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


def apply_attention(
    query_states: torch.Tensor, 
    key_states: torch.Tensor, 
    value_states: torch.Tensor, 
    attention_mask: torch.Tensor = None, 
    output_attentions: bool = False,
    is_causal: bool = None,
    dropout: float = 0.0
) -> AttentionOutput:
    ''' 
    Compute scaled dot-product attention.

    Args:
        query_states (torch.Tensor): Query states tensor.
        key_states (torch.Tensor): Key states tensor.
        value_states (torch.Tensor): Value states tensor.
        attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
        output_attentions (bool, optional): Whether to output attention weights. Defaults to False.
        is_causal (bool, optional): Whether to apply a causal mask. Defaults to None.
        dropout (float, optional): Dropout probability. Defaults to 0.0.

    Returns:
        AttentionOutput: Object containing attention output and optional weights.
    '''
    
    if (
        FLASH_ATTN_AVAILABLE 
        and query_states.is_cuda
        and query_states.dtype in [torch.float16, torch.bfloat16]
        and not output_attentions
        and attention_mask is None
    ):
        q = query_states.transpose(1, 2)
        k = key_states.transpose(1, 2)
        v = value_states.transpose(1, 2)
        
        out = flash_attn_func(q, k, v, dropout_p=dropout, causal=bool(is_causal))
        
        output = out.transpose(1, 2)
        return AttentionOutput(output=output, attention_weights=None)
    
    if attention_mask is not None:
        if attention_mask.dtype != torch.float32:
            attention_mask = attention_mask.float()
        
        if attention_mask.max() <= 1.0:
            attention_mask = torch.where(attention_mask == 0, float('-inf'), 0.0)
        
    if is_causal:
        tgt_len = query_states.size(-2)
        src_len = key_states.size(-2)
        
        causal_mask = torch.tril(
            torch.ones((tgt_len, src_len), device=query_states.device, dtype=query_states.dtype)
        ).view(1, 1, tgt_len, src_len)
        
        causal_mask = torch.where(causal_mask == 0, float('-inf'), 0.0)
        
        if attention_mask is not None:
            attention_mask = attention_mask + causal_mask
        else:
            attention_mask = causal_mask
        
        is_causal = False
        
    if not output_attentions:
        if attention_mask is None and is_causal is None: 
            is_causal = True
        
        try:
            output = F.scaled_dot_product_attention(
                query_states, 
                key_states, 
                value_states, 
                attn_mask=attention_mask,
                is_causal=is_causal,
                dropout_p=dropout
            )
            return AttentionOutput(output=output, attention_weights=None)
        except RuntimeError: pass
    # Manual Fallback Path
    d_k = query_states.size(-1)
    scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(d_k)

    if attention_mask is not None:
        scores = scores + attention_mask
    attention_weights = torch.softmax(scores, dim=-1)
    
    if dropout > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout)
    output = torch.matmul(attention_weights, value_states)
    
    return AttentionOutput(output=output, attention_weights=attention_weights)


def _chunk_linear_attn_native(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
):
    B, H, L, D = q.shape
    V_dim = v.shape[-1]
    
    orig_dtype = q.dtype
    
    q = (q * scale).to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    
    if initial_state is not None:
        state = initial_state.to(torch.float32).clone()
    else:
        state = torch.zeros(B, H, D, V_dim, device=q.device, dtype=torch.float32)
        
    output = torch.empty_like(v)
    
    for t in range(L):
        q_t = q[:, :, t, :]
        k_t = k[:, :, t, :]
        v_t = v[:, :, t, :]
        
        # State: [B, H, D, V_dim]
        state = state + torch.einsum('bhd,bhv->bhdv', k_t, v_t)
        # O: [B, H, V_dim]
        output[:, :, t, :] = torch.einsum('bhd,bhdv->bhv', q_t, state)
        
    return output.to(orig_dtype), (state if output_final_state else None)


def _chunk_gla_native(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
):
    B, H, L, D = q.shape
    V_dim = v.shape[-1]
    
    orig_dtype = q.dtype
    
    q = (q * scale).to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    
    decay = torch.exp(g.to(torch.float32))
    if decay.dim() == 3:
        decay = decay.unsqueeze(-1)
        
    if initial_state is not None:
        state = initial_state.to(torch.float32).clone()
    else:
        state = torch.zeros(B, H, D, V_dim, device=q.device, dtype=torch.float32)
        
    output = torch.empty_like(v)
    
    for t in range(L):
        q_t = q[:, :, t, :]
        k_t = k[:, :, t, :]
        v_t = v[:, :, t, :]
        d_t = decay[:, :, t, :]  # [B, H, 1]
        
        # State = State * decay + K^T * V
        state = state * d_t.unsqueeze(-1) + torch.einsum('bhd,bhv->bhdv', k_t, v_t)
        output[:, :, t, :] = torch.einsum('bhd,bhdv->bhv', q_t, state)
        
    return output.to(orig_dtype), (state if output_final_state else None)

if FLA_ENABLA:
    chunk_linear_attn: callable = _fla_chunk_linear_attn
    chunk_gla: callable = _fla_chunk_gla
else:
    chunk_linear_attn: callable = _chunk_linear_attn_native
    chunk_gla: callable = _chunk_gla_native