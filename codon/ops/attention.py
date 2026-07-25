from codon import *


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
    if attention_mask is not None and attention_mask.numel() == 0:
        attention_mask = None

    tgt_len = query_states.size(-2)
    src_len = key_states.size(-2)
    if attention_mask is not None:
        # Match query dtype rather than forcing FP32, so mixed-precision graphs
        # (e.g. FP16 ONNX exports) stay type-consistent end-to-end.
        if attention_mask.dtype != query_states.dtype:
            attention_mask = attention_mask.to(query_states.dtype)

        if attention_mask.max() <= 1.0:
            neg_inf = torch.tensor(
                float('-inf'), dtype=attention_mask.dtype, device=attention_mask.device
            )
            zero = torch.tensor(
                0.0, dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.where(attention_mask == 0, neg_inf, zero)
        
    if is_causal and tgt_len > 1:
        # Offset diagonal by (src_len - tgt_len) so causal masking remains correct
        # when a KV cache is present (i.e., src_len > tgt_len). For pure prefill
        # (src_len == tgt_len) this reduces to the standard lower-triangular mask.
        causal_mask = torch.tril(
            torch.ones((tgt_len, src_len), device=query_states.device, dtype=query_states.dtype),
            diagonal=src_len - tgt_len
        ).view(1, 1, tgt_len, src_len)

        causal_mask = torch.where(causal_mask == 0, float('-inf'), 0.0)
        
        if attention_mask is not None:
            attention_mask = attention_mask + causal_mask
        else:
            attention_mask = causal_mask
        is_causal = False
    else:
        is_causal = False
        
    if not output_attentions:
        if attention_mask is None and is_causal is None: is_causal = True
        
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

    d_k = query_states.size(-1)
    scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(d_k)
    
    if attention_mask is not None:
        scores = scores + attention_mask
    attention_weights = torch.softmax(scores, dim=-1)
    
    if dropout > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout)
    output = torch.matmul(attention_weights, value_states)
    
    return AttentionOutput(output=output, attention_weights=attention_weights)
