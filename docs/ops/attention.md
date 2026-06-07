# Attention Operations Documentation

## Overview

Core attention operations for transformer models.

## Data Classes

### AttentionOutput

Output of the attention mechanism.

```python
@dataclass
class AttentionOutput:
    output: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
```

---

## Functions

### apply_attention()

Compute scaled dot-product attention.

```python
def apply_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor = None,
    output_attentions: bool = False,
    is_causal: bool = None,
    dropout: float = 0.0
) -> AttentionOutput
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| query_states | torch.Tensor | - | Query states tensor |
| key_states | torch.Tensor | - | Key states tensor |
| value_states | torch.Tensor | - | Value states tensor |
| attention_mask | torch.Tensor | None | Attention mask |
| output_attentions | bool | False | Whether to output attention weights |
| is_causal | bool | None | Whether to apply causal mask |
| dropout | float | 0.0 | Dropout probability |

**Input shapes:**
- `query_states`: `[batch, heads, q_len, head_dim]`
- `key_states`: `[batch, heads, k_len, head_dim]`  
- `value_states`: `[batch, heads, v_len, head_dim]`

**Returns:** `AttentionOutput` containing:
- `output`: Attention output `[batch, heads, q_len, head_dim]`
- `attention_weights`: Attention weights (if output_attentions=True)
- `past_key_value`: Not used in this function (always None)

#### Example Usage

```python
import torch
from codon.ops import apply_attention, AttentionOutput

# Create dummy tensors
batch_size = 2
num_heads = 12
seq_len = 64
head_dim = 64

Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
K = torch.randn(batch_size, num_heads, seq_len, head_dim)
V = torch.randn(batch_size, num_heads, seq_len, head_dim)

# Apply attention
output = apply_attention(
    query_states=Q,
    key_states=K,
    value_states=V,
    is_causal=True,
    output_attentions=True
)

print(f"Output shape: {output.output.shape}")            # [2, 12, 64, 64]
print(f"Attention weights shape: {output.attention_weights.shape}")  # [2, 12, 64, 64]
```

---

## Usage Patterns

### Causal Attention

```python
output = apply_attention(
    Q, K, V,
    is_causal=True  # Automatically applies causal mask
)
```

### With Attention Mask

```python
# Create padding mask
mask = torch.ones(batch_size, seq_len)
mask[:, 32:] = 0  # Mask out last 32 positions

output = apply_attention(
    Q, K, V,
    attention_mask=mask,
    is_causal=True
)
```

### KV Cache Support

```python
# Prefill
output = apply_attention(Q_prefill, K_prefill, V_prefill, is_causal=True)

# Decode (with cached KV)
# K_cache includes both past and current keys
output = apply_attention(
    Q_decode,  # Only current query
    K_cache,   # Past + current keys
    V_cache,   # Past + current values
    is_causal=True
)
```

---

## Notes

1. **Mixed Precision**: The function automatically handles dtype consistency for mixed-precision operations.
2. **Flash Attention**: When available, uses PyTorch's `F.scaled_dot_product_attention` for optimized performance.
3. **Causal Mask**: Automatically handles KV cache scenarios where `src_len > tgt_len`.
4. **Mask Format**: Accepts both binary masks (0/1) and additive masks (-inf/0).