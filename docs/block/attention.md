# Attention Module Documentation

## Overview

The attention module provides implementations of multi-head attention mechanisms, including support for Grouped Query Attention (GQA), QK Normalization, and Gating mechanisms.

## Classes

### MultiHeadAttention

Multi-Head Attention module supporting GQA, QK Normalization, and Gating.

#### Constructor

```python
MultiHeadAttention(
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int = None,
    use_qk_norm: bool = True,
    use_gate: bool = False,
    dropout: float = 0.1,
    bias: bool = True,
    is_causal: bool = True
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| hidden_size | int | - | Size of the hidden layer |
| num_heads | int | - | Number of attention heads |
| num_kv_heads | int | None | Number of key/value heads for GQA (defaults to num_heads) |
| use_qk_norm | bool | True | Whether to apply RMSNorm to queries and keys |
| use_gate | bool | False | Whether to apply a gating mechanism |
| dropout | float | 0.1 | Dropout probability |
| bias | bool | True | Whether to use bias in linear layers |
| is_causal | bool | True | Whether to apply a causal mask |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| q_proj | nn.Linear | Linear layer for query projection |
| k_proj | nn.Linear | Linear layer for key projection |
| v_proj | nn.Linear | Linear layer for value projection |
| o_proj | nn.Linear | Linear layer for output projection |
| q_norm | nn.RMSNorm | Normalization layer for queries (if use_qk_norm) |
| k_norm | nn.RMSNorm | Normalization layer for keys (if use_qk_norm) |
| g_proj | nn.Linear | Linear layer for gating mechanism (if use_gate) |

#### forward()

```python
def forward(
    hidden_states: torch.Tensor,
    kv_states: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
    output_attentions: bool = False,
    position_emb: BasicEmbedding = None,
    embedding_start: int = 0,
    embedding_pos: torch.Tensor = None,
    past_key_value: tuple[torch.Tensor, torch.Tensor] = None,
    use_cache: bool = False
) -> AttentionOutput
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| hidden_states | torch.Tensor | - | Input hidden states |
| kv_states | torch.Tensor | None | Hidden states for keys/values (defaults to hidden_states) |
| attention_mask | torch.Tensor | None | Attention mask |
| output_attentions | bool | False | Whether to output attention weights |
| position_emb | BasicEmbedding | None | Positional embedding module |
| embedding_start | int | 0 | Starting position for embedding |
| embedding_pos | torch.Tensor | None | Explicit position indices for positional embedding |
| past_key_value | tuple | None | Past key-value cache |
| use_cache | bool | False | Whether to use KV cache |

**Returns:** `AttentionOutput` object containing:
- `output`: Output tensor
- `attention_weights`: Attention weights (if output_attentions=True)
- `past_key_value`: KV cache (if use_cache=True)

#### Example Usage

```python
import torch
from codon.block import MultiHeadAttention

# Create attention module
attention = MultiHeadAttention(
    hidden_size=768,
    num_heads=12,
    num_kv_heads=4,  # GQA: 12 query heads, 4 key/value heads
    use_qk_norm=True,
    use_gate=True,
    dropout=0.1
)

# Forward pass
hidden_states = torch.randn(2, 64, 768)  # [batch, seq_len, hidden_size]
output = attention(hidden_states)
print(f"Output shape: {output.output.shape}")  # [2, 64, 768]
```

---

### MultiHeadAttentionKEV

Multi-Head Attention module where K = V (Key and Value are identical). Based on the paper "Do Transformers Need Three Projections?" [arXiv:2606.04032].

#### Constructor

```python
MultiHeadAttentionKEV(
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int = None,
    use_qk_norm: bool = True,
    use_gate: bool = False,
    dropout: float = 0.1,
    bias: bool = True,
    is_causal: bool = True
)
```

**Parameters:** Same as `MultiHeadAttention`.

#### Key Differences from MultiHeadAttention

1. **Single KV Projection**: Uses `kv_proj` instead of separate `k_proj` and `v_proj`
2. **Optimized KV Cache**: Can store a single tensor for both K and V when `optimize_kv_cache=True`
3. **Memory Efficient**: Reduces memory footprint by 50% for KV cache during inference

#### Example Usage

```python
import torch
from codon.block import MultiHeadAttentionKEV

# Create KEV attention module
attention = MultiHeadAttentionKEV(
    hidden_size=768,
    num_heads=12,
    num_kv_heads=4,
    use_qk_norm=True,
    dropout=0.1
)

# Forward pass with KV caching
hidden_states = torch.randn(2, 64, 768)
output = attention(
    hidden_states,
    use_cache=True
)
print(f"KV cache shape: {output.past_key_value.shape}")
```

---

## Usage Notes

### Grouped Query Attention (GQA)

GQA reduces memory bandwidth by sharing key/value heads across multiple query heads:

```python
# GQA configuration
attention = MultiHeadAttention(
    hidden_size=768,
    num_heads=12,       # 12 query heads
    num_kv_heads=4      # 4 key/value heads (shared across 3 query heads each)
)
```

### QK Normalization

QK normalization improves training stability and can enable training with larger batch sizes:

```python
attention = MultiHeadAttention(
    hidden_size=768,
    num_heads=12,
    use_qk_norm=True  # Apply RMSNorm to Q and K before dot-product
)
```

### Gating Mechanism

Gating can improve model performance by dynamically scaling attention outputs:

```python
attention = MultiHeadAttention(
    hidden_size=768,
    num_heads=12,
    use_gate=True  # Apply sigmoid gating to output
)
```

### KV Caching for Inference

Enable KV caching for efficient autoregressive generation:

```python
# Prefill
output = attention(
    hidden_states,
    use_cache=True
)
past_kv = output.past_key_value

# Decode (subsequent steps)
next_token = torch.randn(2, 1, 768)
output = attention(
    next_token,
    past_key_value=past_kv,
    use_cache=True
)
```

---

### MultiHeadFourier

Multi-Head Fourier (MHF) module from 'Caracal: Causal Architecture via Spectral Mixing' [arXiv:2605.00292 cs.LG]. Replaces dense attention with O(L log L) frequency-domain mixing.

#### Constructor

```python
MultiHeadFourier(
    hidden_size: int,
    num_heads: int,
    **kwargs
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| hidden_size | int | - | Size of the hidden layer |
| num_heads | int | - | Number of attention heads |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| pre_conv | nn.Conv1d | Depthwise convolution for input processing |
| ln | nn.LayerNorm | Layer normalization |
| W_V | nn.Linear | Linear projection for value stream |
| W_G1 | nn.Linear | First linear projection for gate stream |
| W_G2 | nn.Conv1d | Depthwise convolution for gate stream |
| linear | nn.Linear | Output linear layer |

#### forward()

```python
def forward(
    hidden_states: torch.Tensor,
    use_cache: bool = False,
    past_key_value: tuple = None,
    **kwargs
) -> AttentionOutput
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| hidden_states | torch.Tensor | - | Input hidden states |
| use_cache | bool | False | Whether to use cache |
| past_key_value | tuple | None | Past cache (conv_state, v_cache, g_cache) |

**Returns:** `AttentionOutput` object containing:
- `output`: Output tensor
- `attention_weights`: None (not computed in Fourier mixing)
- `past_key_value`: Cache tuple (if use_cache=True)

#### Example Usage

```python
import torch
from codon.block import MultiHeadFourier

# Create Fourier attention module
fourier_attn = MultiHeadFourier(
    hidden_size=768,
    num_heads=12
)

# Forward pass
hidden_states = torch.randn(2, 64, 768)  # [batch, seq_len, hidden_size]
output = fourier_attn(hidden_states)
print(f"Output shape: {output.output.shape}")  # [2, 64, 768]

# With KV caching
output = fourier_attn(hidden_states, use_cache=True)
print(f"Cache state shapes: conv={output.past_key_value[0].shape}, v={output.past_key_value[1].shape}, g={output.past_key_value[2].shape}")
```

---

## Fourier Mixing Operation

### apply_fourier_mixing()

Perform causal mixing in the frequency domain via FFT.

```python
def apply_fourier_mixing(
    x_v: torch.Tensor,
    x_g: torch.Tensor,
    seq_len: int
) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| x_v | torch.Tensor | Content stream tensor [batch_size, n_heads, seq_len, d_head] |
| x_g | torch.Tensor | Gate stream tensor [batch_size, n_heads, seq_len, d_head] |
| seq_len | int | Original sequence length L |

**Returns:** Mixed sequence tensor truncated to original length.

**Algorithm:**
1. Pad sequence to length 2*L for causal FFT
2. Apply FFT to both content and gate streams
3. Multiply in frequency domain
4. Apply inverse FFT
5. Truncate to original length

#### Example Usage

```python
import torch
from codon.ops.fourier import apply_fourier_mixing

x_v = torch.randn(2, 12, 64, 64)  # [batch, heads, seq_len, d_head]
x_g = torch.randn(2, 12, 64, 64)

result = apply_fourier_mixing(x_v, x_g, seq_len=64)
print(f"Result shape: {result.shape}")  # [2, 12, 64, 64]
```