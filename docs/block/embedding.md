# Embedding Module Documentation

## Overview

The embedding module provides various positional embedding implementations for transformer models, including sinusoidal embedding and rotary positional embedding (RoPE).

## Classes

### BasicEmbedding (Base Class)

Base class for positional embeddings.

#### Constructor

```python
class BasicEmbedding(BasicModel)
```

#### forward()

```python
def forward(self, x: torch.Tensor, positions: torch.Tensor = None, start_pos: int = 0) -> torch.Tensor
```

---

### SinusoidalEmbedding

Sinusoidal absolute positional encoding.

#### Constructor

```python
SinusoidalEmbedding(model_dim: int, max_len: int = 131072, base: int = 500000)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_dim | int | - | The dimension of the model |
| max_len | int | 131072 | Maximum sequence length |
| base | int | 500000 | Base for computing frequencies |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| model_dim | int | Model dimension |
| max_len | int | Maximum sequence length |
| base | int | Base for frequency computation |
| pe | torch.Tensor | Buffer containing positional encodings |

**Formula:**
```
PE(pos, 2i) = sin(pos / base^(2i/model_dim))
PE(pos, 2i+1) = cos(pos / base^(2i/model_dim))
```

#### Example Usage

```python
import torch
from codon.block import SinusoidalEmbedding

embedding = SinusoidalEmbedding(model_dim=512, max_len=1024)
x = torch.randn(2, 64, 512)
output = embedding(x)
print(f"Output shape: {output.shape}")  # [2, 64, 512]
```

---

### RotaryEmbedding

Rotary Positional Embedding (RoPE).

#### Constructor

```python
RotaryEmbedding(model_dim: int, max_len: int = 131072, base: int = 500000)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_dim | int | - | Model dimension (or head_dim) |
| max_len | int | 131072 | Maximum sequence length |
| base | int | 500000 | Base for computing frequencies |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| cos_cached | torch.Tensor | Cached cosine values |
| sin_cached | torch.Tensor | Cached sine values |

#### forward()

```python
def forward(
    self,
    x: torch.Tensor,
    positions: torch.Tensor = None,
    start_pos: Union[int, torch.Tensor] = 0
) -> torch.Tensor
```

**Input shapes supported:**
- `[Batch, Seq_Len, Dim]`
- `[Batch, Head, Seq_Len, Head_Dim]`

#### Example Usage

```python
import torch
from codon.block import RotaryEmbedding

rope = RotaryEmbedding(model_dim=64, max_len=1024)
x = torch.randn(2, 12, 64, 64)  # [Batch, Heads, Seq_Len, Head_Dim]
output = rope(x)
print(f"Output shape: {output.shape}")  # [2, 12, 64, 64]
```

---

### InterleavedRotaryEmbedding

Interleaved Multimodal Rotary Positional Embedding (MRoPE-Interleave).

#### Constructor

```python
InterleavedRotaryEmbedding(
    model_dim: int,
    max_len: int = 131072,
    base: int = 500000,
    num_axes: int = 3
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_dim | int | - | Model dimension (must be even and divisible by num_axes) |
| max_len | int | 131072 | Maximum sequence length |
| base | int | 500000 | Base for computing frequencies |
| num_axes | int | 3 | Number of positional axes (e.g., 3 for time, height, width) |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| num_axes | int | Number of positional axes |
| axis_mask | torch.Tensor | Mask for assigning frequency channels to axes |
| interleave_idx | torch.Tensor | Indices for interleaving |

#### forward()

```python
def forward(
    self,
    x: torch.Tensor,
    positions: torch.Tensor = None,
    start_pos: int = 0
) -> torch.Tensor
```

**Input shapes supported:**
- `[Batch, Seq_Len, Dim]`
- `[Batch, Head, Seq_Len, Head_Dim]`

**Position formats:**
- 1D: `[Batch, Seq_Len]` - automatically expanded for all axes
- 2D: `[Batch, Seq_Len, num_axes]` - explicit positions per axis

#### Example Usage

```python
import torch
from codon.block import InterleavedRotaryEmbedding

# For video: time, height, width axes
rope = InterleavedRotaryEmbedding(
    model_dim=128,
    num_axes=3
)

x = torch.randn(2, 12, 64, 128)  # [Batch, Heads, Seq_Len, Head_Dim]

# Positions: [Batch, Seq_Len, num_axes]
positions = torch.randint(0, 100, (2, 64, 3))

output = rope(x, positions=positions)
print(f"Output shape: {output.shape}")  # [2, 12, 64, 128]
```

---

## Usage Patterns

### Using RoPE with Attention

```python
import torch
from codon.block import MultiHeadAttention, RotaryEmbedding

# Create attention with RoPE
hidden_size = 768
num_heads = 12
head_dim = hidden_size // num_heads

rope = RotaryEmbedding(model_dim=head_dim)
attention = MultiHeadAttention(
    hidden_size=hidden_size,
    num_heads=num_heads,
    use_qk_norm=True
)

# Forward pass with positional embedding
x = torch.randn(2, 64, 768)
output = attention(
    hidden_states=x,
    position_emb=rope,
    embedding_start=0
)
print(f"Output shape: {output.output.shape}")  # [2, 64, 768]
```

### Using Interleaved RoPE for Vision

```python
import torch
from codon.block import MultiHeadAttention, InterleavedRotaryEmbedding

# For 2D vision inputs
head_dim = 64
rope = InterleavedRotaryEmbedding(
    model_dim=head_dim,
    num_axes=2  # height, width
)

attention = MultiHeadAttention(
    hidden_size=head_dim * 8,
    num_heads=8,
    is_causal=False  # Not causal for vision
)

# Create grid positions
num_patches_h, num_patches_w = 16, 16
positions_h = torch.arange(num_patches_h)
positions_w = torch.arange(num_patches_w)
grid_h, grid_w = torch.meshgrid(positions_h, positions_w, indexing='ij')
positions = torch.stack([grid_h.flatten(), grid_w.flatten()], dim=-1)
positions = positions.unsqueeze(0).float()  # [1, 256, 2]

x = torch.randn(1, 256, 512)  # [Batch, Num_Patches, Hidden_Size]
output = attention(
    hidden_states=x,
    position_emb=rope,
    embedding_pos=positions
)
print(f"Output shape: {output.output.shape}")  # [1, 256, 512]
```

### KV Cache with RoPE

```python
import torch
from codon.block import RotaryEmbedding

rope = RotaryEmbedding(model_dim=64, max_len=1024)

# Prefill
x_prefill = torch.randn(1, 32, 64)
output_prefill = rope(x_prefill, start_pos=0)

# Decode (with KV cache)
x_decode = torch.randn(1, 1, 64)
output_decode = rope(x_decode, start_pos=32)  # Continue from position 32

print(f"Prefill output: {output_prefill.shape}")   # [1, 32, 64]
print(f"Decode output: {output_decode.shape}")    # [1, 1, 64]
```

---

## Notes

1. **RoPE vs Sinusoidal**: RoPE is generally preferred for modern transformers as it maintains relative positional information better.
2. **Head Dimension**: For multi-head attention, RoPE is typically applied per-head, so `model_dim` should be `hidden_size // num_heads`.
3. **KV Cache**: When using KV cache, use `start_pos` to correctly position the embeddings.
4. **Multimodal**: Use `InterleavedRotaryEmbedding` for multimodal inputs with multiple positional axes.
5. **Validation**: Both `SinusoidalEmbedding` and `RotaryEmbedding` validate the `max_len` and `base` parameters to ensure numerical stability.