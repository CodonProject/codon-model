# Transformer Module Documentation

## Overview

The transformer module provides decoder layers for building transformer-based models, supporting both dense MLP and Mixture-of-Experts (MoE) feed-forward networks.

## Classes

### TransformerDenseDecoder

Transformer Decoder layer with a dense MLP feed-forward network.

#### Constructor

```python
TransformerDenseDecoder(
    model_dim: int = 1024,
    num_heads: int = 16,
    num_kv_heads: int = 4,
    mlp_ratio: float = 4.0,
    use_mlp_gate: bool = False,
    use_qk_norm: bool = True,
    use_attn_gate: bool = False,
    attn_type: str = 'multihead',
    use_swiglu: bool = False,
    dropout: float = 0.1,
    attn_bias: bool = False,
    mlp_bias: bool = False,
    idx: Union[int, str] = None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_dim | int | 1024 | Model dimension |
| num_heads | int | 16 | Number of attention heads |
| num_kv_heads | int | 4 | Number of KV heads for GQA |
| mlp_ratio | float | 4.0 | Ratio of MLP hidden dimension to model dimension |
| use_mlp_gate | bool | False | Whether to use gating in MLP |
| use_qk_norm | bool | True | Whether to apply RMSNorm to queries and keys |
| use_attn_gate | bool | False | Whether to apply gating to attention output |
| attn_type | str | 'multihead' | Attention type: 'multihead' or 'fourier' |
| use_swiglu | bool | False | Whether to use SwiGLU activation |
| dropout | float | 0.1 | Dropout probability |
| attn_bias | bool | False | Whether to use bias in attention projections |
| mlp_bias | bool | False | Whether to use bias in MLP layers |
| idx | Union[int, str] | None | Layer identifier |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| idx | str | Layer identifier |
| model_dim | int | Model dimension |
| num_heads | int | Number of attention heads |
| attn_norm | nn.RMSNorm | Pre-attention normalization |
| attn | MultiHeadAttention | Multi-head attention module |
| fn_norm | nn.RMSNorm | Pre-feed-forward normalization |
| mlp | MLP | Multi-layer perceptron module |

#### forward()

```python
def forward(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor = None,
    output_attentions: bool = False,
    position_emb: BasicEmbedding = None,
    embedding_start: Union[int, torch.Tensor] = 0,
    embedding_pos: torch.Tensor = None,
    past_key_value: tuple[torch.Tensor, torch.Tensor] = None,
    use_cache: bool = False
) -> TransformerDecoderOutput
```

**Returns:** `TransformerDecoderOutput` object containing:
- `idx`: Layer identifier
- `output`: Output hidden states
- `attention_weights`: Attention weights (if output_attentions=True)
- `attention_mask`: Attention mask used
- `aux_loss`: Auxiliary loss (None for dense decoder)
- `past_key_value`: KV cache (if use_cache=True)
- `use_emb`: Positional embedding module used
- `emb_start`: Start position for embedding
- `emb_pos`: Explicit positions for embedding

#### Example Usage

```python
import torch
from codon.block import TransformerDenseDecoder

# Create decoder layer
decoder = TransformerDenseDecoder(
    model_dim=768,
    num_heads=12,
    num_kv_heads=4,
    mlp_ratio=4.0,
    use_swiglu=True,
    dropout=0.1
)

# Forward pass
hidden_states = torch.randn(2, 64, 768)
output = decoder(hidden_states)
print(f"Output shape: {output.output.shape}")  # [2, 64, 768]
```

---

### TransformerMoEDecoder

Transformer Decoder layer with a Mixture-of-Experts (MoE) feed-forward network.

#### Constructor

```python
TransformerMoEDecoder(
    model_dim: int = 1024,
    num_heads: int = 16,
    num_kv_heads: int = 4,
    top_k: int = 2,
    num_experts: int = 8,
    num_shared_experts: int = 0,
    use_aux_loss: bool = False,
    use_expert_gate: bool = False,
    use_qk_norm: bool = True,
    use_attn_gate: bool = False,
    attn_type: str = 'multihead',
    dropout: float = 0.1,
    attn_bias: bool = False,
    idx: Union[int, str] = None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_dim | int | 1024 | Model dimension |
| num_heads | int | 16 | Number of attention heads |
| num_kv_heads | int | 4 | Number of KV heads for GQA |
| top_k | int | 2 | Number of experts to select per token |
| num_experts | int | 8 | Total number of experts |
| num_shared_experts | int | 0 | Number of shared experts |
| use_aux_loss | bool | False | Whether to compute auxiliary loss for load balancing |
| use_expert_gate | bool | False | Whether to use Gated MLP for experts |
| use_qk_norm | bool | True | Whether to apply RMSNorm to queries and keys |
| use_attn_gate | bool | False | Whether to apply gating to attention output |
| attn_type | str | 'multihead' | Attention type |
| dropout | float | 0.1 | Dropout probability |
| attn_bias | bool | False | Whether to use bias in attention projections |
| idx | Union[int, str] | None | Layer identifier |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| moe | MoE | Mixture-of-Experts module |

#### Example Usage

```python
import torch
from codon.block import TransformerMoEDecoder

# Create MoE decoder layer
decoder = TransformerMoEDecoder(
    model_dim=768,
    num_heads=12,
    num_kv_heads=4,
    top_k=2,
    num_experts=8,
    use_aux_loss=True,
    dropout=0.1
)

# Forward pass
hidden_states = torch.randn(2, 64, 768)
output = decoder(hidden_states)
print(f"Output shape: {output.output.shape}")  # [2, 64, 768]
print(f"Aux loss: {output.aux_loss}")
```

---

### _TransformerDecoder (Base Class)

Abstract base class for all transformer decoder layers.

#### Methods

| Method | Description |
|--------|-------------|
| `forward()` | Standard forward pass |
| `forward_dc()` | Forward pass using TransformerDecoderOutput object |
| `flow()` | Abstract method for feed-forward network (must be implemented by subclasses) |

---

## Data Classes

### FlowOutput

Output from the flow (feed-forward) layer.

```python
@dataclass
class FlowOutput:
    output: torch.Tensor
    aux_loss: Optional[torch.Tensor] = None
```

### TransformerDecoderOutput

Output from the Transformer Decoder layer.

```python
@dataclass
class TransformerDecoderOutput:
    idx: str
    output: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    use_emb: Optional[BasicEmbedding] = None
    emb_start: Optional[int] = 0
    emb_pos: Optional[torch.Tensor] = None
```

---

## Usage Patterns

### Building a Transformer Decoder Stack

```python
import torch
import torch.nn as nn
from codon.block import TransformerDenseDecoder

class TransformerModel(nn.Module):
    def __init__(self, model_dim=768, num_layers=12, num_heads=12):
        super().__init__()
        self.decoder = nn.ModuleList([
            TransformerDenseDecoder(
                model_dim=model_dim,
                num_heads=num_heads,
                num_kv_heads=num_heads // 3,  # GQA
                use_swiglu=True,
                dropout=0.1,
                idx=str(i)
            )
            for i in range(num_layers)
        ])
    
    def forward(self, hidden_states):
        for layer in self.decoder:
            output = layer(hidden_states)
            hidden_states = output.output
        return hidden_states

# Create model
model = TransformerModel(model_dim=768, num_layers=12, num_heads=12)
x = torch.randn(2, 64, 768)
output = model(x)
print(f"Final output shape: {output.shape}")  # [2, 64, 768]
```

### Using SwiGLU Activation

SwiGLU is recommended for better performance in large models:

```python
decoder = TransformerDenseDecoder(
    model_dim=768,
    num_heads=12,
    use_swiglu=True,  # Uses SwiGLU activation in MLP
    mlp_ratio=4.0     # Hidden dimension = 768 * 4 = 3072
)
```

### MoE with Auxiliary Loss

For MoE models, enable auxiliary loss for better load balancing:

```python
decoder = TransformerMoEDecoder(
    model_dim=768,
    num_heads=12,
    num_experts=8,
    top_k=2,
    use_aux_loss=True  # Enable load balancing loss
)
```