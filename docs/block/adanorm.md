# Adaptive Normalization Documentation

## Overview

Adaptive normalization modules that apply scale and shift parameters computed from conditional embeddings.

## Classes

### AdaLayerNorm

Adaptive Layer Normalization module.

#### Constructor

```python
AdaLayerNorm(
    features_dim: int,
    embedding_dim: int,
    hidden_features: int = None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| features_dim | int | - | Dimension of input features |
| embedding_dim | int | - | Dimension of embedding features |
| hidden_features | int | None | Hidden dimension in MLP (defaults to features_dim) |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| features_dim | int | Input feature dimension |
| embedding_dim | int | Embedding dimension |
| mlp | MLP | MLP predicting scale and shift |

#### forward()

```python
def forward(
    input_tensor: torch.Tensor,
    embedding_tensor: torch.Tensor
) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| input_tensor | torch.Tensor | Input to normalize [batch, ..., features_dim] |
| embedding_tensor | torch.Tensor | Condition embedding [batch, embedding_dim] |

**Returns:** Normalized and modulated output.

#### Example Usage

```python
import torch
from codon.block import AdaLayerNorm

ada_ln = AdaLayerNorm(
    features_dim=768,
    embedding_dim=512
)

x = torch.randn(2, 64, 768)  # Input features
emb = torch.randn(2, 512)    # Condition embedding

output = ada_ln(x, emb)
print(f"Output shape: {output.shape}")  # [2, 64, 768]
```

---

## Usage Patterns

### Conditional Feature Modulation

```python
class ConditionalBlock(nn.Module):
    def __init__(self, features_dim, cond_dim):
        super().__init__()
        self.ada_ln = AdaLayerNorm(features_dim, cond_dim)
        self.linear = nn.Linear(features_dim, features_dim)
    
    def forward(self, x, condition):
        x = self.ada_ln(x, condition)
        x = self.linear(x)
        return x
```

---

## Notes

1. **Scale and Shift**: The MLP outputs 2 * features_dim values, split into scale and shift.
2. **Condition Embedding**: The embedding can come from any source (class label, time step, etc.).
3. **LayerNorm**: Uses PyTorch's F.layer_norm internally.