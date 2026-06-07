# MLP Module Documentation

## Overview

The MLP module provides implementations of Multi-Layer Perceptrons, including standard MLP and Gated MLP architectures like SwiGLU.

## Classes

### MLP

Multilayer Perceptron module supporting standard and gated architectures.

#### Constructor

```python
MLP(
    in_features: int,
    hidden_features: int,
    out_features: int = None,
    bias: bool = True,
    use_gate: bool = False,
    dropout: float = 0.0,
    act_layer: str = 'silu'
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| in_features | int | - | Dimension of input features |
| hidden_features | int | - | Dimension of hidden layer features |
| out_features | int | None | Dimension of output features (defaults to in_features) |
| bias | bool | True | Whether to use bias in linear layers |
| use_gate | bool | False | Whether to use the gating mechanism |
| dropout | float | 0.0 | Dropout probability |
| act_layer | str | 'silu' | Activation function name ('silu', 'gelu', 'relu') |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| fc1 | nn.Linear | First linear layer (standard MLP) |
| fc2 | nn.Linear | Second linear layer (standard MLP) |
| gate_proj | nn.Linear | Gating linear layer (gated MLP) |
| up_proj | nn.Linear | Up-projection linear layer (gated MLP) |
| down_proj | nn.Linear | Down-projection linear layer (gated MLP) |
| act | nn.Module | Activation function (SiLU, GELU, or ReLU) |
| dropout | nn.Dropout | Dropout layer |

#### forward()

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| x | torch.Tensor | Input tensor |

**Returns:** `torch.Tensor` - Output tensor

#### Example Usage

```python
import torch
from codon.block import MLP

# Standard MLP
mlp = MLP(
    in_features=768,
    hidden_features=3072,
    out_features=768,
    dropout=0.1,
    act_layer='gelu'
)

x = torch.randn(2, 64, 768)
output = mlp(x)
print(f"Output shape: {output.shape}")  # [2, 64, 768]
```

---

### SwiGLU Factory Method

Static method to create a SwiGLU MLP module.

```python
@staticmethod
def SwiGLU(
    in_features: int,
    hidden_features: int = None,
    out_features: int = None,
    bias: bool = False,
    dropout: float = 0.0
) -> 'MLP'
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| in_features | int | - | Input dimension |
| hidden_features | int | None | Intermediate dimension (defaults to in_features * 8/3 rounded up to nearest 128) |
| out_features | int | None | Output dimension (defaults to in_features) |
| bias | bool | False | Whether to use bias (LLMs typically set False) |
| dropout | float | 0.0 | Dropout rate |

**Returns:** `MLP` - Configured SwiGLU module

#### SwiGLU Formula

```
output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

#### Example Usage

```python
import torch
from codon.block import MLP

# SwiGLU MLP (common in modern LLMs)
swiglu = MLP.SwiGLU(
    in_features=768,
    bias=False  # LLMs typically use no bias
)

x = torch.randn(2, 64, 768)
output = swiglu(x)
print(f"Output shape: {output.shape}")  # [2, 64, 768]
```

---

## Architecture Variants

### Standard MLP

```python
mlp = MLP(
    in_features=768,
    hidden_features=3072,
    out_features=768,
    use_gate=False,  # Standard MLP
    act_layer='gelu'
)
```

**Architecture:** `fc1 -> act -> dropout -> fc2`

### Gated MLP

```python
mlp = MLP(
    in_features=768,
    hidden_features=3072,
    out_features=768,
    use_gate=True,  # Gated MLP
    act_layer='silu'
)
```

**Architecture:** `down_proj(SiLU(gate_proj(x)) * up_proj(x))`

### SwiGLU (Recommended)

```python
mlp = MLP.SwiGLU(in_features=768)
```

**Architecture:** Same as gated MLP with optimal defaults for LLMs.

---

## Configuration Examples

### LLM-Style MLP

```python
# Typical configuration for large language models
mlp = MLP.SwiGLU(
    in_features=1024,
    bias=False,
    dropout=0.0  # SwiGLU typically doesn't use dropout
)
```

### Vision-Style MLP

```python
# Configuration for vision models
mlp = MLP(
    in_features=512,
    hidden_features=2048,
    out_features=512,
    act_layer='gelu',
    dropout=0.1
)
```

### Small MLP Head

```python
# Classification head
head = MLP(
    in_features=768,
    hidden_features=512,
    out_features=10,  # 10 classes
    act_layer='relu'
)
```

---

## Notes

1. **SwiGLU is recommended** for transformer models due to better performance and stability.
2. **No bias** is standard in LLMs for both attention and MLP layers.
3. **Hidden dimension** is typically 4x the model dimension for standard MLPs, and 8/3x for SwiGLU.
4. **Activation functions**:
   - `silu`: SiLU (Swish) - recommended for SwiGLU
   - `gelu`: GELU - recommended for standard MLPs
   - `relu`: ReLU - simple, but may suffer from vanishing gradients