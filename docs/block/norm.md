# Normalization Documentation

## Overview

Root Mean Square Normalization variants for neural networks.

## Classes

### RMSNorm

Root Mean Square Normalization.

**Formula:** `y = (x / RMS(x)) * gamma`

#### Constructor

```python
RMSNorm(
    d_model: int,
    eps: float = 1e-6,
    channel_first: bool = False
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| d_model | int | - | Feature dimension |
| eps | float | 1e-6 | Numerical stability constant |
| channel_first | bool | False | If True, features in 1st dimension [B, C, ...] |

#### forward()

```python
def forward(x: torch.Tensor) -> torch.Tensor
```

#### Example Usage

```python
import torch
from codon.block import RMSNorm

# Standard usage (last dimension)
norm = RMSNorm(d_model=768)
x = torch.randn(2, 64, 768)
output = norm(x)

# Channel-first usage
norm_cf = RMSNorm(d_model=64, channel_first=True)
x_cf = torch.randn(2, 64, 32, 32)  # [B, C, H, W]
output_cf = norm_cf(x_cf)
```

---

### ZCRMSNorm

Zero-Centered Root Mean Square Normalization.

**Formula:** `y = (x / RMS(x)) * (1 + gamma)`

Gamma is initialized to 0, preserving identity mapping at initialization.

#### Constructor

```python
ZCRMSNorm(
    d_model: int,
    eps: float = 1e-6,
    channel_first: bool = False
)
```

**Parameters:** Same as `RMSNorm`.

#### Example Usage

```python
import torch
from codon.block import ZCRMSNorm

norm = ZCRMSNorm(d_model=768)
x = torch.randn(2, 64, 768)
output = norm(x)
```

---

## Comparison

| Feature | RMSNorm | ZCRMSNorm |
|---------|---------|-----------|
| Gamma init | 1 | 0 |
| Identity at init | No | Yes |
| Gradient flow | Standard | Potentially more stable |

---

## Notes

1. **Numerical Precision**: Operations are performed in float32 for stability.
2. **Channel First**: Supports both [B, ..., C] and [B, C, ...] layouts.
3. **Efficiency**: More efficient than LayerNorm (no mean subtraction).