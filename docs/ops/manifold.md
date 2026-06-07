# Manifold Operations Documentation

## Overview

Riemannian manifold operations for hyperspherical neural networks, with Triton kernel optimization.

## Functions

### riemannian_manifold_linear()

Applies Riemannian manifold linear projection.

```python
def riemannian_manifold_linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    kappa: torch.Tensor,
    lambda_rate: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    rule: str = 'near',
    op: str = 'triton'
) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| input_tensor | torch.Tensor | - | Input [batch_size, in_features] |
| weight | torch.Tensor | - | Weights [out_features, in_features] |
| kappa | torch.Tensor | - | vMF concentration parameter |
| lambda_rate | torch.Tensor | - | Gravitational attraction coefficient |
| scale | torch.Tensor | - | Vector amplifier |
| bias | torch.Tensor | - | Manifold bias |
| rule | str | 'near' | Attraction rule ('near' or 'far') |
| op | str | 'triton' | Operation mode ('triton' or 'pytorch') |

**Returns:** Output tensor [batch_size, out_features].

#### Example Usage

```python
import torch
from codon.ops.manifold import riemannian_manifold_linear

batch, in_f, out_f = 32, 128, 64

x = torch.randn(batch, in_f, device='cuda')
w = torch.randn(out_f, in_f, device='cuda')
kappa = torch.ones(out_f, device='cuda')
lambda_rate = torch.ones(out_f, device='cuda') * 0.5
scale = torch.ones(out_f, device='cuda')
bias = torch.zeros(out_f, device='cuda')

output = riemannian_manifold_linear(
    x, w, kappa, lambda_rate, scale, bias,
    rule='near'
)
print(f"Output shape: {output.shape}")  # [32, 64]
```

---

### riemannian_manifold_conv2d()

Applies Riemannian manifold 2D convolution.

```python
def riemannian_manifold_conv2d(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    weight_ones: torch.Tensor,
    kappa: torch.Tensor,
    lambda_rate: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    rule: str = 'near',
    use_norm: bool = False,
    op: str = 'triton'
) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| input_tensor | torch.Tensor | - | Input [batch, in_channels, H, W] |
| weight | torch.Tensor | - | Conv weights [out_channels, in_channels, kH, kW] |
| weight_ones | torch.Tensor | - | All-ones kernel for norm calculation |
| kappa | torch.Tensor | - | vMF concentration parameter |
| lambda_rate | torch.Tensor | - | Gravitational attraction coefficient |
| scale | torch.Tensor | - | Vector amplifier |
| bias | torch.Tensor | - | Manifold bias |
| stride | int | 1 | Convolution stride |
| padding | int | 0 | Convolution padding |
| dilation | int | 1 | Convolution dilation |
| rule | str | 'near' | Attraction rule |
| use_norm | bool | False | Scale output by patch norm |
| op | str | 'triton' | Operation mode |

#### Example Usage

```python
import torch
from codon.ops.manifold import riemannian_manifold_conv2d

batch, in_c, out_c, k = 8, 64, 128, 3
h, w = 32, 32

x = torch.randn(batch, in_c, h, w, device='cuda')
weight = torch.randn(out_c, in_c, k, k, device='cuda')
weight_ones = torch.ones(1, in_c, k, k, device='cuda')
kappa = torch.ones(out_c, device='cuda')
lambda_rate = torch.ones(out_c, device='cuda') * 0.5
scale = torch.ones(out_c, device='cuda')
bias = torch.zeros(out_c, device='cuda')

output = riemannian_manifold_conv2d(
    x, weight, weight_ones, kappa, lambda_rate, scale, bias,
    stride=1, padding=1
)
print(f"Output shape: {output.shape}")  # [8, 128, 32, 32]
```

---

## Mathematical Background

### von Mises-Fisher Distribution

The operations use the vMF distribution for modeling directional data on hyperspheres:
- `kappa`: Concentration parameter (higher = more concentrated)
- `exp(kappa * (cosine - 1))`: vMF-based attraction

### Attraction Rules

- **'near'**: Points are attracted to nearby points on the manifold
- **'far'**: Points are repelled from nearby points (1 - vMF probability)

### Effective Angle

```
effective_theta = theta * (1 - lambda * attraction)
```

---

## Notes

1. **Triton Optimization**: Uses Triton kernels when available on CUDA.
2. **Numerical Stability**: Clamps cosine values to avoid NaN in acos.
3. **Hybrid Approach**: Uses cuBLAS for matrix multiplication, Triton for element-wise fusion.