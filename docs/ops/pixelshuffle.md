# Pixel Shuffle Operations Documentation

## Overview

Low-level pixel shuffle operations for efficient upsampling and downsampling across 1D, 2D, and 3D data.

## Functions

### pixel_shuffle()

Performs pixel shuffle operation (depth-to-space).

```python
def pixel_shuffle(
    input_tensor: torch.Tensor,
    upscale_factor: int,
    out_channels: int,
    dim: int
) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| input_tensor | torch.Tensor | Input tensor [batch_size, channels, *spatial_dims] |
| upscale_factor | int | Factor to increase spatial resolution by |
| out_channels | int | Number of output channels after shuffle |
| dim | int | Dimensionality (1, 2, or 3) |

**Input Requirements:**
- Input channels must equal `out_channels * (upscale_factor ** dim)`

**Shape Transformations:**

| Dimension | Input Shape | Output Shape |
|-----------|-------------|--------------|
| 1D | [B, C*r, L] | [B, C, L*r] |
| 2D | [B, C*r², H, W] | [B, C, H*r, W*r] |
| 3D | [B, C*r³, D, H, W] | [B, C, D*r, H*r, W*r] |

#### Example Usage

```python
import torch
from codon.ops.pixelshuffle import pixel_shuffle

# 2D example
x = torch.randn(2, 128, 32, 32)  # channels = 32 * 2^2 = 128
output = pixel_shuffle(x, upscale_factor=2, out_channels=32, dim=2)
print(f"Output shape: {output.shape}")  # [2, 32, 64, 64]

# 1D example
x = torch.randn(2, 64, 128)  # channels = 16 * 2^2 = 64
output = pixel_shuffle(x, upscale_factor=2, out_channels=16, dim=1)
print(f"Output shape: {output.shape}")  # [2, 16, 256]

# 3D example
x = torch.randn(2, 256, 16, 32, 32)  # channels = 32 * 2^3 = 256
output = pixel_shuffle(x, upscale_factor=2, out_channels=32, dim=3)
print(f"Output shape: {output.shape}")  # [2, 32, 32, 64, 64]
```

---

### unpixel_shuffle()

Performs inverse pixel shuffle operation (space-to-depth).

```python
def unpixel_shuffle(
    input_tensor: torch.Tensor,
    downscale_factor: int,
    dim: int
) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| input_tensor | torch.Tensor | Input tensor [batch_size, channels, *spatial_dims] |
| downscale_factor | int | Factor to decrease spatial resolution by |
| dim | int | Dimensionality (1, 2, or 3) |

**Input Requirements:**
- Spatial dimensions must be divisible by `downscale_factor`

**Shape Transformations:**

| Dimension | Input Shape | Output Shape |
|-----------|-------------|--------------|
| 1D | [B, C, L] | [B, C*r, L/r] |
| 2D | [B, C, H, W] | [B, C*r², H/r, W/r] |
| 3D | [B, C, D, H, W] | [B, C*r³, D/r, H/r, W/r] |

#### Example Usage

```python
import torch
from codon.ops.pixelshuffle import unpixel_shuffle

# 2D example
x = torch.randn(2, 32, 64, 64)
output = unpixel_shuffle(x, downscale_factor=2, dim=2)
print(f"Output shape: {output.shape}")  # [2, 128, 32, 32]

# 1D example
x = torch.randn(2, 16, 256)
output = unpixel_shuffle(x, downscale_factor=2, dim=1)
print(f"Output shape: {output.shape}")  # [2, 64, 128]
```

---

## Usage Patterns

### Autoencoder Building Block

```python
import torch
import torch.nn as nn
from codon.ops.pixelshuffle import pixel_shuffle, unpixel_shuffle

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * (upscale_factor ** 2),
            kernel_size=3,
            padding=1
        )
        self.upscale_factor = upscale_factor
        self.out_channels = out_channels
    
    def forward(self, x):
        x = self.conv(x)
        x = pixel_shuffle(x, self.upscale_factor, self.out_channels, dim=2)
        return x

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downscale_factor=2):
        super().__init__()
        self.downscale_factor = downscale_factor
        self.conv = nn.Conv2d(
            in_channels * (downscale_factor ** 2),
            out_channels,
            kernel_size=3,
            padding=1
        )
    
    def forward(self, x):
        x = unpixel_shuffle(x, self.downscale_factor, dim=2)
        x = self.conv(x)
        return x

# Usage
upsampler = UpsampleBlock(64, 32)
downsampler = DownsampleBlock(32, 64)

x = torch.randn(2, 64, 32, 32)
x_up = upsampler(x)      # [2, 32, 64, 64]
x_down = downsampler(x_up)  # [2, 64, 32, 32]
```

---

## Notes

1. **Channel Requirements**: For `pixel_shuffle`, input channels must be exactly `out_channels * (upscale_factor ** dim)`.
2. **Spatial Divisibility**: For `unpixel_shuffle`, spatial dimensions must be divisible by `downscale_factor`.
3. **Memory Efficiency**: Pixel shuffle avoids explicit upsampling operations (like transposed convolution) which can be memory-intensive.
4. **No Learnable Parameters**: These are pure reshaping operations without any learnable parameters.
5. **Dimensionality Support**: Both functions support 1D, 2D, and 3D inputs with the same API.