# Pixel Shuffle Module Documentation

## Overview

Pixel Shuffle modules for efficient upsampling and downsampling operations. Supports 1D, 2D, and 3D data.

## Classes

### PixelShuffleUpSample

Pixel Shuffle Upsampling Module (Depth-to-Space).

#### Constructor

```python
PixelShuffleUpSample(
    in_channels: int,
    out_channels: int,
    upscale_factor: int,
    dim: int = 2,
    norm: Optional[str] = None,
    activation: str = 'relu',
    dropout: float = 0.0
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| in_channels | int | - | Number of input channels |
| out_channels | int | - | Number of output channels after pixel shuffle |
| upscale_factor | int | - | Factor to increase spatial resolution by |
| dim | int | 2 | Dimensionality of the data (1, 2, or 3) |
| norm | Optional[str] | None | Normalization type |
| activation | str | 'relu' | Activation function type |
| dropout | float | 0.0 | Dropout probability |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| conv | ConvBlock | Convolution block projecting input to intermediate channels |
| dim | int | Dimensionality of the data |
| upscale_factor | int | Upsampling factor |
| out_channels | int | Final output channels |

#### forward()

```python
def forward(input_tensor: torch.Tensor) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| input_tensor | torch.Tensor | Input data [batch_size, in_channels, *spatial_dims] |

**Returns:** Upsampled tensor [batch_size, out_channels, *upsampled_spatial_dims]

#### auto_build()

```python
@staticmethod
def auto_build(
    input_shape: Tuple[int, ...],
    output_shape: Optional[Tuple[int, ...]] = None,
    upscale_factor: Optional[int] = None,
    norm: Optional[str] = None,
    activation: str = 'relu',
    dropout: float = 0.0,
    depth_level: int = 1
) -> nn.Module
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| input_shape | Tuple[int, ...] | - | Input shape without batch size |
| output_shape | Optional[Tuple[int, ...]] | None | Desired output shape |
| upscale_factor | Optional[int] | None | Upsampling factor (auto-detected if None) |
| depth_level | int | 1 | Network depth multiplier |

#### Example Usage

```python
import torch
from codon.block import PixelShuffleUpSample

# 2D Upsampling
upsampler = PixelShuffleUpSample(
    in_channels=64,
    out_channels=32,
    upscale_factor=2,
    dim=2
)

x = torch.randn(2, 64, 32, 32)  # [batch, channels, height, width]
output = upsampler(x)
print(f"Output shape: {output.shape}")  # [2, 32, 64, 64]

# Auto-build based on input/output shapes
upsampler = PixelShuffleUpSample.auto_build(
    input_shape=(64, 32, 32),
    output_shape=(32, 128, 128)  # Will automatically use upscale_factor=4
)
```

---

### UnPixelShuffleDownSample

UnPixel Shuffle Downsampling Module (Space-to-Depth).

#### Constructor

```python
UnPixelShuffleDownSample(
    in_channels: int,
    out_channels: int,
    downscale_factor: int,
    dim: int = 2,
    norm: Optional[str] = None,
    activation: str = 'relu',
    dropout: float = 0.0
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| in_channels | int | - | Number of input channels |
| out_channels | int | - | Number of output channels after convolution |
| downscale_factor | int | - | Factor to decrease spatial resolution by |
| dim | int | 2 | Dimensionality of the data (1, 2, or 3) |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| conv | ConvBlock | Convolution block reducing intermediate channels |
| dim | int | Dimensionality of the data |
| downscale_factor | int | Downsampling factor |
| out_channels | int | Final output channels |

#### forward()

```python
def forward(input_tensor: torch.Tensor) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| input_tensor | torch.Tensor | Input data [batch_size, in_channels, *spatial_dims] |

**Returns:** Downsampled tensor [batch_size, out_channels, *downsampled_spatial_dims]

#### Example Usage

```python
import torch
from codon.block import UnPixelShuffleDownSample

# 2D Downsampling
downsampler = UnPixelShuffleDownSample(
    in_channels=32,
    out_channels=64,
    downscale_factor=2,
    dim=2
)

x = torch.randn(2, 32, 64, 64)
output = downsampler(x)
print(f"Output shape: {output.shape}")  # [2, 64, 32, 32]
```

---

## Operations

### pixel_shuffle()

Performs pixel shuffle operation (depth-to-space) for 1D, 2D, and 3D data.

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
| out_channels | int | Number of output channels |
| dim | int | Dimensionality (1, 2, or 3) |

**Returns:** Upsampled tensor.

**Shape Transformation:**
- 1D: [B, C*r, L] → [B, C, L*r]
- 2D: [B, C*r², H, W] → [B, C, H*r, W*r]
- 3D: [B, C*r³, D, H, W] → [B, C, D*r, H*r, W*r]

#### Example Usage

```python
import torch
from codon.ops.pixelshuffle import pixel_shuffle

# 2D pixel shuffle
x = torch.randn(2, 128, 32, 32)  # channels = out_channels * (r^2) = 32 * 4 = 128
output = pixel_shuffle(x, upscale_factor=2, out_channels=32, dim=2)
print(f"Output shape: {output.shape}")  # [2, 32, 64, 64]
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

**Returns:** Downsampled tensor with channels increased by (downscale_factor^dim).

**Shape Transformation:**
- 1D: [B, C, L] → [B, C*r, L/r]
- 2D: [B, C, H, W] → [B, C*r², H/r, W/r]
- 3D: [B, C, D, H, W] → [B, C*r³, D/r, H/r, W/r]

#### Example Usage

```python
import torch
from codon.ops.pixelshuffle import unpixel_shuffle

# 2D unpixel shuffle
x = torch.randn(2, 32, 64, 64)
output = unpixel_shuffle(x, downscale_factor=2, dim=2)
print(f"Output shape: {output.shape}")  # [2, 128, 32, 32]
```

---

## Usage Patterns

### Autoencoder Upsampling

```python
import torch
from codon.block import PixelShuffleUpSample, ConvBlock

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.upsample = PixelShuffleUpSample.auto_build(
            input_shape=(256, 8, 8),
            output_shape=(out_channels, 64, 64),
            norm='batch',
            activation='relu'
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 8, 8)
        x = self.upsample(x)
        return x
```

### Video Processing (3D)

```python
# 3D upsampling for video
upsampler_3d = PixelShuffleUpSample(
    in_channels=64,
    out_channels=32,
    upscale_factor=2,
    dim=3
)

video = torch.randn(2, 64, 16, 32, 32)  # [batch, channels, depth, height, width]
output = upsampler_3d(video)
print(f"Output shape: {output.shape}")  # [2, 32, 32, 64, 64]
```

---

## Notes

1. **Input Requirements**: For `pixel_shuffle`, input channels must equal `out_channels * (upscale_factor ** dim)`.
2. **Spatial Divisibility**: For `unpixel_shuffle`, spatial dimensions must be divisible by downscale_factor.
3. **Auto-build**: The `auto_build` method can automatically determine the upscale/downscale factor based on input/output shapes.
4. **Adaptive Pooling**: When output shape doesn't match exactly, adaptive pooling is automatically added.