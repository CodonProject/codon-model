# Patch Discriminator Documentation

## Overview

PatchGAN discriminator for adversarial training, outputting N×N predictions instead of a single scalar.

## Classes

### PatchDiscriminator

PatchGAN discriminator where each output point represents whether a patch is real or fake.

#### Constructor

```python
PatchDiscriminator(
    in_channels: int = 3,
    hidden_dim: int = 64,
    num_layers: int = 3,
    norm: str = 'batch',
    activation: str = 'leaky_relu',
    leaky_relu: float = 0.2
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| in_channels | int | 3 | Input image channels |
| hidden_dim | int | 64 | Base number of filters |
| num_layers | int | 3 | Number of discriminator layers |
| norm | str | 'batch' | Normalization type |
| activation | str | 'leaky_relu' | Activation function |
| leaky_relu | float | 0.2 | LeakyReLU negative slope |

#### forward()

```python
def forward(input_tensor: torch.Tensor) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| input_tensor | torch.Tensor | Input images [batch, in_channels, H, W] |

**Returns:** Patch predictions [batch, 1, H', W'].

#### auto_build()

```python
@staticmethod
def auto_build(
    in_channels: int,
    hidden_dim: int,
    image_size: int,
    norm: str = 'batch',
    activation: str = 'leaky_relu',
    leaky_relu: float = 0.2
) -> PatchDiscriminator
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| in_channels | int | Input channels |
| hidden_dim | int | Base filter count |
| image_size | int | Input image size |

#### Example Usage

```python
import torch
from codon.model import PatchDiscriminator

# Manual construction
disc = PatchDiscriminator(
    in_channels=3,
    hidden_dim=64,
    num_layers=3
)

x = torch.randn(2, 3, 256, 256)
output = disc(x)
print(f"Output shape: {output.shape}")  # [2, 1, 30, 30]

# Auto-build based on image size
disc = PatchDiscriminator.auto_build(
    in_channels=3,
    hidden_dim=64,
    image_size=256
)
```

---

## Usage Patterns

### GAN Training

```python
from codon.model import PatchDiscriminator

discriminator = PatchDiscriminator.auto_build(
    in_channels=3,
    hidden_dim=64,
    image_size=256
)

# Real images
real_output = discriminator(real_images)
real_loss = torch.mean((real_output - 1) ** 2)

# Fake images
fake_output = discriminator(fake_images.detach())
fake_loss = torch.mean(fake_output ** 2)

d_loss = (real_loss + fake_loss) / 2
```

---

## Notes

1. **Patch Output**: Output is N×N matrix, not scalar.
2. **Channel Multiplier**: Channels increase as 64 → 128 → 256 → ... (capped at 512).
3. **Auto-build**: Automatically determines num_layers based on image size.