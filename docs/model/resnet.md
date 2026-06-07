# ResNet Model Documentation

## Overview

Residual Network (ResNet) implementation with automatic architecture building.

## Classes

### ResNet

Residual Network model.

#### Constructor

```python
class ResNet(BasicModel)
```

#### Static Methods

**auto_build()** - Automatically build ResNet based on input/output shape:

```python
@staticmethod
def auto_build(
    input_shape: tuple,
    output_shape: tuple,
    depth_level: int = 1
) -> 'ResNet'
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| input_shape | tuple | - | Input shape (channels, height, width) |
| output_shape | tuple | - | Output shape (features,) |
| depth_level | int | 1 | Depth multiplier |

#### Example Usage

```python
import torch
from codon.model import ResNet

# Build ResNet automatically
model = ResNet.auto_build(
    input_shape=(3, 32, 32),
    output_shape=(512,),
    depth_level=1
)

x = torch.randn(2, 3, 32, 32)
output = model(x)
print(f"Output shape: {output.shape}")  # [2, 512]
```

---

## Usage Patterns

### Feature Extractor

```python
model = ResNet.auto_build(
    input_shape=(3, 224, 224),
    output_shape=(2048,),
    depth_level=3  # Deeper network
)

image = torch.randn(1, 3, 224, 224)
features = model(image)
print(f"Features shape: {features.shape}")  # [1, 2048]
```

---

## Notes

1. **Depth Levels**: Higher depth levels increase the number of residual blocks.
2. **Output Shape**: Should be a tuple with a single integer for feature dimension.
3. **Input Shape**: Should be (channels, height, width) for 2D images.