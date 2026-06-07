# Convolution Module Documentation

## Overview

The convolution module provides various convolutional building blocks for deep learning, including causal convolutions, residual blocks, and depthwise separable convolutions.

## Functions

### calculate_causal_layer()

Calculates the required number of layers and receptive field for causal convolution.

```python
def calculate_causal_layer(step: int, kernel_size: int = 3) -> Tuple[int, int]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| step | int | - | Target sequence length or number of time steps |
| kernel_size | int | 3 | Kernel size |

**Returns:** Tuple of `(L, R)` where:
- `L`: Required number of layers
- `R`: Final receptive field size

**Example:**

```python
from codon.block import calculate_causal_layer

layers, receptive_field = calculate_causal_layer(step=1024, kernel_size=3)
print(f"Layers needed: {layers}")
print(f"Receptive field: {receptive_field}")
```

---

## Classes

### CausalConv1d

Causal 1D Convolution layer implemented via dilated convolution.

#### Constructor

```python
CausalConv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    dilation: int = 1,
    norm: str = None,
    activation: str = 'leaky_relu',
    leaky_relu: float = 0.1,
    use_res: bool = True,
    dropout: float = 0.2
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| in_channels | int | - | Number of input channels |
| out_channels | int | - | Number of output channels |
| kernel_size | int | 3 | Kernel size |
| dilation | int | 1 | Dilation factor |
| norm | str | None | Normalization type ('batch', 'group', 'layer', 'instance') |
| activation | str | 'leaky_relu' | Activation function type |
| leaky_relu | float | 0.1 | Negative slope for LeakyReLU |
| use_res | bool | True | Whether to use residual connection |
| dropout | float | 0.2 | Dropout probability |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| block | ConvBlock | Main convolution block |
| downsample | ConvBlock | Downsampling layer for residual connection |
| padding | int | Amount of padding applied |
| use_res | bool | Whether residual connection is enabled |

#### forward()

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

**Input shape:** `[Batch, in_channels, Seq_Len]`

**Output shape:** `[Batch, out_channels, Seq_Len]`

#### Example Usage

```python
import torch
from codon.block import CausalConv1d

conv = CausalConv1d(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    dilation=2,
    norm='batch',
    activation='leaky_relu'
)

x = torch.randn(2, 64, 128)
output = conv(x)
print(f"Output shape: {output.shape}")  # [2, 128, 128]
```

#### Static Methods

**auto_block()** - Automatically builds multiple causal convolution blocks:

```python
@staticmethod
def auto_block(
    in_channels: int,
    out_channels: int,
    step: int,
    kernel_size: int = 3,
    norm: str = None,
    activation: str = 'leaky_relu',
    leaky_relu: float = 0.1,
    use_res: bool = True,
    dropout: float = 0.2
) -> nn.Sequential
```

**manual_block()** - Manually builds causal convolution blocks:

```python
@staticmethod
def manual_block(
    in_channels: int,
    num_channels: List[int],
    kernel_size: int = 3,
    ...
) -> nn.Sequential
```

---

### ConvBlock

General Convolution Block (Conv-Norm-Act-Dropout).

#### Constructor

```python
ConvBlock(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, ...]] = 3,
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...], str] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
    bias: bool = True,
    dim: int = 2,
    norm: str = 'batch',
    activation: str = 'relu',
    dropout: float = 0.0,
    pre_norm: bool = False,
    leaky_relu: float = 0.1
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| in_channels | int | - | Number of input channels |
| out_channels | int | - | Number of output channels |
| kernel_size | int or tuple | 3 | Kernel size |
| stride | int or tuple | 1 | Stride |
| padding | int or tuple or str | 0 | Padding |
| dilation | int or tuple | 1 | Dilation factor |
| groups | int | 1 | Number of groups |
| bias | bool | True | Whether to use bias |
| dim | int | 2 | Convolution dimension (1, 2, 3) |
| norm | str | 'batch' | Normalization type |
| activation | str | 'relu' | Activation function type |
| dropout | float | 0.0 | Dropout probability |
| pre_norm | bool | False | Whether to use Pre-Norm structure |
| leaky_relu | float | 0.1 | Negative slope for LeakyReLU |

#### forward()

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

**Example Usage:**

```python
import torch
from codon.block import ConvBlock

conv_block = ConvBlock(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    norm='batch',
    activation='gelu',
    dim=2
)

x = torch.randn(2, 3, 32, 32)
output = conv_block(x)
print(f"Output shape: {output.shape}")  # [2, 64, 32, 32]
```

---

### DepthwiseSeparableConv

Depthwise Separable Convolution Block.

#### Constructor

```python
DepthwiseSeparableConv(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, ...]] = 3,
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...], str] = 1,
    dilation: Union[int, Tuple[int, ...]] = 1,
    bias: bool = False,
    dim: int = 2,
    norm: str = 'batch',
    activation: str = 'relu',
    dropout: float = 0.0,
    use_res: bool = True
)
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| depthwise | ConvBlock | Depthwise convolution block |
| pointwise | ConvBlock | Pointwise convolution block |
| use_res | bool | Whether to use residual connection |

#### Example Usage

```python
import torch
from codon.block import DepthwiseSeparableConv

conv = DepthwiseSeparableConv(
    in_channels=32,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    norm='batch',
    activation='relu'
)

x = torch.randn(2, 32, 32, 32)
output = conv(x)
print(f"Output shape: {output.shape}")  # [2, 64, 32, 32]
```

---

### ResBasicBlock

Residual Basic Block for ResNet-style architectures.

#### Constructor

```python
ResBasicBlock(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, ...]] = 3,
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...], str] = 1,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
    bias: bool = False,
    dim: int = 2,
    norm: str = 'batch',
    activation: str = 'relu',
    dropout: float = 0.0,
    variant: str = 'original'
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| variant | str | 'original' | Variant type ('original' or 'pre_act') |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| conv1 | ConvBlock | First convolution block |
| conv2 | ConvBlock | Second convolution block |
| downsample | ConvBlock | Downsampling block |
| act | nn.Module | Activation function |
| variant | str | Variant type |

#### Example Usage

```python
import torch
from codon.block import ResBasicBlock

block = ResBasicBlock(
    in_channels=64,
    out_channels=128,
    stride=2,
    norm='batch',
    activation='relu',
    variant='pre_act'
)

x = torch.randn(2, 64, 32, 32)
output = block(x)
print(f"Output shape: {output.shape}")  # [2, 128, 16, 16]
```

---

## Usage Patterns

### Building a WaveNet-like Model

```python
import torch.nn as nn
from codon.block import CausalConv1d

class WaveNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, num_layers=10):
        super().__init__()
        self.layers = nn.Sequential(*[
            CausalConv1d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=3,
                dilation=2 ** i,
                norm='layer',
                activation='leaky_relu'
            )
            for i in range(num_layers)
        ])
    
    def forward(self, x):
        return self.layers(x)

model = WaveNet(num_layers=10)
x = torch.randn(2, 1, 1024)
output = model(x)
print(f"Output shape: {output.shape}")  # [2, 256, 1024]
```

### Building a Simple ResNet

```python
import torch.nn as nn
from codon.block import ResBasicBlock

class SimpleResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.layer1 = nn.Sequential(
            ResBasicBlock(64, 64),
            ResBasicBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            ResBasicBlock(64, 128, stride=2),
            ResBasicBlock(128, 128)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

model = SimpleResNet()
x = torch.randn(2, 3, 224, 224)
output = model(x)
print(f"Output shape: {output.shape}")  # [2, 128, 56, 56]
```