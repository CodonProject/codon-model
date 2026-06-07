# Fusion Module Documentation

## Overview

The fusion module provides various multimodal fusion techniques for combining features from different modalities.

## Classes

### LowRankFusion

Low-rank Multimodal Fusion (LMF) module.

#### Constructor

```python
LowRankFusion(
    in_features: List[int],
    out_features: int,
    rank: int,
    dropout: float = 0.0,
    channel_first: bool = False
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| in_features | List[int] | - | Feature dimensions for each modality |
| out_features | int | - | Output feature dimension |
| rank | int | - | Rank of low-rank decomposition |
| dropout | float | 0.0 | Dropout probability |
| channel_first | bool | False | Whether feature dim is first |

#### forward()

```python
def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor
```

**Example Usage:**

```python
import torch
from codon.block import LowRankFusion

fusion = LowRankFusion(
    in_features=[512, 768],  # Visual, Text
    out_features=1024,
    rank=128
)

visual = torch.randn(2, 64, 512)
text = torch.randn(2, 64, 768)
output = fusion([visual, text])
print(f"Output shape: {output.shape}")  # [2, 64, 1024]
```

---

### GatedMultimodalUnit

Gated Multimodal Unit (GMU) module.

#### Constructor

```python
GatedMultimodalUnit(
    in_features: List[int],
    out_features: int,
    channel_first: bool = False
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| in_features | List[int] | - | Feature dimensions for each modality |
| out_features | int | - | Hidden layer feature dimension |
| channel_first | bool | False | Whether feature dim is first |

#### forward()

```python
def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor
```

#### Example Usage:

```python
import torch
from codon.block import GatedMultimodalUnit

gmu = GatedMultimodalUnit(
    in_features=[512, 768],
    out_features=1024
)

visual = torch.randn(2, 64, 512)
text = torch.randn(2, 64, 768)
output = gmu([visual, text])
print(f"Output shape: {output.shape}")  # [2, 64, 1024]
```

---

### DiffusionMapsFusion

Diffusion Maps Fusion module.

#### Constructor

```python
DiffusionMapsFusion(
    in_features: List[int],
    out_features: int,
    sigma: float = 1.0,
    channel_first: bool = False
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| in_features | List[int] | - | Feature dimensions for two modalities |
| out_features | int | - | Output feature dimension |
| sigma | float | 1.0 | Bandwidth parameter for Gaussian kernel |
| channel_first | bool | False | Whether feature dim is first |

**Note:** Currently only supports exactly 2 modalities.

#### forward()

```python
def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor
```

#### Example Usage:

```python
import torch
from codon.block import DiffusionMapsFusion

fusion = DiffusionMapsFusion(
    in_features=[512, 768],
    out_features=1024,
    sigma=0.5
)

modal1 = torch.randn(2, 64, 512)
modal2 = torch.randn(2, 64, 768)
output = fusion([modal1, modal2])
print(f"Output shape: {output.shape}")  # [2, 64, 1024]
```

---

### CompactMultimodalPooling

Compact Multimodal Pooling (MCB/CBP) module.

#### Constructor

```python
CompactMultimodalPooling(
    in_features: List[int],
    out_features: int,
    channel_first: bool = False
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| in_features | List[int] | - | Feature dimensions for each modality |
| out_features | int | - | Output feature dimension |
| channel_first | bool | False | Whether feature dim is first |

#### forward()

```python
def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor
```

#### Example Usage:

```python
import torch
from codon.block import CompactMultimodalPooling

fusion = CompactMultimodalPooling(
    in_features=[512, 768, 256],  # Three modalities
    out_features=1024
)

modal1 = torch.randn(2, 64, 512)
modal2 = torch.randn(2, 64, 768)
modal3 = torch.randn(2, 64, 256)
output = fusion([modal1, modal2, modal3])
print(f"Output shape: {output.shape}")  # [2, 64, 1024]
```

---

## Usage Patterns

### Multimodal Transformer

```python
import torch.nn as nn
from codon.block import LowRankFusion, MultiHeadAttention

class MultimodalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_proj = nn.Linear(512, 768)
        self.text_proj = nn.Linear(768, 768)
        self.fusion = LowRankFusion([768, 768], 768, rank=64)
        self.attention = MultiHeadAttention(768, 12)
    
    def forward(self, visual, text):
        visual = self.visual_proj(visual)
        text = self.text_proj(text)
        fused = self.fusion([visual, text])
        output = self.attention(fused)
        return output

model = MultimodalTransformer()
visual = torch.randn(2, 64, 512)
text = torch.randn(2, 64, 768)
output = model(visual, text)
print(f"Output shape: {output.output.shape}")  # [2, 64, 768]
```

---

## Fusion Technique Comparison

| Technique | Complexity | Use Case |
|-----------|------------|----------|
| LowRankFusion | O(r * d) | General purpose, memory efficient |
| GatedMultimodalUnit | O(d^2) | Needs adaptive weighting |
| DiffusionMapsFusion | O(n^2) | Manifold alignment |
| CompactMultimodalPooling | O(d log d) | High-dimensional inputs |

Where:
- `r`: rank
- `d`: feature dimension  
- `n`: number of samples