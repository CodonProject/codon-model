# FiLM Module Documentation

## Overview

The FiLM (Feature-wise Linear Modulation) module provides feature-wise affine transformation for conditional computation, commonly used in conditional generation and multimodal models.

## Data Classes

### FiLMOutput

Output container for the FiLM module.

```python
@dataclass
class FiLMOutput:
    output: torch.Tensor
    gate: Optional[torch.Tensor] = None
    
    @property
    def gated_output(self):
        if self.gate is None: return self.output
        return self.output * torch.tanh(self.gate)
```

---

## Classes

### FiLM

Feature-wise Linear Modulation module.

#### Constructor

```python
FiLM(
    in_features: int,
    cond_features: int,
    use_beta: bool = True,
    use_gamma: bool = True,
    use_gate: bool = True,
    use_context_gate: bool = False,
    channel_first: bool = False
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| in_features | int | - | Dimension of input features |
| cond_features | int | - | Dimension of conditional features |
| use_beta | bool | True | Whether to use translation term |
| use_gamma | bool | True | Whether to use scaling term |
| use_gate | bool | True | Whether to use gating term |
| use_context_gate | bool | False | Whether to use context gating |
| channel_first | bool | False | Whether feature dim is first (CNN) |

**Formula:** `FiLM(x) = (1 + gamma(z)) * x + beta(z)`

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| proj | nn.Linear | Linear layer to project conditional features |
| gate_proj | nn.Module | Linear layer for context gating |

#### forward()

```python
def forward(self, x: torch.Tensor, cond: torch.Tensor) -> FiLMOutput
```

**Input shapes:**
- `x`: `[B, C, ...]` (if channel_first=True) or `[B, ..., C]`
- `cond`: `[B, ..., cond_features]`

**Returns:** `FiLMOutput` containing:
- `output`: Modulated features
- `gate`: Gating values for residual connections

#### Example Usage

```python
import torch
from codon.block import FiLM

film = FiLM(
    in_features=512,
    cond_features=256,
    use_beta=True,
    use_gamma=True,
    use_gate=True,
    channel_first=False
)

x = torch.randn(2, 64, 512)    # [Batch, Seq, Features]
cond = torch.randn(2, 256)     # [Batch, Cond_Features]

output = film(x, cond)
print(f"Output shape: {output.output.shape}")  # [2, 64, 512]
print(f"Gate shape: {output.gate.shape}")      # [2, 64, 512]
```

---

## Usage Patterns

### Conditional Generation

```python
import torch
import torch.nn as nn
from codon.block import FiLM

class ConditionalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 256)  # Encode condition
        self.decoder = nn.LSTM(512, 512, batch_first=True)
        self.film = FiLM(512, 256)
    
    def forward(self, x, condition):
        cond_emb = self.encoder(condition)  # Encode condition
        output, _ = self.decoder(x)
        output = self.film(output, cond_emb)  # Modulate with FiLM
        return output

model = ConditionalGenerator()
x = torch.randn(2, 32, 512)
condition = torch.randn(2, 10)
output = model(x, condition)
print(f"Output shape: {output.output.shape}")  # [2, 32, 512]
```

### Multimodal Fusion

```python
import torch
from codon.block import FiLM

# Visual features conditioned on text
film = FiLM(
    in_features=1024,  # Visual features
    cond_features=768,  # Text features
    use_context_gate=True  # Use context gate for better fusion
)

visual = torch.randn(2, 196, 1024)  # Image patches
text = torch.randn(2, 768)           # Text embedding

output = film(visual, text)
print(f"Output shape: {output.output.shape}")  # [2, 196, 1024]
```

---

## Notes

1. **Initialization**: Parameters are initialized so that gamma=0 and beta=0 initially, resulting in identity mapping.
2. **Context Gate**: Uses concatenation of input and conditional features for gating.
3. **Channel First**: Set `channel_first=True` for CNN-style inputs `[B, C, H, W]`.
4. **Gated Output**: Use `output.gated_output` to apply the gate with tanh activation.