# LoRA Module Documentation

## Overview

The LoRA (Low-Rank Adaptation) module provides implementations of Low-Rank Adaptation for efficient fine-tuning of large models. It supports various LoRA variants including Gated LoRA and DoRA.

## Classes

### BasicLoRA (Base Class)

Base class for Low-Rank Adaptation modules.

#### Constructor

```python
BasicLoRA(
    original_layer: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    merge_weights: bool = False,
    gate: bool = False,
    dora: bool = False,
    gradient_checkpointing: bool = False
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| original_layer | nn.Module | - | The original layer to be adapted |
| r | int | 8 | Rank of the low-rank adaptation |
| lora_alpha | int | 16 | Scaling factor for LoRA |
| lora_dropout | float | 0.05 | Dropout probability for LoRA path |
| merge_weights | bool | False | Whether to merge LoRA weights upon initialization |
| gate | bool | False | Whether to use Gated LoRA |
| dora | bool | False | Whether to use DoRA |
| gradient_checkpointing | bool | False | Whether to use gradient checkpointing |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| original_layer | nn.Module | Original layer |
| r | int | Rank |
| lora_alpha | int | Scaling factor |
| scaling | float | Actual scaling ratio (lora_alpha / r) |
| merged | bool | Whether weights are merged |
| gate | bool | Whether Gated LoRA is enabled |
| dora | bool | Whether DoRA is enabled |
| lora_gate | nn.Parameter | Gate parameter (if enabled) |
| lora_a | nn.Module | Dimension reduction component |
| lora_b | nn.Module | Dimension expansion component |
| lora_dropout | nn.Module | Dropout layer |
| dora_m | nn.Parameter | Magnitude vector for DoRA |

#### Methods

| Method | Description |
|--------|-------------|
| `reset_parameters()` | Resets LoRA parameters |
| `merge()` | Merges LoRA weights into original layer |
| `unmerge()` | Unmerges LoRA weights |
| `train()` | Sets training mode and ensures weights are unmerged |
| `forward()` | Forward pass with gradient checkpointing support |

---

### LinearLoRA

Implements LoRA for linear layers.

#### Constructor

```python
LinearLoRA(
    original_layer: nn.Linear,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    merge_weights: bool = False,
    gate: bool = False,
    dora: bool = False,
    gradient_checkpointing: bool = False
)
```

**Formula:** `h = W_0 x + B A x * scaling`

#### Example Usage

```python
import torch
import torch.nn as nn
from codon.block import LinearLoRA

# Original linear layer
linear = nn.Linear(768, 768)

# Apply LoRA
lora_linear = LinearLoRA(
    original_layer=linear,
    r=8,
    lora_alpha=16,
    gate=True,
    dora=False
)

x = torch.randn(2, 64, 768)
output = lora_linear(x)
print(f"Output shape: {output.shape}")  # [2, 64, 768]
```

---

### Conv2dLoRA

Implements LoRA for Conv2d layers.

#### Constructor

```python
Conv2dLoRA(
    original_layer: nn.Conv2d,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    merge_weights: bool = False,
    gate: bool = False,
    dora: bool = False,
    gradient_checkpointing: bool = False
)
```

**Architecture:**
1. A layer: Reduces channels to r, maintains kernel_size
2. B layer: Restores channels, uses 1x1 kernel

#### Example Usage

```python
import torch
import torch.nn as nn
from codon.block import Conv2dLoRA

# Original conv layer
conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)

# Apply LoRA
lora_conv = Conv2dLoRA(
    original_layer=conv,
    r=8,
    lora_alpha=16
)

x = torch.randn(2, 64, 32, 32)
output = lora_conv(x)
print(f"Output shape: {output.shape}")  # [2, 128, 32, 32]
```

---

### Conv1dLoRA

Implements LoRA for Conv1d layers.

#### Constructor

```python
Conv1dLoRA(
    original_layer: nn.Conv1d,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    merge_weights: bool = False,
    gate: bool = False,
    dora: bool = False,
    gradient_checkpointing: bool = False
)
```

---

### EmbeddingLoRA

Implements LoRA for Embedding layers.

#### Constructor

```python
EmbeddingLoRA(
    original_layer: nn.Embedding,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    merge_weights: bool = False,
    gate: bool = False,
    dora: bool = False,
    gradient_checkpointing: bool = False
)
```

**Formula:** `h = W_0[idx] + (A[idx] @ B.T) * scaling`

#### Example Usage

```python
import torch
import torch.nn as nn
from codon.block import EmbeddingLoRA

# Original embedding layer
embedding = nn.Embedding(10000, 768)

# Apply LoRA
lora_embedding = EmbeddingLoRA(
    original_layer=embedding,
    r=8,
    lora_alpha=16
)

x = torch.randint(0, 10000, (2, 64))
output = lora_embedding(x)
print(f"Output shape: {output.shape}")  # [2, 64, 768]
```

---

## LoRA Variants

### Standard LoRA

```python
lora = LinearLoRA(
    original_layer=linear,
    r=8,
    lora_alpha=16
)
```

### Gated LoRA

Adds a learnable gate to scale the LoRA contribution:

```python
lora = LinearLoRA(
    original_layer=linear,
    r=8,
    lora_alpha=16,
    gate=True  # Adds learnable gate
)
```

### DoRA (Weight-Decomposed LoRA)

Applies weight normalization to improve training stability:

```python
lora = LinearLoRA(
    original_layer=linear,
    r=8,
    lora_alpha=16,
    dora=True  # Enables DoRA
)
```

**DoRA Formula:**
```
W_final = m * (W0 + BA) / ||W0 + BA||
```

---

## Usage Patterns

### Applying LoRA to a Transformer Model

```python
import torch.nn as nn
from codon.block import LinearLoRA, MultiHeadAttention

class LoRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = MultiHeadAttention(768, 12)
        
        # Apply LoRA to attention projections
        self.attention.q_proj = LinearLoRA(
            self.attention.q_proj,
            r=8,
            lora_alpha=16
        )
        self.attention.v_proj = LinearLoRA(
            self.attention.v_proj,
            r=8,
            lora_alpha=16
        )
    
    def forward(self, x):
        return self.attention(x)

model = LoRAModel()

# Freeze all parameters except LoRA
for name, param in model.named_parameters():
    if 'lora' not in name:
        param.requires_grad = False

# Only LoRA parameters will be updated during training
```

### Merging Weights for Inference

```python
# Training mode
lora_linear.train()

# Switch to inference mode (merge weights for speed)
lora_linear.eval()
lora_linear.merge()  # Merges LoRA weights into original layer

# Now the layer operates as a single linear layer
x = torch.randn(2, 64, 768)
output = lora_linear(x)

# Unmerge if you need to resume training
lora_linear.unmerge()
```

---

## Notes

1. **Rank Selection**: The rank `r` controls the capacity of the LoRA adaptation. Smaller values (e.g., 4-16) are typical.
2. **Scaling**: `lora_alpha / r` determines the scaling factor. A common choice is `lora_alpha = 2 * r`.
3. **Parameter Efficiency**: LoRA only adds `r * (in_features + out_features)` parameters per layer.
4. **Freezing**: Always freeze the original layer parameters when using LoRA.
5. **DoRA**: Recommended for better training stability, especially with larger ranks.
6. **Merging**: Merge weights for inference to improve latency and throughput.