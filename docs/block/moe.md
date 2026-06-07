# MoE Module Documentation

## Overview

The MoE (Mixture of Experts) module provides implementations of Mixture-of-Experts architectures, supporting shared experts and auxiliary loss for load balancing.

## Data Classes

### MoEOutput

Output of the Mixture-of-Experts model.

```python
@dataclass
class MoEOutput:
    output: torch.Tensor
    aux_loss: Union[torch.Tensor, None]
```

### MoEInfo

Parameter counts for the MoE model.

```python
@dataclass
class MoEInfo:
    total_count: int
    active_count: int
```

---

## Classes

### Expert

A single expert module in the Mixture-of-Experts architecture. Wraps the MLP module.

#### Constructor

```python
Expert(
    in_features: int,
    hidden_features: int,
    out_features: int,
    use_gate: bool = False,
    dropout: float = 0.1
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| in_features | int | - | Number of input features |
| hidden_features | int | - | Number of hidden features |
| out_features | int | - | Number of output features |
| use_gate | bool | False | Whether to use Gated MLP |
| dropout | float | 0.1 | Dropout probability |

**Example Usage:**

```python
import torch
from codon.block import Expert

expert = Expert(
    in_features=768,
    hidden_features=3072,
    out_features=768,
    use_gate=True
)

x = torch.randn(128, 768)
output = expert(x)
print(f"Output shape: {output.shape}")  # [128, 768]
```

---

### MoE

Mixture-of-Experts module with support for shared experts and auxiliary loss.

#### Constructor

```python
MoE(
    model_dim: int,
    top_k: int,
    num_experts: int,
    num_shared_experts: int = 0,
    use_aux_loss: bool = False,
    use_gate: bool = False
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_dim | int | - | Model dimension |
| top_k | int | - | Number of experts to route to per token |
| num_experts | int | - | Total number of experts |
| num_shared_experts | int | 0 | Number of shared experts |
| use_aux_loss | bool | False | Whether to use auxiliary loss for load balancing |
| use_gate | bool | False | Whether to use Gated MLP for experts |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| experts | nn.ModuleList | List of expert modules |
| shared_experts | nn.ModuleList | List of shared expert modules |
| gate | nn.Linear | Gating network to route tokens to experts |
| use_gate | bool | Whether experts use gating mechanism |

#### Methods

**count_params()** - Count parameters with support for active parameter counting:

```python
def count_params(
    self,
    trainable_only: bool = False,
    active_only: bool = False,
    human_readable: bool = False,
    seen: set = None
) -> Union[int, str]
```

**info** - Property to get parameter count information:

```python
@property
def info(self) -> MoEInfo
```

#### forward()

```python
def forward(self, x: torch.Tensor) -> MoEOutput
```

**Input shape:** `[Batch, Seq, Dim]`

**Output:** `MoEOutput` containing:
- `output`: Output tensor with shape `[Batch, Seq, Dim]`
- `aux_loss`: Auxiliary loss (if enabled)

#### Example Usage

```python
import torch
from codon.block import MoE

moe = MoE(
    model_dim=768,
    top_k=2,
    num_experts=8,
    num_shared_experts=1,
    use_aux_loss=True,
    use_gate=True
)

x = torch.randn(2, 64, 768)
output = moe(x)
print(f"Output shape: {output.output.shape}")  # [2, 64, 768]
print(f"Aux loss: {output.aux_loss}")
```

---

## Usage Patterns

### Building a MoE Transformer

```python
import torch.nn as nn
from codon.block import TransformerMoEDecoder

class MoETransformer(nn.Module):
    def __init__(self, model_dim=768, num_layers=12, num_heads=12):
        super().__init__()
        self.decoder = nn.ModuleList([
            TransformerMoEDecoder(
                model_dim=model_dim,
                num_heads=num_heads,
                num_kv_heads=4,
                top_k=2,
                num_experts=8,
                use_aux_loss=True,
                idx=str(i)
            )
            for i in range(num_layers)
        ])
    
    def forward(self, x):
        total_aux_loss = 0
        for layer in self.decoder:
            output = layer(x)
            x = output.output
            if output.aux_loss is not None:
                total_aux_loss += output.aux_loss
        return x, total_aux_loss

model = MoETransformer(model_dim=768, num_layers=12)
x = torch.randn(2, 64, 768)
output, aux_loss = model(x)
print(f"Output shape: {output.shape}")      # [2, 64, 768]
print(f"Total aux loss: {aux_loss}")
```

### Understanding Active vs Total Parameters

```python
moe = MoE(
    model_dim=768,
    top_k=2,
    num_experts=8
)

info = moe.info
print(f"Total parameters: {info.total_count}")
print(f"Active parameters: {info.active_count}")  # Only top_k experts per token
```

### Using Shared Experts

Shared experts are always used by all tokens:

```python
moe = MoE(
    model_dim=768,
    top_k=2,
    num_experts=8,
    num_shared_experts=1  # This expert is always applied
)

x = torch.randn(2, 64, 768)
output = moe(x)
```

---

## Notes

1. **Top-K Routing**: Each token is routed to `top_k` experts based on gating scores.
2. **Auxiliary Loss**: Helps balance expert utilization and prevent some experts from being unused.
3. **Shared Experts**: Always applied to all tokens, useful for shared computations.
4. **Parameter Efficiency**: MoE models have many parameters but only activate a subset per token, making them computationally efficient.
5. **Expert Capacity**: Consider setting appropriate batch sizes to avoid overloading individual experts.