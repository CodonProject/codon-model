# TemporalConvNet Documentation

## Overview

Temporal Convolutional Network (TCN) for sequence modeling.

## Classes

### TemporalConvNet

Temporal Convolutional Network model.

#### Constructor

```python
class TemporalConvNet(BasicModel)
```

#### Example Usage

```python
import torch
from codon.model import TemporalConvNet

# Create TCN
model = TemporalConvNet()

x = torch.randn(2, 64, 128)  # [Batch, Channels, Seq_Len]
output = model(x)
print(f"Output shape: {output.shape}")
```

---

## Notes

1. **Sequence Modeling**: Designed for processing sequential data.
2. **Causal Convolutions**: Uses causal padding to prevent future lookahead.