# Language Training Kit Documentation

## Overview

Training utilities for language models.

## Functions

### train_language_model()

Train a causal language model.

```python
def train_language_model(
    model: CausalLanguageModel,
    train_dataloader,
    val_dataloader=None,
    optimizer=None,
    scheduler=None,
    num_epochs: int = 10,
    device=None,
    output_dir: str = './output'
) -> dict
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | CausalLanguageModel | - | Language model to train |
| train_dataloader | DataLoader | - | Training data loader |
| val_dataloader | DataLoader | None | Validation data loader |
| optimizer | Optimizer | None | Optimizer (defaults to AdamW) |
| scheduler | LRScheduler | None | Learning rate scheduler |
| num_epochs | int | 10 | Number of training epochs |
| device | torch.device | None | Training device |
| output_dir | str | './output' | Output directory |

**Returns:** Dictionary with training metrics.

#### Example Usage

```python
from codon.kit.train import train_language_model
from codon.motif import MotifA1

model = MotifA1()
train_loader = ...  # Your training dataloader

results = train_language_model(
    model=model,
    train_dataloader=train_loader,
    num_epochs=5,
    device='cuda'
)

print(f"Final train loss: {results['train_loss'][-1]}")
```

---

## Notes

1. **Loss Function**: Uses cross-entropy loss for language modeling.
2. **Logging**: Logs training progress and metrics.
3. **Checkpointing**: Saves model checkpoints during training.