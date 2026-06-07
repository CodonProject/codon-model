# Vision Training Kit Documentation

## Overview

Training utilities for vision models.

## Functions

### train_autoencoder()

Train an autoencoder vision model.

```python
def train_autoencoder(
    model: AutoencoderVisionModel,
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
| model | AutoencoderVisionModel | - | Autoencoder to train |
| train_dataloader | DataLoader | - | Training data loader |
| val_dataloader | DataLoader | None | Validation data loader |
| optimizer | Optimizer | None | Optimizer |
| scheduler | LRScheduler | None | Learning rate scheduler |
| num_epochs | int | 10 | Number of training epochs |
| device | torch.device | None | Training device |
| output_dir | str | './output' | Output directory |

**Returns:** Dictionary with training metrics.

#### Example Usage

```python
from codon.kit.train import train_autoencoder
from codon.motif import MotifV1

model = MotifV1()
train_loader = ...  # Your training dataloader

results = train_autoencoder(
    model=model,
    train_dataloader=train_loader,
    num_epochs=10,
    device='cuda'
)

print(f"Final train loss: {results['train_loss'][-1]}")
print(f"Final PSNR: {results['psnr'][-1]}")
```

---

## Notes

1. **Loss Function**: Combines reconstruction loss with quantization loss.
2. **Metrics**: Tracks PSNR for image reconstruction quality.
3. **Checkpointing**: Saves model checkpoints during training.