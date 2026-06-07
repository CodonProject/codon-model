# Codebook Module Documentation

## Overview

The codebook module provides Lookup-Free Quantization (LFQ) for efficient image compression and representation learning.

## Data Classes

### LookupFreeQuantizationOutput

Output of the LookupFreeQuantization module.

```python
@dataclass
class LookupFreeQuantizationOutput:
    z_q: torch.Tensor          # Quantized latent tensor
    loss: torch.Tensor         # Total quantization loss
    indices: torch.Tensor      # Integer indices
    entropy: torch.Tensor      # Average bit-wise entropy
    perplexity: torch.Tensor   # Perplexity (2^entropy)
```

---

## Classes

### LookupFreeQuantization

Lookup-Free Quantization module based on MagViT-2.

#### Constructor

```python
LookupFreeQuantization(
    latent_dim: int = 256,
    codebook_dim: int = 18,
    entropy_weight: float = 0.1,
    commitment_weight: float = 0.25,
    diversity_gamma: float = 1.0
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| latent_dim | int | 256 | Dimension of input/output features |
| codebook_dim | int | 18 | Dimension of quantization space (bits) |
| entropy_weight | float | 0.1 | Weight for entropy loss |
| commitment_weight | float | 0.25 | Weight for commitment loss |
| diversity_gamma | float | 1.0 | Scaling factor for entropy penalty |

**Vocabulary Size**: `2^codebook_dim`

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| latent_dim | int | Latent dimension |
| codebook_dim | int | Codebook dimension (bits) |
| project_in | nn.Module | Projection layer (latent_dim -> codebook_dim) |
| project_out | nn.Module | Projection layer (codebook_dim -> latent_dim) |
| basis | torch.Tensor | Buffer for converting bits to indices |

#### Methods

**entropy_loss()** - Calculates bit-based entropy loss:

```python
def entropy_loss(self, affine_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

#### forward()

```python
def forward(self, z: torch.Tensor) -> LookupFreeQuantizationOutput
```

**Input shape:** `[B, C, H, W]` where C = latent_dim

**Returns:** `LookupFreeQuantizationOutput` containing:
- `z_q`: Quantized latent `[B, C, H, W]`
- `loss`: Total loss
- `indices`: Integer indices `[B, H, W]`
- `entropy`: Average entropy
- `perplexity`: Perplexity value

#### Example Usage

```python
import torch
from codon.block import LookupFreeQuantization

lfq = LookupFreeQuantization(
    latent_dim=256,
    codebook_dim=18,  # 262,144 possible codes
    entropy_weight=0.1,
    commitment_weight=0.25
)

z = torch.randn(2, 256, 16, 16)
output = lfq(z)
print(f"z_q shape: {output.z_q.shape}")       # [2, 256, 16, 16]
print(f"indices shape: {output.indices.shape}") # [2, 16, 16]
print(f"Loss: {output.loss}")
print(f"Perplexity: {output.perplexity}")
```

---

## Usage Patterns

### Autoencoder with LFQ

```python
import torch.nn as nn
from codon.block import LookupFreeQuantization

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 256, 4, stride=2, padding=1)
        )
        self.codebook = LookupFreeQuantization(
            latent_dim=256,
            codebook_dim=18
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        quantized = self.codebook(z)
        recon = self.decoder(quantized.z_q)
        return recon, quantized.loss

model = Autoencoder()
x = torch.randn(2, 3, 64, 64)
recon, loss = model(x)
print(f"Reconstruction shape: {recon.shape}")  # [2, 3, 64, 64]
print(f"Quantization loss: {loss}")
```

---

## Notes

1. **Vocabulary Size**: The vocabulary size is `2^codebook_dim`. For `codebook_dim=18`, this is 262,144 codes.
2. **Entropy Loss**: Encourages uniform usage of the codebook.
3. **Commitment Loss**: Pulls encoder outputs closer to quantized values.
4. **Differentiable**: Uses straight-through estimation for backpropagation.
5. **No Lookup Table**: Avoids storing a large codebook table, saving memory.