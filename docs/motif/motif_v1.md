# MotifV1 Documentation

## Overview

MotifV1 is a vision autoencoder model with Lookup-Free Quantization (LFQ) for efficient image compression and generation.

## Classes

### MotifV1Encoder

Encoder component of MotifV1.

#### Constructor

```python
MotifV1Encoder(
    in_features: int = 3,
    patch_size: int = 12,
    latent_dim: int = 256,
    num_heads: int = 4,
    num_kv_heads: int = 4,
    codebook_dim: int = 18,
    entropy_weight: float = 0.1,
    commitment_weight: float = 0.25,
    diversity_gamma: float = 1.0,
    rope_emb: InterleavedRotaryEmbedding = None,
    use_attention: bool = True,
    depth_level: int = 1
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| in_features | int | 3 | Number of input channels |
| patch_size | int | 12 | Spatial size of patches |
| latent_dim | int | 256 | Latent dimension |
| num_heads | int | 4 | Number of attention heads |
| num_kv_heads | int | 4 | Number of KV heads |
| codebook_dim | int | 18 | Codebook dimension (bits) |
| entropy_weight | float | 0.1 | Weight for entropy loss |
| commitment_weight | float | 0.25 | Weight for commitment loss |
| diversity_gamma | float | 1.0 | Scaling factor for entropy penalty |
| rope_emb | InterleavedRotaryEmbedding | None | 2D rotary positional embedding |
| use_attention | bool | True | Whether to use attention |
| depth_level | int | 1 | Network depth multiplier |

#### forward()

```python
def forward(
    splited_image: torch.Tensor,
    grid_shape: tuple,
    rope_emb: InterleavedRotaryEmbedding = None
) -> AutoVisionEncoderOutput
```

**Input shape:** `[num_patches, channels, patch_size, patch_size]`

**Returns:** `AutoVisionEncoderOutput` containing:
- `z_q`: Quantized latent
- `loss`: Quantization loss
- `indices`: Quantized indices
- `entropy`: Average entropy
- `perplexity`: Perplexity
- `hidden_states`: Hidden states
- `grid_shape`: Grid shape

---

### MotifV1Decoder

Decoder component of MotifV1.

#### Constructor

```python
MotifV1Decoder(
    latent_dim: int = 256,
    out_features: int = 3,
    patch_size: int = 12,
    num_heads: int = 4,
    num_kv_heads: int = 4,
    base_channels: int = 64,
    initial_size: int = None,
    rope_emb: InterleavedRotaryEmbedding = None,
    norm: str = 'batch',
    activation: str = 'relu',
    use_attention: bool = True,
    depth_level: int = 1
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| latent_dim | int | 256 | Latent dimension |
| out_features | int | 3 | Number of output channels |
| patch_size | int | 12 | Output patch size |
| num_heads | int | 4 | Number of attention heads |
| num_kv_heads | int | 4 | Number of KV heads |
| base_channels | int | 64 | Base channel width |
| initial_size | int | None | Initial feature map size |
| rope_emb | InterleavedRotaryEmbedding | None | 2D rotary positional embedding |
| norm | str | 'batch' | Normalization type |
| activation | str | 'relu' | Activation function |
| use_attention | bool | True | Whether to use attention |
| depth_level | int | 1 | Network depth multiplier |

#### forward()

```python
def forward(
    z_q: torch.Tensor,
    grid_shape: tuple,
    rope_emb: InterleavedRotaryEmbedding = None
) -> AutoVisionDecoderOutput
```

**Input shape:** `[num_patches, latent_dim]`

**Returns:** `AutoVisionDecoderOutput` containing:
- `reconstructed`: Reconstructed patches
- `grid_shape`: Grid shape
- `hidden_states`: Hidden states

---

### MotifV1

Complete autoencoder model combining encoder and decoder.

#### Constructor

```python
MotifV1(
    in_features: int = 3,
    out_features: int = 3,
    patch_size: int = 12,
    latent_dim: int = 256,
    num_heads: int = 4,
    num_kv_heads: int = 4,
    codebook_dim: int = 15,
    entropy_weight: float = 0.1,
    commitment_weight: float = 0.25,
    diversity_gamma: float = 1.0,
    base_channels: int = 128,
    initial_size: int = None,
    rope_emb: InterleavedRotaryEmbedding = None,
    norm: str = 'batch',
    activation: str = 'silu',
    encoder_use_attention: bool = True,
    decoder_use_attention: bool = True,
    encoder_depth_level: int = 6,
    decoder_depth_level: int = 6
)
```

#### Methods

**encode()** - Encode an image to latent representation:

```python
def encode(self, x: torch.Tensor) -> AutoVisionEncoderOutput
```

**decode()** - Decode a latent representation to an image:

```python
def decode(self, encoder_output: AutoVisionEncoderOutput) -> AutoVisionDecoderOutput
```

**forward()** - Forward pass:

```python
def forward(self, splited_image: torch.Tensor, grid_shape: tuple) -> AutoVisionEncoderOutput
```

#### Example Usage

```python
import torch
from codon.motif import MotifV1

model = MotifV1(
    in_features=3,
    out_features=3,
    patch_size=12,
    latent_dim=256,
    codebook_dim=15
)

# Encode an image
image = torch.randn(1, 3, 96, 96)  # [Batch, Channels, Height, Width]
encoder_output = model.encode(image)
print(f"Encoded indices shape: {encoder_output.indices.shape}")
print(f"Quantization loss: {encoder_output.loss}")

# Decode
decoder_output = model.decode(encoder_output)
print(f"Reconstructed shape: {decoder_output.reconstructed.shape}")
```

---

## Architecture

### Encoder
```
Image Patches -> ResNet -> Attention (optional) -> 2D RoPE -> LFQ -> Latent
```

### Decoder
```
Latent -> Attention (optional) -> 2D RoPE -> Linear -> PixelShuffle Upsample -> Image Patches
```

### Key Features

1. **Patch-Based Processing**: Images are split into patches for efficient processing
2. **2D Rotary Embedding**: Applied to capture spatial relationships
3. **Lookup-Free Quantization**: No codebook storage required
4. **ResNet Backbone**: Used for feature extraction

---

## Usage Patterns

### Full Image Reconstruction

```python
import torch
from codon.motif import MotifV1

model = MotifV1()
model.eval()

# Load an image (assuming 96x96 input)
image = torch.randn(1, 3, 96, 96)

# Encode
encoder_output = model.encode(image)
print(f"Codebook indices: {encoder_output.indices}")

# Decode
decoder_output = model.decode(encoder_output)
reconstructed = decoder_output.reconstructed

# Reconstruct full image from patches
full_recon = model._reconstruct_image(
    decoder_output.reconstructed,
    decoder_output.grid_shape
)
print(f"Full reconstruction shape: {full_recon.shape}")  # [1, 3, 96, 96]
```

### Calculate PSNR

```python
psnr = model.compute_psnr(image, full_recon)
print(f"PSNR: {psnr.item():.2f} dB")
```

---

## Notes

1. **Image Size**: Input images must have dimensions divisible by `patch_size`
2. **Codebook Size**: Vocabulary size is `2^codebook_dim`
3. **Attention**: Can be disabled for faster inference
4. **Depth Levels**: Higher depth levels increase model capacity