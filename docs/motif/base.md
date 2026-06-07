# Motif Base Classes Documentation

## Overview

Base classes for building causal language models and autoencoder vision models.

## Data Classes

### AutoVisionEncoderOutput

Output of autoencoder vision model encoder.

```python
@dataclass
class AutoVisionEncoderOutput:
    z_q: torch.Tensor
    loss: torch.Tensor = None
    indices: torch.Tensor = None
    grid_shape: tuple = None
    entropy: torch.Tensor = None
    perplexity: torch.Tensor = None
    hidden_states: torch.Tensor = None
```

### AutoVisionDecoderOutput

Output of autoencoder vision model decoder.

```python
@dataclass
class AutoVisionDecoderOutput:
    reconstructed: torch.Tensor
    grid_shape: tuple = None
    hidden_states: torch.Tensor = None
```

### CausalLanguageModelOutput

Output of causal language model.

```python
@dataclass
class CausalLanguageModelOutput:
    logits: torch.Tensor
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    aux_loss: Optional[torch.Tensor] = None
    attentions: Optional[List[torch.Tensor]] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
```

---

## Classes

### KVCache

Key-Value Cache container for autoregressive generation.

#### Methods

| Method | Description |
|--------|-------------|
| `update()` | Update cache with new KV states |
| `current_len` | Property returning current sequence length |
| `clear()` | Flush the cache |

#### Example Usage

```python
from codon.motif.base import KVCache

kv_cache = KVCache()
kv_cache.update([(k1, v1), (k2, v2)])  # List of (key, value) tuples
print(f"Current length: {kv_cache.current_len}")
kv_cache.clear()
```

---

### Sampler

Sampler for autoregressive generation.

#### Constructor

```python
Sampler(
    temperature: float = 0.7,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.15
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| temperature | float | 0.7 | Sampling temperature |
| top_k | int | None | Top-k sampling |
| top_p | float | None | Nucleus sampling |
| repetition_penalty | float | 1.15 | Repetition penalty |

#### __call__()

```python
def __call__(self, logits: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor
```

**Input shape:** `logits` - `[batch_size, vocab_size]`

**Returns:** `torch.Tensor` - Next token IDs `[batch_size, 1]`

#### Example Usage

```python
import torch
from codon.motif.base import Sampler

sampler = Sampler(
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1
)

logits = torch.randn(2, 8192)
input_ids = torch.randint(0, 8192, (2, 64))
next_token = sampler(logits, input_ids)
print(f"Next token shape: {next_token.shape}")  # [2, 1]
```

---

### CausalLanguageModel

Base class for causal language models.

#### generate()

```python
def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    sampler: Optional[Sampler] = None,
    eos_token_id: Optional[int] = None,
    use_cache: bool = True
) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| input_ids | torch.Tensor | - | Input prompt token IDs |
| max_new_tokens | int | 100 | Maximum new tokens to generate |
| sampler | Sampler | None | Sampler instance |
| eos_token_id | int | None | End-of-sequence token ID |
| use_cache | bool | True | Whether to use KV caching |

**Returns:** `torch.Tensor` - Generated token IDs

#### compute_perplexity()

```python
def compute_perplexity(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| logits | torch.Tensor | Model output logits `[batch, seq_len, vocab_size]` |
| targets | torch.Tensor | Target token IDs `[batch, seq_len]` |

**Returns:** `torch.Tensor` - Perplexity value

#### Example Usage

```python
import torch
from codon.motif.base import CausalLanguageModel, Sampler

class MyModel(CausalLanguageModel):
    def __init__(self):
        super().__init__()
        # ... model layers ...
    
    def forward(self, input_ids, **kwargs):
        # ... forward logic ...
        pass

model = MyModel()
model.eval()

# Generation
input_ids = torch.tensor([[1, 2, 3]])
generated = model.generate(
    input_ids,
    max_new_tokens=50,
    sampler=Sampler(temperature=0.7),
    eos_token_id=100
)

# Perplexity calculation
logits = torch.randn(2, 64, 8192)
targets = torch.randint(0, 8192, (2, 64))
perplexity = model.compute_perplexity(logits, targets)
print(f"Perplexity: {perplexity.item():.2f}")
```

---

### AutoencoderVisionModel

Base class for autoencoder vision models.

#### Methods

| Method | Description |
|--------|-------------|
| `encode()` | Encode image to latent |
| `decode()` | Decode latent to image |
| `compute_psnr()` | Compute PSNR between two images |

#### Example Usage

```python
import torch
from codon.motif.base import AutoencoderVisionModel

class MyAutoencoder(AutoencoderVisionModel):
    def __init__(self):
        super().__init__()
        self.codebook_size = 2**15
    
    def _encode(self, x):
        # ... encoding logic ...
        pass
    
    def _decode(self, encoder_output):
        # ... decoding logic ...
        pass

model = MyAutoencoder()

# Encode
image = torch.randn(1, 3, 64, 64)
encoder_output = model.encode(image)

# Decode
decoder_output = model.decode(encoder_output)

# PSNR
reconstructed = decoder_output.reconstructed
psnr = model.compute_psnr(image, reconstructed)
print(f"PSNR: {psnr.item():.2f} dB")
```

---

### VisionEmbedding

Vision embedding module with dead code handling.

#### Constructor

```python
VisionEmbedding(
    hidden_dim: int,
    dead_codes: Union[List[int], str],
    codebook_dim: int = 15,
    vision_model: Optional[AutoencoderVisionModel] = None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| hidden_dim | int | - | Embedding dimension |
| dead_codes | List[int] or str | - | List of dead codes or path to JSON file |
| codebook_dim | int | 15 | Codebook dimension |
| vision_model | AutoencoderVisionModel | None | Vision model for embedding images |

#### forward()

```python
def forward(self, original_indices: torch.Tensor) -> torch.Tensor
```

#### embed_image()

```python
def embed_image(self, image: torch.Tensor) -> torch.Tensor
```

#### Example Usage

```python
import torch
from codon.motif.base import VisionEmbedding, AutoencoderVisionModel

vision_model = AutoencoderVisionModel()  # Your vision model
embedding = VisionEmbedding(
    hidden_dim=768,
    dead_codes=[0, 1, 2],  # Dead code indices
    codebook_dim=15,
    vision_model=vision_model
)

# Embed indices
indices = torch.randint(0, 2**15, (2, 64))
output = embedding(indices)
print(f"Output shape: {output.shape}")  # [2, 64, 768]

# Embed image
image = torch.randn(2, 3, 64, 64)
image_embedding = embedding.embed_image(image)
print(f"Image embedding shape: {image_embedding.shape}")  # [2, num_patches, 768]
```