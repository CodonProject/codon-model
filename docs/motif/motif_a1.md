# MotifA1 Documentation

## Overview

MotifA1 is a causal language model built on the Transformer architecture, designed for efficient text generation with support for Grouped Query Attention (GQA) and SwiGLU activation.

## Classes

### MotifA1Tokenizer

Tokenizer for MotifA1 model, inherits from PackedTokenizer.

```python
class MotifA1Tokenizer(PackedTokenizer):
    __remote_resource__ = {
        'repo': 'CodonProject/MotifA1-SFT',
        'files': ['motif.vocab']
    }
```

---

### MotifA1

Causal language model with GQA and SwiGLU.

#### Constructor

```python
MotifA1(
    vocab_size: int = 2**13,
    model_dim: int = 768,
    num_layers: int = 16,
    num_heads: int = 8,
    num_kv_heads: int = 2,
    dropout: float = 0.1,
    tie_weights: bool = True
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| vocab_size | int | 8192 | Vocabulary size |
| model_dim | int | 768 | Model dimension |
| num_layers | int | 16 | Number of decoder layers |
| num_heads | int | 8 | Number of attention heads |
| num_kv_heads | int | 2 | Number of KV heads (GQA) |
| dropout | float | 0.1 | Dropout probability |
| tie_weights | bool | True | Whether to tie embedding and output weights |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| token_emb | nn.Embedding | Token embedding layer |
| position_emb | RotaryEmbedding | Rotary positional embedding |
| dropout | nn.Dropout | Dropout layer |
| decoder | nn.ModuleList | List of TransformerDenseDecoder layers |
| norm | nn.RMSNorm | Final normalization |
| proj_out | nn.Linear | Output projection |

#### forward()

```python
def forward(
    input_ids: torch.Tensor,
    mask: torch.Tensor = None,
    start_pos: Union[int, torch.Tensor] = 0,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    use_cache: bool = False,
    output_attentions: bool = False
) -> CausalLanguageModelOutput
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| input_ids | torch.Tensor | - | Input token IDs |
| mask | torch.Tensor | None | Attention mask |
| start_pos | Union[int, torch.Tensor] | 0 | Starting position for KV cache |
| past_key_values | List | None | Past key-value states |
| use_cache | bool | False | Whether to use KV cache |
| output_attentions | bool | False | Whether to output attention weights |

**Returns:** `CausalLanguageModelOutput` containing:
- `logits`: Prediction logits
- `past_key_values`: List of past key-value states
- `aux_loss`: Auxiliary loss
- `attentions`: List of attention weights

#### Example Usage

```python
import torch
from codon.motif import MotifA1

model = MotifA1(
    vocab_size=8192,
    model_dim=768,
    num_layers=16,
    num_heads=8,
    num_kv_heads=2
)

input_ids = torch.randint(0, 8192, (2, 64))
output = model(input_ids)
print(f"Logits shape: {output.logits.shape}")  # [2, 64, 8192]
```

---

## Generation

### Using generate()

The model inherits from `CausalLanguageModel` which provides a `generate()` method:

```python
import torch
from codon.motif import MotifA1

model = MotifA1()
model.eval()

input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Batch of 1, sequence of 5 tokens
generated = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7,
    top_k=50
)
print(f"Generated shape: {generated.shape}")  # [1, 105]
```

### Custom Sampling

```python
from codon.motif.base import Sampler

sampler = Sampler(
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1
)

generated = model.generate(
    input_ids,
    max_new_tokens=100,
    sampler=sampler,
    eos_token_id=100
)
```

---

## Loading Pretrained Weights

### Local Loading

```python
from codon.motif import MotifA1

model = MotifA1()
model.load_pretrained('path/to/MotifA1_SFT.safetensors')
model.eval()

# Now ready for inference
input_ids = torch.tensor([[1, 2, 3]])
output = model(input_ids)
```

### Remote Loading (Auto-download)

MotifA1 supports automatic downloading from ModelScope or Hugging Face using the `from_remote()` method:

```python
from codon.motif import MotifA1

# Auto-detect optimal platform (ModelScope or Hugging Face)
model = MotifA1().from_remote()
model.eval()

# Specify platform explicitly
model = MotifA1().from_remote(platform='modelscope')
# or
model = MotifA1().from_remote(platform='huggingface')

# Custom cache directory
model = MotifA1().from_remote(cache_dir='./cache')
```

**Remote Configuration:**

```python
class MotifA1(BasicModel):
    __modelscope__ = {
        'repo': 'CodonProject/MotifA1-SFT',
        'files': ['model.safetensors'],
        'branch': 'master'
    }
    
    __huggingface__ = {
        'repo': 'CodonProject/MotifA1-SFT',
        'files': ['model.safetensors'],
        'branch': 'main'
    }
```

#### Automatic Platform Selection

The `from_remote()` method automatically selects the best platform:
1. Checks for cached files first
2. If no cache, auto-detects optimal platform based on network conditions
3. Falls back to alternative platform if download fails

---

### Tokenizer Remote Loading

The tokenizer also supports remote loading:

```python
from codon.motif import MotifA1Tokenizer

# Auto-download vocabulary
tokenizer = MotifA1Tokenizer().from_remote()

# With custom cache
tokenizer = MotifA1Tokenizer().from_remote(cache_dir='./cache')
```

**Tokenizer Remote Configuration:**

```python
class MotifA1Tokenizer(PackedTokenizer):
    __remote_resource__ = {
        'repo': 'CodonProject/MotifA1-SFT',
        'files': ['motif.vocab']
    }
```

---

### Full Workflow with Remote Loading

```python
from codon.motif import MotifA1, MotifA1Tokenizer

# Load model and tokenizer from remote
model = MotifA1().from_remote()
tokenizer = MotifA1Tokenizer().from_remote()

model.eval()

# Tokenize input
text = "Hello, world!"
input_ids = tokenizer.encode(text, return_tensors='pt')

# Generate
output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7
)

# Decode
generated_text = tokenizer.decode(output[0])
print(generated_text)
```

---

## Model Architecture

```
Input Embedding -> Transformer Decoder Layers -> RMSNorm -> Output Projection

Transformer Decoder Layer:
  RMSNorm -> MultiHeadAttention -> Residual
  RMSNorm -> SwiGLU MLP -> Residual
```

### Key Features

1. **Grouped Query Attention**: `num_kv_heads=2` means 4 query heads share each KV head
2. **Rotary Embedding**: Applied to queries and keys
3. **SwiGLU**: Used in feed-forward networks
4. **Weight Tying**: Embedding weights tied to output projection

---

## Parameter Count

```python
model = MotifA1()
print(f"Total parameters: {model.count_params(human_readable=True)}")
# Output: ~100M parameters (approximate)
```

---

## Notes

1. **GQA Configuration**: `num_heads` should be divisible by `num_kv_heads`
2. **Training**: Use gradient checkpointing for large models
3. **Inference**: Enable KV caching for efficient generation
4. **FP16/FP8**: Supports mixed-precision training and inference