# Utility Functions Documentation

## Theta Validation (RoPE)

### validate_rope_config()

Validate RoPE base for given sequence length.

```python
def validate_rope_config(
    max_len: int,
    base: float
) -> ValidateRoPEConfig
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| max_len | int | Maximum sequence length |
| base | float | RoPE base value |

**Returns:** `ValidateRoPEConfig` with validation status.

#### Example Usage

```python
from codon.utils.theta import validate_rope_config

result = validate_rope_config(max_len=8192, base=10000.0)
print(f"Passed: {result.is_passed}")
print(f"Info: {result.info}")
print(f"Suggested base: {result.suggested_base}")
```

---

## Mask Utilities

### make_padding_mask()

Create padding mask.

```python
def make_padding_mask(
    src: torch.Tensor,
    pad_idx: int = 0
) -> torch.Tensor
```

**Returns:** Mask [B, 1, 1, L_src].

### make_lookahead_mask()

Create causal lookahead mask.

```python
def make_lookahead_mask(
    size: int,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor
```

**Returns:** Lower triangular mask [size, size].

### make_causal_mask()

Create combined causal mask.

```python
def make_causal_mask(
    tgt: torch.Tensor,
    pad_idx: int = 0
) -> torch.Tensor
```

**Returns:** Combined mask [B, 1, L, L].

### make_sliding_window_mask()

Create sliding window attention mask.

```python
def make_sliding_window_mask(
    tensor: torch.Tensor,
    window_size: int,
    pad_idx: int = 0,
    causal: bool = True
) -> torch.Tensor
```

#### Example Usage

```python
import torch
from codon.utils.mask import make_causal_mask, make_sliding_window_mask

seq = torch.tensor([[1, 2, 3, 4, 0]])

# Full causal mask
causal = make_causal_mask(seq)
print(f"Causal mask shape: {causal.shape}")  # [1, 1, 5, 5]

# Sliding window mask (window=2)
window = make_sliding_window_mask(seq, window_size=2)
print(f"Window mask shape: {window.shape}")  # [1, 1, 5, 5]
```

---

## TokenMask

Token masking based on special tokens.

### MaskMode

```python
class MaskMode(Enum):
    FIRST_MASK_PRE   # Mask before first occurrence
    FIRST_MASK_POST  # Mask after first occurrence
    LAST_MASK_PRE    # Mask before last occurrence
    LAST_MASK_POST   # Mask after last occurrence
    ALL_MASK_FIRST   # Alternating, first masked
    ALL_KEEP_FIRST   # Alternating, first kept
```

### TokenMask Class

```python
class TokenMask:
    def __init__(self, tokenizer: Tokenizer)
    
    def mask(
        self,
        content: str,
        special_token: Union[str, int, list],
        mode: MaskMode = MaskMode.FIRST_MASK_PRE,
        tensor_mask: bool = True
    ) -> MaskedContent
```

#### Example Usage

```python
from codon.utils.mask import TokenMask, MaskMode
from codon.motif import MotifA1Tokenizer

tokenizer = MotifA1Tokenizer().from_remote()
masker = TokenMask(tokenizer)

text = "Question: What is AI? Answer: AI is..."
result = masker.mask(
    text,
    special_token='Answer:',
    mode=MaskMode.FIRST_MASK_POST
)

print(f"Tokens: {result.tokenized}")
print(f"Mask: {result.mask}")
```

---

## Lifecycle Management

### ExitManager

Singleton for managing exit callbacks.

```python
from codon.utils.lifecycle import register_exit

@register_exit
def cleanup():
    print("Cleaning up...")

# Or with arguments
register_exit(save_checkpoint, model, path='./ckpt')
```

---

## Notes

1. **RoPE Validation**: Ensures base is sufficient for sequence length.
2. **Mask Shapes**: All masks use boolean tensors (True = attend).
3. **Exit Callbacks**: Called in reverse registration order.