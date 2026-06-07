# Session Documentation

## Overview

Chat session management with token masking for training.

## Classes

### Message

A single message in a chat session.

```python
@dataclass
class Message:
    ids: list[int]
    ignore_mask: list[bool]
    role: Optional[str] = None
    images: list[torch.Tensor] = field(default_factory=list)
```

#### Methods

- `mask_all()` - Mask all tokens
- `unmask_all()` - Unmask all tokens
- `mask_before(index)` - Mask tokens before index
- `mask_after(index)` - Mask tokens after index
- `find(token_id)` - Find first occurrence of token

---

### Session

Chat session with masking policies.

#### Constructor

```python
Session(
    tokenizer: PackedTokenizer,
    patch_size: int = 12
)
```

#### Methods

**add_message()** - Add a message to the session:

```python
def add_message(
    message: dict,
    mask: Optional[MaskPolicy] = None
) -> Session
```

**add_generation_prompt()** - Add generation prompt:

```python
def add_generation_prompt(
    enable_thinking: bool = False,
    disable_thinking: bool = False
) -> Session
```

**to_tensors()** - Convert to tensors:

```python
def to_tensors(
    device: Union[str, torch.device] = 'cpu',
    pad_to: Optional[int] = None,
    batch_dim: bool = False
) -> dict[str, torch.Tensor]
```

#### Example Usage

```python
import torch
from codon.utils.session import Session
from codon.motif import MotifA1Tokenizer

tokenizer = MotifA1Tokenizer().from_remote()
session = Session(tokenizer)

# Add messages
session.add_message({'role': 'system', 'content': 'You are helpful.'})
session.add_message({'role': 'user', 'content': 'Hello!'})
session.add_message({'role': 'assistant', 'content': 'Hi there!'})

# Add generation prompt
session.add_generation_prompt()

# Get tensors
tensors = session.to_tensors(device='cuda', batch_dim=True)
print(f"Input IDs: {tensors['input_ids'].shape}")
print(f"Labels: {tensors['labels'].shape}")
```

---

## Mask Policies

| Policy | Description |
|--------|-------------|
| 'all' | Mask all tokens (no loss) |
| 'none' | Unmask all tokens |
| 'content' | Unmask only content (after CoT) |
| 'thought' | Unmask only Chain of Thought |
| 'answer' | Unmask only final answer |
| 'fim' | Unmask fill-in-middle region |

### Setting Policies

```python
session.set_policy('model', 'content')  # Train on content only
session.set_policy('model', 'thought')  # Train on reasoning only
```

---

## Multimodal Support

```python
session.add_message({
    'role': 'user',
    'content': [
        {'type': 'text', 'text': 'Describe this image:'},
        {'type': 'image', 'image': image_tensor}
    ]
})
```

---

## Notes

1. **Special Tokens**: Uses `[im_start]`, `[im_end]`, `[cot_start]`, `[cot_end]`, etc.
2. **Image Patches**: Automatically expands image placeholders to patch tokens.
3. **Labels**: Masked positions have label -100 (ignored in loss).