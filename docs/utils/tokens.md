# Tokenization Utilities Documentation

## Overview

Utilities for tokenization, including tokenizer training and a packed tokenizer class.

## Constants

### Special Tokens

```python
core_tokens = ['[pad]', '[unk]', '[sep]', '[cls]']
chat_tokens = [
    '[im_start]', '[im_end]',
    '[system]', '[user]', '[model]', '[tool]',
    '[interruption]', '[fim]',
]
reasoning_tokens = ['[cot_start]', '[cot_end]']
code_tokens = ['[fim_pre]', '[fim_mid]', '[fim_suf]']
tool_tokens = ['[tool_start]', '[tool_name]', '[tool_args]', '[tool_end]']
multimodal_tokens = [
    '[image_start]', '[image_pad]', '[image_end]',
    '[audio_start]', '[audio_pad]', '[audio_end]',
    '[video_start]', '[video_pad]', '[video_end]'
]
```

### Default Chat Template

Jinja2 template for formatting chat conversations:

```jinja
{% for message in messages %}
    {{ '[im_start]' }}
    {% if message['role'] == 'user' %}
        {{ '[user]' }}{{ message['content'] }}{{ '[im_end]' }}
    {% elif message['role'] == 'assistant' %}
        {{ '[model]' }}{{ '[cot_start][cot_end]' }}{{ message['content'] }}{{ '[im_end]' }}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
    {{ '[im_start][model][cot_start]' }}
{% endif %}
```

---

## Data Classes

### TokenizerTrainerResult

Result of tokenizer trainer creation.

```python
@dataclass
class TokenizerTrainerResult:
    tokenizer: Tokenizer
    trainer: BpeTrainer
    
    def train_from_iterator(self, iter: Generator) -> 'TokenizerTrainerResult': ...
    
    @property
    def packed_tokenizer(self) -> 'PackedTokenizer': ...
```

---

## Functions

### create_tokenizer_trainer()

Creates a BPE Tokenizer trainer.

```python
def create_tokenizer_trainer(
    unk_token: str = '[unk]',
    vocab_size: int = 32000,
    special_tokens: list[str] = base_special_tokens
) -> TokenizerTrainerResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| unk_token | str | '[unk]' | Unknown token identifier |
| vocab_size | int | 32000 | Target vocabulary size |
| special_tokens | list[str] | base_special_tokens | List of special tokens |

**Example:**

```python
from codon.utils.tokens import create_tokenizer_trainer

trainer_result = create_tokenizer_trainer(
    vocab_size=8192,
    special_tokens=['[pad]', '[unk]', '[sep]']
)

# Train from iterator
def data_iterator():
    for _ in range(1000):
        yield "sample text for training"

trainer_result.train_from_iterator(data_iterator())

# Get packed tokenizer
tokenizer = trainer_result.packed_tokenizer
```

---

## Classes

### PackedTokenizer

Tokenizer wrapper with chat template support.

#### Constructor

```python
PackedTokenizer(tokenizer: Optional[Union[Tokenizer, str]] = None)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tokenizer | Tokenizer or str | None | Tokenizer object or path to saved tokenizer |

#### Methods

| Method | Description |
|--------|-------------|
| `set_chat_template()` | Set custom chat template |
| `reset_chat_template()` | Reset to default template |
| `token_to_id()` | Convert token to ID |
| `apply_chat_template()` | Apply chat template to messages |
| `encode()` | Encode text to token IDs |
| `decode()` | Decode token IDs to text |
| `save()` | Save tokenizer to file |
| `load()` | Load tokenizer from file |

#### Example Usage

```python
from codon.utils.tokens import PackedTokenizer

# Load tokenizer
tokenizer = PackedTokenizer('path/to/tokenizer.zip')

# Encode text
text = "Hello, world!"
ids = tokenizer.encode(text)
print(f"Token IDs: {ids}")

# Decode
decoded = tokenizer.decode(ids)
print(f"Decoded: {decoded}")

# Apply chat template
messages = [
    {'role': 'user', 'content': 'Hello!'},
    {'role': 'assistant', 'content': 'Hi there!'}
]
input_ids = tokenizer.apply_chat_template(messages)
print(f"Chat template IDs: {input_ids}")
```

#### apply_chat_template()

```python
def apply_chat_template(
    self,
    messages: List[Dict[str, Any]],
    add_generation_prompt: bool = True,
    **kwargs
) -> List[int]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| messages | List[Dict] | - | List of message dictionaries |
| add_generation_prompt | bool | True | Whether to add generation prompt |

**Message format:**
```python
{
    'role': 'user' | 'assistant' | 'system' | 'tool',
    'content': str | List[Dict],  # Text or list of content items
    'thought': str,  # Optional Chain of Thought
    'tool_calls': List,  # Optional tool calls
    'tools': List,  # Optional tools
}
```

#### Example Chat Template Usage

```python
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': [
        {'type': 'image'},
        {'type': 'text', 'text': 'Describe this image.'}
    ]},
    {'role': 'assistant', 'content': 'This is a cat.', 'thought': 'The image shows a cat.'}
]

input_ids = tokenizer.apply_chat_template(messages)
```

---

## Usage Patterns

### Training a Custom Tokenizer

```python
from codon.utils.tokens import create_tokenizer_trainer

# Create trainer
trainer_result = create_tokenizer_trainer(vocab_size=16384)

# Training data
def get_training_data():
    for filename in training_files:
        with open(filename, 'r', encoding='utf-8') as f:
            yield f.read()

# Train
trainer_result.train_from_iterator(get_training_data())

# Save
tokenizer = trainer_result.packed_tokenizer
tokenizer.save('my_tokenizer.zip')
```

### Using Chain of Thought

```python
messages = [
    {'role': 'user', 'content': 'Solve: 2 + 2 * 3'},
    {'role': 'assistant', 'content': '8', 'thought': '2 + 2 * 3 = 2 + 6 = 8'}
]

input_ids = tokenizer.apply_chat_template(messages)
# Output includes [cot_start] + thought + [cot_end]
```

---

## Notes

1. **Token Format**: Tokenizer uses BPE (Byte-Pair Encoding).
2. **Normalization**: Uses NFKC normalization by default.
3. **Special Tokens**: Includes comprehensive set of special tokens for chat, reasoning, and multimodal use cases.
4. **Safety**: Automatically escapes `]` characters to prevent template injection.