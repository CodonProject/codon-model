# Generation Utilities Documentation

## Overview

Utilities for streaming text generation with Chain of Thought support.

## Data Classes

### ChatChunk

A data chunk returned during streaming generation.

```python
@dataclass
class ChatChunk:
    content: str        # Decoded text fragment
    is_cot: bool        # Whether fragment is Chain of Thought
    cot_ended: bool     # Whether CoT just ended
```

---

## Functions

### chat()

Generates chat responses in streaming fashion.

```python
def chat(
    model: CausalLanguageModel,
    tokenizer: PackedTokenizer,
    device: torch.device,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> Generator[ChatChunk, None, None]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | CausalLanguageModel | - | Language model for generation |
| tokenizer | PackedTokenizer | - | Tokenizer for encoding/decoding |
| device | torch.device | - | Device for computation |
| messages | List[Dict] | - | List of dialogue messages |
| max_new_tokens | int | 1024 | Maximum tokens to generate |
| temperature | float | 0.3 | Sampling temperature |
| top_k | int | None | Top-k sampling |
| top_p | float | None | Nucleus sampling |

**Yields:** `ChatChunk` objects with generated content.

**Message format:**
```python
{
    'role': 'user' | 'assistant' | 'system',
    'content': str
}
```

#### Example Usage

```python
import torch
from codon.utils.generate import chat, ChatChunk
from codon.motif import MotifA1, MotifA1Tokenizer

# Load model and tokenizer
model = MotifA1().to('cuda')
tokenizer = MotifA1Tokenizer()

# Messages
messages = [
    {'role': 'user', 'content': 'Explain quantum computing in simple terms.'}
]

# Streaming generation
for chunk in chat(
    model=model,
    tokenizer=tokenizer,
    device=torch.device('cuda'),
    messages=messages,
    max_new_tokens=512,
    temperature=0.3
):
    # Handle Chain of Thought
    if chunk.is_cot:
        print(f"Thinking: {chunk.content}", end='', flush=True)
    else:
        print(chunk.content, end='', flush=True)
    
    # CoT ended, maybe add formatting
    if chunk.cot_ended:
        print("\n---")
```

---

## Usage Patterns

### Web UI Integration

```python
async def generate_stream(request):
    messages = await request.json()
    
    async for chunk in chat(
        model=model,
        tokenizer=tokenizer,
        device=device,
        messages=messages
    ):
        # Send chunk as Server-Sent Event
        yield {
            'content': chunk.content,
            'is_cot': chunk.is_cot,
            'cot_ended': chunk.cot_ended
        }
```

### Chain of Thought Handling

```python
thinking_buffer = []
final_response = []

for chunk in chat(model, tokenizer, device, messages):
    if chunk.is_cot:
        thinking_buffer.append(chunk.content)
    else:
        final_response.append(chunk.content)
    
    if chunk.cot_ended:
        # Process thinking
        print(f"Reasoning: {''.join(thinking_buffer)}")
        thinking_buffer = []

print(f"Final Answer: {''.join(final_response)}")
```

---

## Notes

1. **Streaming**: The function is a generator that yields chunks incrementally.
2. **Chain of Thought**: Detects `[cot_start]` and `[cot_end]` tokens to track thinking process.
3. **KV Caching**: Uses efficient KV caching for decoding.
4. **Special Tokens**: Automatically handles special tokens like `[im_end]` and `[pad]`.