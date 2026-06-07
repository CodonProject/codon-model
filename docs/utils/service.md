# Service Documentation

## Overview

OpenAI-compatible FastAPI service for hosting CausalLanguageModel instances.

## Classes

### ModelCard

Model information card for registering models.

```python
@dataclass
class ModelCard:
    model: CausalLanguageModel
    tokenizer: PackedTokenizer
    model_id: str
    owned: str
```

---

### Service

OpenAI-compatible FastAPI service wrapper.

#### Constructor

```python
Service(models: List[ModelCard])
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| models | List[ModelCard] | List of model cards to host |

#### run()

Start the FastAPI server.

```python
def run(
    host: str = '0.0.0.0',
    port: int = 11305,
    **kwargs
) -> None
```

#### Example Usage

```python
from codon.motif import MotifA1, MotifA1Tokenizer
from codon.utils.service import Service, ModelCard

# Load model and tokenizer
model = MotifA1().from_remote()
tokenizer = MotifA1Tokenizer().from_remote()
model.eval()

# Create model card
card = ModelCard(
    model=model,
    tokenizer=tokenizer,
    model_id='motif-a1-sft',
    owned='codon'
)

# Start service
service = Service([card])
service.run(host='0.0.0.0', port=8080)
```

---

## API Endpoints

### GET /v1/models

List available models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "motif-a1-sft",
      "object": "model",
      "created": 1234567890,
      "owned_by": "codon"
    }
  ]
}
```

### POST /v1/chat/completions

Chat completion endpoint (OpenAI-compatible).

**Request:**
```json
{
  "model": "motif-a1-sft",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": false
}
```

**Response (non-streaming):**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "model": "motif-a1-sft",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you?"
      },
      "finish_reason": "stop"
    }
  ]
}
```

---

## Streaming

Set `stream: true` for Server-Sent Events:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="dummy"
)

stream = client.chat.completions.create(
    model="motif-a1-sft",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end='')
```

---

## Notes

1. **CORS**: CORS middleware is enabled for all origins.
2. **Concurrency**: Uses asyncio locks per model for thread safety.
3. **Reasoning**: Supports `reasoning_content` field for Chain of Thought.