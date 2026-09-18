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
      "owned_by": "codon",
      "capabilities": {
        "supports_image": false,
        "supports_thinking": true,
        "supports_tool": false,
        "supports_audio": false,
        "audio_subtypes": []
      }
    }
  ]
}
```

`capabilities` 就是模型的 `ModelMeta`（`codon.model.types.language.ModelMeta`），客户端可以据此
判断能不能发图片 / 音频。

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

**Multimodal request**（`content` 传成列表即可，媒体载荷在服务端解码成张量）：

```json
{
  "model": "motif-chord",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image and transcribe the audio."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0..."}},
        {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}
      ]
    }
  ]
}
```

支持的载荷写法（见 `codon.utils.media`）：

| 模态 | 写法 | 载荷 |
|------|------|------|
| image | `{'type': 'image', 'image': ...}` | data URL / 裸 base64 / 本地路径 / http(s) URL / `{'url': ...}` |
| image | `{'type': 'image_url', 'image_url': {'url': ...}}` | 同上（OpenAI 风格） |
| audio | `{'type': 'audio', 'audio': ...}` | mel 张量（进程内）/ `.npy` / `.pt` / PCM wav / base64 / data URL |
| audio | `{'type': 'input_audio', 'input_audio': {'data': ..., 'format': 'wav'}}` | 同上（OpenAI 风格），波形按 Whisper 前端转 mel |

错误响应（OpenAI 风格 `error.code`）：

| HTTP | code | 触发条件 |
|------|------|----------|
| 400 | `invalid_response_format` | `response_format` 非法 |
| 400 | `unsupported_modality` | 消息带了模型没声明的模态（图片 / 音频 / 视频）；`GET /v1/models` 的 `capabilities` 可事先查 |
| 400 | `invalid_content` | 媒体载荷无法解码（坏 base64、坏图片、音频格式不支持等） |
| 404 | `model_not_found` | `model` 不在注册表里 |

图片 / 音频的占位符数量由模型的 `image_patch_size` / `audio_pool_stride` /
`audio_num_mel_bins` 决定（MotifChord 分别是 16 / 8 / 80），服务端自动读取，无需在请求里指定。

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
4. **Multimodal**: 多模态请求由 `codon.utils.media.normalize_messages` 解码，再交给
   `codon.utils.generate.chat`；模态特征只在 prefill 注入占位符位置，decode 阶段不再重复编码
   （图片 / 音频塔每轮请求只跑一次）。