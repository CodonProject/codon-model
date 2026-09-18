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
    messages: List[Dict[str, Any]],
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    enable_thinking: bool = True,
    response_format: Optional[Dict[str, Any]] = None,
    patch_size: Optional[int] = None,
    audio_pool_stride: Optional[int] = None,
    num_mel_bins: Optional[int] = None,
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
| patch_size | int | None | 图像占位符的 patch 尺寸，默认取模型 `image_patch_size`（纯文本模型 12） |
| audio_pool_stride | int | None | 音频占位符的池化步长，默认取模型 `audio_pool_stride` |
| num_mel_bins | int | None | 波形转 mel 的维数，默认取模型 `audio_num_mel_bins`（80） |

**Yields:** `ChatChunk` objects with generated content.

**Message format:**
```python
{
    'role': 'user' | 'assistant' | 'system' | 'environment',
    'content': str                      # 纯文本
}

{
    'role': 'user',
    'content': [                        # 多模态（codon.j2 占位符）
        {'type': 'text',  'text': 'describe this'},
        {'type': 'image', 'image': <张量 | 路径 | base64 | data URL | PIL | ndarray>},
        {'type': 'audio', 'audio': <mel 张量 | .npy/.pt | PCM wav | base64>},
        {'type': 'video', 'video': 'demo.mp4',
         'frames': [{'image': <同上>, 'timestamp': '0.00'}, ...]},   # 逐帧视频复用图像塔
        {'type': 'action', 'action': <[T, action_dim] 张量>},         # 连续值占位符
        {'type': 'proprio', 'proprio': <[T, proprio_dim] 张量>},
        {'type': 'tactile', 'tactile': <[T, tactile_dim] 张量>},
        {'type': 'bbox', 'bbox': '[0.1,0.2,0.3,0.4]'},               # grounding：字符串载荷
    ]
}
```

`role='environment'` 渲染成 `<|im_start|><|environment|>…<|im_end|>`（逐轮环境反馈，不进 loss）。
grounding 块（`bbox` / `point` / `ref` / `grasp` / `force` / `camera` / `trajectory` / `waypoint`）
的载荷是调用方格式化好的字符串，模板只负责包上结构 token。

OpenAI 风格的 `{'type': 'image_url', 'image_url': {'url': ...}}` 与
`{'type': 'input_audio', 'input_audio': {'data': ..., 'format': 'wav'}}` 同样接受。
媒体载荷由 `codon.utils.media` 解码成模型要的张量；模型没有声明对应模态能力时抛
`codon.utils.media.UnsupportedModalityError`。

多模态只在 **prefill** 阶段注入（`x[b, *_patch_indices] = 模态特征`，序列长度不变），
decode 步只喂新 token，因此每轮请求图片 / 音频塔只跑一次。

### 多模态与 KV cache

`chat()` 用 `ModelCache` 做增量解码，多模态与它是天然兼容的：

- 占位符是**就地写入**、序列长度不变 ⇒ `ModelCache.seq_length` 就是真实位置，
  decode 的 `start_pos` 直接取 cache 长度，不需要任何「模态长度补偿」；
- 模态特征算出的 key/value 在 prefill 就进了 cache，decode 直接复用（不会重跑视觉 / 音频塔）；
- 混合主干（GDN + GQA）里 GDN 的 cache 按 steps 计数、GQA 的 cache 按 KV 长度计，
  两者的 `seq_length` 在 prefill / decode 后一致（`MotifChord` 有对应测试）；
- 服务端每个请求各建一个 `ModelCache`，并用 per-model `asyncio.Lock` 串行化，
  不存在请求间共享 cache 的问题。

注意：`chat()` 只构造单样本 batch（`batch_dim=True`、attention mask 全 1），所以
decode 步不传 mask 是安全的。若要自己拼带 padding 的多样本 batch，由于 cache 里不保存
mask 状态，decode 仍会 attend 到 padding 的 key —— 这种用法请用 pack 过的序列或 `B=1`。

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