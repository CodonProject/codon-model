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
    audios: list[torch.Tensor] = field(default_factory=list)   # mel，[num_mel_bins, frames]
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
    patch_size: int = 12,
    audio_pool_stride: int = 8,
    image_capab: Optional[bool] = None,
    audio_capab: Optional[bool] = None,
    video_capab: bool = False
)
```

`patch_size` must match the vision tower (DINOv3 ViT-B/16 → `16`, MotifV1 → `12`) and
`audio_pool_stride` the audio tower's pooling (`WhisperTinyAudioEncoder(pool_stride=8)` → `8`);
they decide how many `<|modality_*_pad|>` placeholders one image / one mel expands into.

`image_capab` / `audio_capab` default to `None` = "auto": a message that carries that modality is
rendered as placeholders, others are unaffected. `codon.j2` defaults both flags to `false`, which
renders `Your image modality is unsupported` text instead of placeholders — so pass `True`/`False`
explicitly if you want to force the capability. `video_capab` defaults to `False` (no video tower);
a `video` item that carries `frames` is rendered as per-frame image placeholders instead, so it only
needs `image_capab` (Session turns that on automatically for frames).

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
    disable_thinking: bool = False,
    effort: Optional[str] = None
) -> Session
```

When `enable_thinking=True`, the thinking-effort token is emitted right after `<|thought_start|>`.
`effort` accepts `minimal` / `low` / `medium` / `high` / `xhigh` / `max` / `ultra`, which collapse onto
three tokens:

| effort | token |
|--------|-------|
| `minimal`, `low` | `<|effort_low|>` |
| `medium`, `high`, `xhigh` | `<|effort_high|>` |
| `max`, `ultra` | `<|effort_max|>` |

`effort=None` uses the default level (`codon.res.EFFORT_DEFAULT`, i.e. `'high'`). An unknown level, or an
explicit level whose token is missing from the vocab, raises `ValueError`; the default level is silently
omitted for older vocabularies.

**to_tensors()** - Convert to tensors:

```python
def to_tensors(
    device: Union[str, torch.device] = 'cpu',
    pad_to: Optional[int] = None,
    batch_dim: bool = False
) -> dict[str, torch.Tensor]
```

Returns `input_ids` / `labels` / `attention_mask` / `image_patch_indices` /
`audio_patch_indices` / `action_patch_indices` / `proprio_patch_indices` /
`tactile_patch_indices` plus the modality payloads `images`, `audios`, `actions`, `proprios` and
`tactiles` (lists of tensors, in message/content order). The `*_patch_indices` are the **absolute
positions of the placeholder tokens** in the sequence — feed them straight to the model, which does
`x[b, indices[b]] = modality_features` (sequence length unchanged):

```python
model(
    input_ids=tensors['input_ids'],
    images=tensors['images'],
    image_patch_indices=tensors['image_patch_indices'],
    audios=tensors['audios'],
    audio_patch_indices=tensors['audio_patch_indices'],
    actions=tensors['actions'],
    action_patch_indices=tensors['action_patch_indices'],
    mask=tensors['attention_mask'],
)
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
        {'type': 'image', 'image': image_tensor}          # [C, H, W]
    ]
})

session.add_message({
    'role': 'user',
    'content': [
        {'type': 'audio', 'audio': mel_tensor},           # [num_mel_bins, frames]
        {'type': 'text', 'text': 'Transcribe this audio.'}
    ]
})
```

`<|modality_image_start|>` / `<|modality_image_end|>` (audio 同构) 之间的占位符数量在消息编码后按
张量尺寸自动展开：图片 `(H // patch_size) * (W // patch_size)`，音频
`((frames - 1) // 2 + 1) // audio_pool_stride`（与 `WhisperTinyAudioEncoder` 的下采样链一致）。
两种模板形态都支持：`<start><end>`（中间插入）与 codon.j2 的 `<start><pad><end>`（把单个 pad 展开）。
展开出来的占位符 `ignore_mask=True`，不参与语言模型 loss。

### Embodied Content Items

`codon/res/codon.j2` 还认识一组具身 / 机器人 token。它们分两类：

**连续值占位符**（张量就地写入，形态与图片一致）：

```python
session.add_message({'role': 'model', 'content': [
    {'type': 'text',   'text': 'pick up the cup'},
    {'type': 'action', 'action': torch.randn(8, 7)},      # [T, action_dim] -> T 个占位符
    {'type': 'proprio', 'proprio': torch.randn(7)},       # [proprio_dim] -> 1 个占位符
    {'type': 'tactile', 'tactile': torch.randn(4, 32)},   # [T, tactile_dim]
]})
```

- `action` → `<|action_start|><|action_pad|>…<|action_end|>`，pad 数量 = 张量行数。
- `action_chunk` 是「多个 step」的简写：`'actions'` 为 `list` / `tuple` 时每个元素一个三连、
  用 `<|action_sep|>` 连接，整体包在 `<|action_chunk_start|>` / `<|action_chunk_end|>` 之间；
  再给 `'flag': 'terminate' | 'continue'` 会在末尾追加 `<|action_terminate|>` / `<|action_continue|>`。
  张量（array-like）整体算**一个** block，不会被拆成多个 step。
- `proprio` / `tactile` 分别展开 `<|proprio_pad|>` / `<|tactile_pad|>`。

**透传块**（模板只负责包结构 token，载荷是调用方给好的字符串；结构 token 无法用 `text` 伪造，
因为 `safe_rules` 会把文本里的 `[` `]` `|` 换成 safe escape）：

```python
session.add_message({'role': 'user', 'content': [
    {'type': 'text',  'text': 'grab '},
    {'type': 'ref',   'ref': 'the red cup'},
    {'type': 'bbox',  'bbox': '[0.123,0.456,0.789,0.900]'},
    {'type': 'point', 'point': '[0.512,0.334]'},
    {'type': 'grasp', 'grasp': '[0.51,0.33]'},
    {'type': 'force', 'force': '12.30'},
    {'type': 'camera', 'camera': 'front', 'id': 0},
    {'type': 'trajectory', 'trajectory': '[0,1],[2,3]'},
    {'type': 'waypoint', 'waypoint': '[1.0,2.0]'},
]})
```

**逐帧视频**：

```python
session.add_message({'role': 'user', 'content': [
    {'type': 'video', 'video': 'demo.mp4',
     'frames': [{'image': frame_tensor, 'timestamp': '0.00'}, {'image': frame2}],
     'timestamps': ['0.00', '9.99']},
]})
```

渲染成 `<|modality_video_start|><|frame_start|><|timestamp_start|>t<|timestamp_end|>` +
图像三连 + `<|frame_end|><|frame_sep|>…<|modality_video_end|>`；帧内的 `timestamp` 优先，
否则取 `timestamps[帧下标]`。帧按顺序进 `images`，复用图像占位符展开。不带 `frames` 时仍是
单个 `<|modality_video_pad|>`（需要真正的视频塔）。

### Environment Role

```python
session.add_message({'role': 'environment', 'content': 'collision detected'})
```

渲染成 `<|im_start|><|environment|>…<|im_end|>`，是**逐轮**的独立 role：不会被并入开头 header，
默认 policy 为 `'all'`（环境反馈不进 loss）。`role_strict=True` 时它不会被当成非法 role。

---

## Notes

1. **Special Tokens**: Uses `[im_start]`, `[im_end]`, `[cot_start]`, `[cot_end]`, etc.
2. **Image / Audio / Vector Patches**: Automatically expands image, audio, action, proprio and
   tactile placeholders to patch tokens.
3. **Labels**: Masked positions have label -100 (ignored in loss).