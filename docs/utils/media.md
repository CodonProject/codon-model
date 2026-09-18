# Media Utilities Documentation

## Overview

`codon.utils.media` 是多模态输入的统一解码层：`Session` 与模型只接受张量（图片 `[C, H, W]`、
mel `[num_mel_bins, frames]`），而 `chat()` / `Service` 的调用方通常拿到的是文件路径、
base64 / data URL、PIL / numpy 对象。本模块把这些形态统一成模型要的张量，并在模型没有声明
对应模态能力时抛出明确错误。

## Exceptions

| Exception | 说明 |
|-----------|------|
| `UnsupportedModalityError(ValueError)` | 消息带了模型没声明的模态（图片 / 音频 / 视频）。服务端映射成 HTTP 400 `unsupported_modality`。 |

## Functions

### load_image()

```python
def load_image(source) -> torch.Tensor     # [3, H, W]，值域 0..1
```

接受：`torch.Tensor`（`[C,H,W]` / `[H,W]` / `[1,C,H,W]`，HWC 会自动转置）、`numpy.ndarray`、
`PIL.Image`、`bytes`、本地路径、裸 base64、`data:image/...;base64,...`、http(s) URL、
`{'url': ...}` 字典。灰色会复制成 3 通道，RGBA 丢掉 alpha，`uint8` / 大于 1 的值域按 255 归一化。

### load_mel()

```python
def load_mel(
    source,
    *,
    num_mel_bins: int = 80,
    sample_rate: int = 16000,
    max_frames: int = 3000,
) -> torch.Tensor                            # [num_mel_bins, frames]
```

| 载荷 | 行为 |
|------|------|
| mel 张量 / ndarray | 直接规整（`[frames, bins]` 会自动转置），裁到 `max_frames` |
| `.npy` / `.pt` 字节、路径、base64、data URL | `np.load` / `torch.load` 后同上 |
| PCM wav 字节 / 路径 / base64 | 标准库 `wave` 解码（无额外依赖），再按 Whisper 前端转 mel |
| 其它容器（mp3 / flac …） | `torchaudio.load`（需要 torchaudio / torchcodec；不可用时给明确报错） |
| `{'data': ..., 'format': ...}` | OpenAI 风格 `input_audio` 载荷 |

波形走 Whisper 前端：重采样到 16 kHz → `n_fft=400` / `hop=160` / 80 维 Slaney mel（power=2）
→ `log10(clamp(min=1e-10))`，与 `WhisperTinyAudioEncoder` 的输入分布一致。默认最多保留 3000 帧
（30 s，对应位置编码上限 1500 × conv 下采样 2），超出部分裁掉。

### modal_config()

```python
def modal_config(model) -> Tuple[int, int, int]
    # (image_patch_size, audio_pool_stride, audio_num_mel_bins)
```

读取模型上的多模态契约属性（`CausalLanguageModel` 默认全是 `None`），纯文本模型退回
`(12, 8, 80)`。`MotifChord` 返回 `(16, 8, 80)`，值来自 `MotifChordConfig`。

### model_modalities()

```python
def model_modalities(model) -> Dict[str, bool]
    # {'image': ..., 'audio': ..., 'video': ..., 'thinking': ..., 'tool': ...}
```

### has_media() / has_video_frames() / normalize_messages()

```python
def has_media(messages) -> Tuple[bool, bool, bool]        # (image, audio, video)

def has_video_frames(messages) -> bool                    # 有没有带 frames 的 video item

def normalize_messages(
    messages,
    *,
    image_capab: bool = False,
    audio_capab: bool = False,
    video_capab: bool = False,
    num_mel_bins: int = 80,
    sample_rate: int = 16000,
    max_frames: int = 3000,
) -> List[Dict[str, Any]]
```

`normalize_messages` 把列表 `content` 里的媒体项解码成张量，输出仍是 `Session.add_message`
吃的格式；已经解码好的张量会原样透传（可重复调用）。能力位为 `False` 而消息里带了该模态时抛
`UnsupportedModalityError`，服务端据此在生成之前返回 400，而不是把 unk token 喂给模型。

视频分两种：带 `frames` 的「逐帧视频」只要求 `image_capab`（帧解码成图像张量、复用图像塔），
裸视频（只有 url）要求 `video_capab`（需要真正的视频塔）。动作 / 本体感觉 / 触觉本来就是张量，
本模块原样透传；`bbox` / `point` / `ref` / `grasp` 这类 grounding 块的载荷是字符串，
由 `codon.j2` 负责包结构 token。

## 消息格式

```python
{'role': 'user', 'content': [
    {'type': 'text',  'text': 'describe this'},
    {'type': 'image', 'image': '<张量 / 路径 / base64 / data URL / http URL>'},   # 或 image_url
    {'type': 'audio', 'audio': '<mel 张量 / .npy / .pt / PCM wav>'},              # 或 input_audio
    {'type': 'video', 'video': 'demo.mp4',
     'frames': [{'image': '<同上>', 'timestamp': '0.00'}, ...],
     'timestamps': ['0.00', '9.99']},                                            # 逐帧视频
    {'type': 'action',  'action':  '<[T, action_dim] 张量>'},
    {'type': 'proprio', 'proprio': '<[T, proprio_dim] 张量>'},
    {'type': 'tactile', 'tactile': '<[T, tactile_dim] 张量>'},
]}
```

## 示例

```python
import torch
from codon.utils.media import load_image, load_mel, normalize_messages

image = load_image('data:image/png;base64,iVBORw0...')     # [3, H, W]
mel = load_mel('speech.wav')                              # [80, T]

messages = normalize_messages(
    [{'role': 'user', 'content': [
        {'type': 'image', 'image': image},
        {'type': 'audio', 'audio': mel},
        {'type': 'text',  'text': '描述图片并转写音频'},
    ]}],
    image_capab=True,
    audio_capab=True,
)
```

配合 `chat()` 使用时不需要手动调用 `normalize_messages`（`chat` 内部会做）：

```python
import torch
from codon.utils.generate import chat

for chunk in chat(
    model=model, tokenizer=tokenizer, device=torch.device('cpu'),
    messages=[{'role': 'user', 'content': [
        {'type': 'image', 'image': 'cat.png'},
        {'type': 'text',  'text': '描述这张图'},
    ]}],
):
    print(chunk.content, end='', flush=True)
```
