'''
多模态输入的统一下载 / 解码层。

`Session` 与模型只接受已经解码好的张量（图片 `[C, H, W]` float、mel `[num_mel_bins, frames]`），
但 `codon.utils.generate.chat` 与 `codon.utils.service.Service` 的调用方通常拿到的是文件路径、
base64 / data URL（HTTP 请求）或 numpy / PIL 对象。本模块把这些形态统一成模型要的张量，
并在模型没有声明该模态能力时抛出明确错误。

约定（与 `codon/res/codon.j2` 一致）：

    {'type': 'text',  'text': str}
    {'type': 'image', 'image': <tensor | path | data url | base64 | http url | PIL | ndarray>}
    {'type': 'audio', 'audio': <mel tensor | .npy/.pt | wav 等波形>}
    {'type': 'video', 'video': <url/path>, 'frames': [{'image': ..., 'timestamp': '0.00'}, ...],
                      'timestamps': [...]}        # 逐帧视频复用图像塔
    {'type': 'action',  'action':  <tensor [T, action_dim]>}     # 连续值占位符
    {'type': 'proprio', 'proprio': <tensor [T, proprio_dim]>}
    {'type': 'tactile', 'tactile': <tensor [T, tactile_dim]>}

动作 / 本体感觉 / 触觉本来就是张量，本模块只做视频帧与图片 / 音频的载荷解码；
`bbox` / `point` / `ref` / `grasp` / `force` / `camera` / `trajectory` / `waypoint`
这类 grounding 块由调用方给「已格式化字符串」，模板负责包结构 token，这里不碰。

OpenAI 风格的写法也能直接吃：`{'type': 'image_url', 'image_url': {'url': ...}}` 与
`{'type': 'input_audio', 'input_audio': {'data': <base64>, 'format': 'wav'}}`。
'''

import base64
import io
import os
import urllib.request
import wave
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image


#: Whisper 前端的参数（与 DINOv3 / Whisper Tiny 塔的输入约定一致）
WHISPER_SAMPLE_RATE = 16000
WHISPER_N_FFT = 400
WHISPER_HOP_LENGTH = 160
WHISPER_MAX_FRAMES = 3000          # 30s

#: 图片 / 音频下载上限（HTTP 或 base64），避免一个请求把内存打爆
MAX_MEDIA_BYTES = 32 * 1024 * 1024


class UnsupportedModalityError(ValueError):
    '''模型没有声明该模态能力（或该模态根本没有编码塔）时抛出。'''


# ---------------------------------------------------------------- 通用载荷读取

def _read_bytes(source: Union[str, bytes, bytearray]) -> bytes:
    '''
    str / bytes -> 原始字节。

    str 支持三种写法：本地文件路径、`data:<mime>;base64,...`（或裸 base64）、http(s) URL。
    '''
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)

    if not isinstance(source, str):
        raise TypeError(f'cannot read bytes from {type(source).__name__}')

    if os.path.isfile(source):
        with open(source, 'rb') as f:
            return f.read(MAX_MEDIA_BYTES)

    if source.startswith(('http://', 'https://')):
        with urllib.request.urlopen(source, timeout=30) as response:
            return response.read(MAX_MEDIA_BYTES)

    payload = source.split(',', 1)[1] if source.startswith('data:') and ',' in source else source
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError(
            'string payload must be an existing file path, an http(s) URL, '
            'a data URL or a base64 string'
        ) from exc


def _payload_of(item: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    '''从消息 item 里取出模态载荷（兼容 `{'image_url': {'url': ...}}` 这类嵌套写法）。'''
    for key in keys:
        if key in item and item[key] is not None:
            value = item[key]
            if isinstance(value, dict):
                for inner in ('url', 'data', 'base64', 'audio', 'image', 'mel'):
                    if inner in value:
                        return value[inner]
            return value
    raise ValueError(f'message item is missing any of {keys}: {item!r}')


# ---------------------------------------------------------------- 图片

def _image_from_tensor(image: torch.Tensor) -> torch.Tensor:
    '''张量 -> [3, H, W] float。接受 [C,H,W] / [H,W] / 单个样本的 [1,C,H,W]。'''
    image = image.float()
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.ndim != 3:
        raise ValueError(f'image tensor must be [C, H, W] or [H, W], got {tuple(image.shape)}')

    # HWC：最后一维是通道数、第一维不是
    if image.shape[0] not in (1, 3, 4) and image.shape[-1] in (1, 3, 4):
        image = image.permute(2, 0, 1)

    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] == 4:
        image = image[:3]
    elif image.shape[0] != 3:
        raise ValueError(f'image tensor must have 1 / 3 / 4 channels, got {image.shape[0]}')

    if float(image.max()) > 1.0:
        image = image / 255.0
    return image.contiguous()


def load_image(source: Any) -> torch.Tensor:
    '''
    把各种图片写法统一成 `[3, H, W]` float（0..1）。

    Args:
        source: torch.Tensor / numpy.ndarray / PIL.Image / bytes / 路径 / base64 / data URL /
            http(s) URL / `{'url': ...}` 字典。

    Returns:
        torch.Tensor: `[3, H, W]`，值域 0..1。
    '''
    if isinstance(source, torch.Tensor):
        return _image_from_tensor(source)

    if isinstance(source, np.ndarray):
        array = source
        if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
            array = array.transpose(2, 0, 1)          # HWC -> CHW
        return _image_from_tensor(torch.from_numpy(np.ascontiguousarray(array)))

    if isinstance(source, Image.Image):
        array = np.array(source.convert('RGB')).transpose(2, 0, 1)
        return _image_from_tensor(torch.from_numpy(np.ascontiguousarray(array)))

    if isinstance(source, dict):
        return load_image(_payload_of(source, ('url', 'data', 'base64', 'image')))

    if isinstance(source, (bytes, bytearray)):
        with Image.open(io.BytesIO(bytes(source))) as image:
            array = np.array(image.convert('RGB')).transpose(2, 0, 1)
        return _image_from_tensor(torch.from_numpy(np.ascontiguousarray(array)))

    if isinstance(source, str):
        return load_image(_read_bytes(source))

    raise TypeError(f'unsupported image payload type: {type(source).__name__}')


# ---------------------------------------------------------------- 音频

def _as_mel(tensor: torch.Tensor, num_mel_bins: int, max_frames: int) -> torch.Tensor:
    '''规整成 [num_mel_bins, frames]（顺手裁到 max_frames 以内）。'''
    tensor = tensor.float()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(
            f'mel must be [num_mel_bins, frames], got {tuple(tensor.shape)}'
        )
    if tensor.shape[0] != num_mel_bins:
        if tensor.shape[1] == num_mel_bins:
            tensor = tensor.transpose(0, 1)
        else:
            raise ValueError(
                f'mel has {tensor.shape[0]} bins, expected {num_mel_bins}'
            )
    return tensor[:, :max_frames].contiguous()


def _waveform_to_mel(
    waveform: torch.Tensor,
    sample_rate: int,
    num_mel_bins: int,
    max_frames: int,
) -> torch.Tensor:
    '''波形 -> Whisper 风格的 log-mel（Slaney mel + log10 截断）。需要 torchaudio。'''
    try:
        import torchaudio
    except ImportError as exc:                    # pragma: no cover - 取决于环境
        raise RuntimeError(
            'waveform audio requires torchaudio; pass a precomputed mel '
            '[num_mel_bins, frames] instead'
        ) from exc

    waveform = waveform.float()
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)           # 多声道 -> 单声道
    if waveform.ndim != 1:
        raise ValueError(f'waveform must be 1D, got {tuple(waveform.shape)}')

    if sample_rate != WHISPER_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, WHISPER_SAMPLE_RATE
        )

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=WHISPER_SAMPLE_RATE,
        n_fft=WHISPER_N_FFT,
        hop_length=WHISPER_HOP_LENGTH,
        n_mels=num_mel_bins,
        power=2.0,
        mel_scale='slaney',
        norm='slaney',
    )(waveform)
    return torch.log10(mel.clamp(min=1e-10))[:, :max_frames]


def _decode_wav_bytes(data: bytes) -> Optional[Tuple[torch.Tensor, int]]:
    '''
    PCM wav -> ([channels, samples] float, sample_rate)，用标准库 `wave` 解码（无额外依赖）。

    不是 wav / 不支持的位宽时返回 None，由调用方继续尝试其它解码器。
    '''
    try:
        with wave.open(io.BytesIO(data), 'rb') as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except Exception:
        return None

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None or not frames:
        return None

    array = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if width == 1:
        array = (array - 128.0) / 128.0
    elif width == 2:
        array = array / 32768.0
    else:
        array = array / 2147483648.0

    waveform = torch.from_numpy(array.copy())
    if channels > 1:
        waveform = waveform.view(-1, channels).transpose(0, 1)     # [C, T]
    else:
        waveform = waveform.unsqueeze(0)
    return waveform, int(rate)


def _bytes_to_audio_tensor(data: bytes) -> Tuple[torch.Tensor, int]:
    '''字节 -> (张量, 采样率)。

    依次尝试：mel 数组（.npy / .pt）-> PCM wav（标准库）-> 其它容器（torchaudio）。
    mel 数组返回的采样率为 0，表示「已经是 mel，不需要再转」。
    '''
    try:
        array = np.load(io.BytesIO(data), allow_pickle=False)
        return torch.from_numpy(np.ascontiguousarray(array)), 0
    except Exception:
        pass

    try:
        tensor = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
        if isinstance(tensor, torch.Tensor):
            return tensor, 0
    except Exception:
        pass

    decoded = _decode_wav_bytes(data)
    if decoded is not None:
        return decoded

    try:
        import torchaudio
    except ImportError as exc:                    # pragma: no cover - 取决于环境
        raise RuntimeError(
            'audio payload is neither a .npy / .pt mel nor a PCM wav and torchaudio '
            'is unavailable; pass a precomputed mel [num_mel_bins, frames] instead'
        ) from exc

    waveform, sample_rate = torchaudio.load(io.BytesIO(data))
    return waveform, int(sample_rate)


def load_mel(
    source: Any,
    *,
    num_mel_bins: int = 80,
    sample_rate: int = WHISPER_SAMPLE_RATE,
    max_frames: int = WHISPER_MAX_FRAMES,
) -> torch.Tensor:
    '''
    把各种音频写法统一成 `[num_mel_bins, frames]` 的 mel 谱。

    支持：mel 张量 / numpy 数组（2D，或 `[1, bins, frames]`）、`.npy` / `.pt` 路径或字节、
    base64 / data URL、`{'data': ..., 'format': ...}`，以及波形（wav 等，需要 torchaudio）
    —— 波形会按 Whisper 前端（16k / n_fft 400 / hop 160 / Slaney mel / log10）转成 mel。

    Args:
        source: 上述任意形态。
        num_mel_bins: mel 维数，需与音频塔一致（Whisper Tiny 是 80）。
        sample_rate: 波形载荷的采样率（仅在无法从容器里读到采样率时作为兜底）。
        max_frames: 最多保留的 mel 帧数（Whisper 位置编码上限 1500 -> 3000 帧 / 30s）。

    Returns:
        torch.Tensor: `[num_mel_bins, frames]`。
    '''
    if isinstance(source, torch.Tensor):
        return _as_mel(source, num_mel_bins, max_frames)

    if isinstance(source, np.ndarray):
        return _as_mel(torch.from_numpy(np.ascontiguousarray(source)), num_mel_bins, max_frames)

    if isinstance(source, dict):
        return load_mel(
            _payload_of(source, ('data', 'audio', 'mel', 'base64', 'url')),
            num_mel_bins=num_mel_bins, sample_rate=sample_rate, max_frames=max_frames,
        )

    if isinstance(source, (bytes, bytearray, str)):
        # 路径 / base64 / data URL / http(s) 都先变成字节，再按内容判定是 mel 还是波形
        data = _read_bytes(source)
        tensor, rate = _bytes_to_audio_tensor(data)
        if rate == 0:                              # mel 数组
            return _as_mel(tensor, num_mel_bins, max_frames)
        return _waveform_to_mel(tensor, rate, num_mel_bins, max_frames)

    raise TypeError(f'unsupported audio payload type: {type(source).__name__}')


# ---------------------------------------------------------------- 模型能力 / 会话参数

def model_modalities(model: Any) -> Dict[str, bool]:
    '''模型声明的模态能力（`supports_image` / `supports_audio` / `supports_video` / `supports_thinking` / `supports_tool`）。'''
    capabilities: Dict[str, bool] = {}
    for name in ('image', 'audio', 'video', 'thinking', 'tool'):
        value = getattr(model, f'supports_{name}', False)
        capabilities[name] = bool(value) if isinstance(value, bool) else False
    return capabilities


def modal_config(model: Any) -> Tuple[int, int, int]:
    '''
    从模型上读多模态占位符需要的参数（纯文本模型退回默认值）。

    Returns:
        Tuple[int, int, int]: `(image_patch_size, audio_pool_stride, audio_num_mel_bins)`。
    '''
    def resolve(name: str, default: int) -> int:
        value = getattr(model, name, None)
        return int(value) if isinstance(value, int) and value > 0 else default

    return (
        resolve('image_patch_size', 12),
        resolve('audio_pool_stride', 8),
        resolve('audio_num_mel_bins', 80),
    )


def _media_kind(item: Dict[str, Any]) -> Optional[str]:
    '''消息 item 属于哪种模态：'text' / 'image' / 'audio' / 'video' / None。'''
    item_type = str(item.get('type', '')).lower()
    if item_type in ('image', 'image_url') or 'image_url' in item or 'image' in item:
        return 'image'
    if item_type in ('audio', 'input_audio') or 'audio_url' in item or 'audio' in item:
        return 'audio'
    if item_type in ('video', 'video_url') or 'video_url' in item or 'video' in item:
        return 'video'
    return None


def has_media(messages: List[Dict[str, Any]]) -> Tuple[bool, bool, bool]:
    '''消息里是否带了图片 / 音频 / 视频。'''
    found = [False, False, False]
    for message in messages:
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            kind = _media_kind(item)
            if kind == 'image':
                found[0] = True
            elif kind == 'audio':
                found[1] = True
            elif kind == 'video':
                found[2] = True
    return found[0], found[1], found[2]


def has_video_frames(messages: List[Dict[str, Any]]) -> bool:
    '''消息里是否有「逐帧视频」（带 frames 的 video item）—— 逐帧视频复用图像塔。'''
    for message in messages:
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if _media_kind(item) == 'video' and item.get('frames'):
                return True
    return False


def _normalize_frames(frames: List[Any]) -> List[Any]:
    '''逐帧视频的帧载荷解码成 `{'image': 张量, 'timestamp': ...}`（已经是张量的帧原样保留）。'''
    normalized: List[Any] = []
    for frame in frames:
        if isinstance(frame, torch.Tensor):
            normalized.append({'image': frame})
            continue
        if not isinstance(frame, dict):
            normalized.append({'image': load_image(frame)})
            continue

        payload = None
        for key in ('image', 'image_url', 'pixel_values'):
            if frame.get(key) is not None:
                payload = frame[key]
                break
        if payload is None:
            raise ValueError(f'video frame is missing an image payload: {frame!r}')

        rebuilt = dict(frame)
        rebuilt['image'] = payload if isinstance(payload, torch.Tensor) else load_image(payload)
        # 帧内的 'image_url' 只是为了兼容输入写法，统一收敛到 'image'
        rebuilt.pop('image_url', None)
        normalized.append(rebuilt)
    return normalized


def normalize_messages(
    messages: List[Dict[str, Any]],
    *,
    image_capab: bool = False,
    audio_capab: bool = False,
    video_capab: bool = False,
    num_mel_bins: int = 80,
    sample_rate: int = WHISPER_SAMPLE_RATE,
    max_frames: int = WHISPER_MAX_FRAMES,
) -> List[Dict[str, Any]]:
    '''
    把消息里的媒体载荷解码成张量，并校验模型是否声明了对应能力。

    返回的消息仍是 `Session.add_message` 吃的格式（`{'type': 'image', 'image': tensor}` /
    `{'type': 'audio', 'audio': mel}`），文本与其它字段原样保留。
    已经解码好的张量会原样传下去（可重复调用）。

    Raises:
        UnsupportedModalityError: 消息带了某模态，但模型没有声明该能力（或视频根本没有塔）。
        ValueError / TypeError: 载荷无法解码。
    '''
    normalized: List[Dict[str, Any]] = []

    for message in messages:
        new_message = dict(message)
        content = new_message.get('content')
        if not isinstance(content, list):
            normalized.append(new_message)
            continue

        items: List[Any] = []
        for item in content:
            if not isinstance(item, dict):
                items.append(item)
                continue

            kind = _media_kind(item)
            if kind == 'image':
                if not image_capab:
                    raise UnsupportedModalityError(
                        'this model does not support image input (supports_image=False)'
                    )
                items.append({
                    'type': 'image',
                    'image': load_image(_payload_of(item, ('image', 'image_url'))),
                })
            elif kind == 'audio':
                if not audio_capab:
                    raise UnsupportedModalityError(
                        'this model does not support audio input (supports_audio=False)'
                    )
                items.append({
                    'type': 'audio',
                    'audio': load_mel(
                        _payload_of(item, ('audio', 'audio_url', 'input_audio', 'mel')),
                        num_mel_bins=num_mel_bins,
                        sample_rate=sample_rate,
                        max_frames=max_frames,
                    ),
                })
            elif kind == 'video':
                frames = item.get('frames')
                if frames:
                    # 逐帧视频复用图像塔：帧解码成图像张量，Session 按帧顺序塞进 images
                    if not image_capab:
                        raise UnsupportedModalityError(
                            'video frames reuse the image tower, but this model does not '
                            'support image input (supports_image=False)'
                        )
                    items.append({**item, 'frames': _normalize_frames(list(frames))})
                elif not video_capab:
                    raise UnsupportedModalityError(
                        'this model does not support video input (supports_video=False); '
                        'pass per-frame images via {\'type\': \'video\', \'frames\': [...]} instead'
                    )
                else:
                    items.append(item)
            else:
                items.append(item)

        new_message['content'] = items
        normalized.append(new_message)

    return normalized
