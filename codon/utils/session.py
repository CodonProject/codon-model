from codon.utils.tokens import PackedTokenizer
from codon.res import EFFORT_DEFAULT, resolve_effort
from dataclasses import dataclass, field
from typing import Optional, Union, Literal, Sequence, Any
import copy
import torch
import json


MaskPolicy = Union[Literal['all', 'none', 'content', 'thought', 'answer', 'fim'], Sequence[bool]]

# A2 / codon.j2 词表风格（<|...|>）。Session 默认用 A1 方括号 token 名；
# 若词表含 <|im_start|> 则按此映射整体切换。
_ANGLE_TOKEN_NAMES = {
    'im_start': '<|im_start|>', 'im_end': '<|im_end|>',
    'system': '<|system|>', 'user': '<|user|>',
    'model': '<|model|>', 'tool': '<|tool_response|>',
    # 逐轮环境反馈（codon.j2: <|im_start|><|environment|>...<|im_end|>）
    'environment': '<|environment|>',
    'fim': '<|fim_middle|>',
    'cot_start': '<|thought_start|>', 'cot_end': '<|thought_end|>',
    # 思考强度，紧跟在 cot_start 之后
    'effort_low': '<|effort_low|>', 'effort_high': '<|effort_high|>', 'effort_max': '<|effort_max|>',
    # 工具调用（codon.j2: <|tool_call_start|>name<|tool_name_divider|>{...}<|tool_call_end|>）
    'tool_call_start': '<|tool_call_start|>', 'tool_call_end': '<|tool_call_end|>',
    'tool_name_divider': '<|tool_name_divider|>',
    'fim_pre': '<|fim_prefix|>', 'fim_mid': '<|fim_middle|>', 'fim_suf': '<|fim_suffix|>',
    'pad': '<|pad|>',
    'image_start': '<|modality_image_start|>', 'image_end': '<|modality_image_end|>',
    'image_patch': '<|modality_image_pad|>',
    'audio_start': '<|modality_audio_start|>', 'audio_end': '<|modality_audio_end|>',
    'audio_patch': '<|modality_audio_pad|>',
    'video_start': '<|modality_video_start|>', 'video_end': '<|modality_video_end|>',
    'video_patch': '<|modality_video_pad|>',
    # 视频帧 / 时间戳（codon.j2: 逐帧 image 三连）
    'frame_start': '<|frame_start|>', 'frame_end': '<|frame_end|>',
    'frame_sep': '<|frame_sep|>',
    'timestamp_start': '<|timestamp_start|>', 'timestamp_end': '<|timestamp_end|>',
    # 动作 / 本体感觉 / 触觉：连续值占位符（start + pad + end 三连，Session 负责展开）
    'action_start': '<|action_start|>', 'action_pad': '<|action_pad|>',
    'action_end': '<|action_end|>',
    'action_chunk_start': '<|action_chunk_start|>', 'action_sep': '<|action_sep|>',
    'action_chunk_end': '<|action_chunk_end|>',
    'action_terminate': '<|action_terminate|>', 'action_continue': '<|action_continue|>',
    'proprio_start': '<|proprio_start|>', 'proprio_pad': '<|proprio_pad|>',
    'proprio_end': '<|proprio_end|>',
    'tactile_start': '<|tactile_start|>', 'tactile_pad': '<|tactile_pad|>',
    'tactile_end': '<|tactile_end|>',
    # 观测 / grounding：纯文本块，模板只负责包 token
    'force_start': '<|force_start|>', 'force_end': '<|force_end|>',
    'camera_start': '<|camera_start|>', 'camera_end': '<|camera_end|>',
    'camera_id': '<|camera_id|>',
    'trajectory_start': '<|trajectory_start|>', 'trajectory_end': '<|trajectory_end|>',
    'waypoint_start': '<|waypoint_start|>', 'waypoint_end': '<|waypoint_end|>',
    'bbox_start': '<|bbox_start|>', 'bbox_end': '<|bbox_end|>',
    'point_start': '<|point_start|>', 'point_end': '<|point_end|>',
    'ref_start': '<|ref_start|>', 'ref_end': '<|ref_end|>',
    'grasp_start': '<|grasp_start|>', 'grasp_end': '<|grasp_end|>',
    # codon/res LM['spec_token'] 里的其余保留 token（词表里有就解析，没有则为 None）
    'safe_escape': '<|safe_escape|>', 'unk': '<|unk|>', 'sep': '<|sep|>',
    'mask': '<|mask|>', 'bos': '<|bos|>', 'eos': '<|eos|>', 'endoftext': '<|endoftext|>',
}

#: 连续值占位符模态：逻辑名 -> (start, end, pad) 的逻辑名。
#: 它们的 codon.j2 形态统一是 `start + 单个 pad + end` 三连，由 Session 按张量行数展开。
_PAD_BLOCK_MODALITIES = {
    'image': ('image_start', 'image_end', 'image_patch'),
    'audio': ('audio_start', 'audio_end', 'audio_patch'),
    'action': ('action_start', 'action_end', 'action_pad'),
    'proprio': ('proprio_start', 'proprio_end', 'proprio_pad'),
    'tactile': ('tactile_start', 'tactile_end', 'tactile_pad'),
}

#: 逻辑名别名：新词表的叫法（thought_end / tool_response…）也能直接查。
_TOKEN_ALIASES = {
    'thought_start': 'cot_start', 'thought_end': 'cot_end',
    'tool_response': 'tool', 'tool_call': 'tool_call_start',
    'modality_image_start': 'image_start', 'modality_image_end': 'image_end',
    'modality_image_pad': 'image_patch',
}

#: A1 词表风格（方括号 token 名）。
_A1_TOKEN_NAMES = {
    'im_start': '[im_start]', 'im_end': '[im_end]',
    'system': '[system]', 'user': '[user]',
    'model': '[model]', 'tool': '[tool]',
    'fim': '[fim]',
    'cot_start': '[cot_start]', 'cot_end': '[cot_end]',
    # 思考强度，紧跟在 cot_start 之后
    'effort_low': '[effort_low]', 'effort_high': '[effort_high]', 'effort_max': '[effort_max]',
    # 工具调用（A1 词表里可能没有，那就解析成 None）
    'tool_call_start': '[tool_call_start]', 'tool_call_end': '[tool_call_end]',
    'tool_name_divider': '[tool_name_divider]',
    'fim_pre': '[fim_pre]', 'fim_mid': '[fim_mid]', 'fim_suf': '[fim_suf]',
    'pad': '[pad]',
    'image_start': '[image_start]', 'image_end': '[image_end]',
    'image_patch': '[unused_43]',
    'audio_start': '[audio_start]', 'audio_end': '[audio_end]',
    'video_start': '[video_start]', 'video_end': '[video_end]',
}


def _match_token_key(table: dict, name: str) -> Optional[str]:
    '''把各种写法归一到逻辑名：'im_end' / '[im_end]' / '<|im_end|>' / 'thought_end'。'''
    if not isinstance(name, str):
        return None
    if name in table:
        return name
    if name in _TOKEN_ALIASES:
        return _TOKEN_ALIASES[name]
    stripped = name.strip().strip('[]<>|').strip()
    if stripped in table:
        return stripped
    if stripped in _TOKEN_ALIASES:
        return _TOKEN_ALIASES[stripped]
    for key, text in table.items():
        if text == name:
            return key
    return None


def token_name_table(tokenizer: PackedTokenizer) -> dict:
    '''按词表风格返回「逻辑名 -> token 文本」表（A2 / chord 词表是 <|...|> 风格）。'''
    if tokenizer.token_to_id(_ANGLE_TOKEN_NAMES['im_start']) is not None:
        return dict(_ANGLE_TOKEN_NAMES)
    return dict(_A1_TOKEN_NAMES)


def resolve_token_name(tokenizer: PackedTokenizer, name: str) -> Optional[str]:
    '''逻辑名 / A1 写法 / A2 写法 -> 实际 token 文本（词表里没有则 None）。'''
    table = token_name_table(tokenizer)
    key = _match_token_key(table, name)
    if key is not None:
        return table[key]
    return name if tokenizer.token_to_id(name) is not None else None


def resolve_token_id(tokenizer: PackedTokenizer, name: str) -> Optional[int]:
    '''
    逻辑名 / A1 写法 / A2 写法 -> token id（词表里没有则 None），不需要构造 Session。

        resolve_token_id(tokenizer, '[im_end]')      # A1 词表
        resolve_token_id(tokenizer, '<|im_end|>')    # A2 / chord 词表
        resolve_token_id(tokenizer, 'im_end')        # 两种词表都可用
    '''
    resolved = resolve_token_name(tokenizer, name)
    return tokenizer.token_to_id(resolved) if resolved else None


@dataclass
class Message:
    ids: list[int]
    ignore_mask: list[bool] = field(default_factory=list)
    role: Optional[str] = None
    images: list[torch.Tensor] = field(default_factory=list)
    audios: list[torch.Tensor] = field(default_factory=list)     # mel 谱 [num_mel_bins, frames]
    actions: list[torch.Tensor] = field(default_factory=list)    # 动作块 [T, action_dim] / [action_dim]
    proprios: list[torch.Tensor] = field(default_factory=list)   # 本体感觉 [T, proprio_dim]
    tactiles: list[torch.Tensor] = field(default_factory=list)   # 触觉读数 [T, tactile_dim]

    def __post_init__(self):
        if not self.ignore_mask:
            self.ignore_mask = [False] * len(self.ids)
        if len(self.ignore_mask) != len(self.ids):
            raise ValueError(
                f'ignore_mask length {len(self.ignore_mask)} != ids length {len(self.ids)}'
            )

    def __len__(self) -> int:
        return len(self.ids)

    def _resolve_range(self, begin: int, end: int, include: tuple[bool, bool]) -> tuple[int, int]:
        lo = begin if include[0] else begin + 1
        hi = end + 1 if include[1] else end
        return max(0, lo), min(len(self.ids), hi)

    def mask_all(self) -> 'Message':
        self.ignore_mask = [True] * len(self.ids)
        return self

    def unmask_all(self) -> 'Message':
        self.ignore_mask = [False] * len(self.ids)
        return self

    def mask_before(self, index: int) -> 'Message':
        for i in range(max(0, min(index, len(self.ids)))):
            self.ignore_mask[i] = True
        return self

    def mask_after(self, index: int) -> 'Message':
        for i in range(max(0, index), len(self.ids)):
            self.ignore_mask[i] = True
        return self

    def mask_between(
        self, begin: int, end: int,
        include_boundaries: tuple[bool, bool] = (False, False)
    ) -> 'Message':
        lo, hi = self._resolve_range(begin, end, include_boundaries)
        for i in range(lo, hi):
            self.ignore_mask[i] = True
        return self

    def unmask_between(
        self, begin: int, end: int,
        include_boundaries: tuple[bool, bool] = (False, False)
    ) -> 'Message':
        lo, hi = self._resolve_range(begin, end, include_boundaries)
        for i in range(lo, hi):
            self.ignore_mask[i] = False
        return self

    def insert_before(self, index: int, data: list[int], ignore_mask: bool = True) -> 'Message':
        mask = [ignore_mask] * len(data) if isinstance(ignore_mask, bool) else list(ignore_mask)
        self.ids = self.ids[:index] + list(data) + self.ids[index:]
        self.ignore_mask = self.ignore_mask[:index] + mask + self.ignore_mask[index:]
        return self

    def insert_after(self, index: int, data: list[int], ignore_mask: bool = True) -> 'Message':
        return self.insert_before(index + 1, data, ignore_mask)

    def find(self, token_id: int, start: int = 0) -> int:
        try:
            return self.ids.index(token_id, start)
        except ValueError:
            return -1

    def find_last(self, token_id: int) -> int:
        for i in range(len(self.ids) - 1, -1, -1):
            if self.ids[i] == token_id:
                return i
        return -1


class Session:
    _ROLE_ALIAS = {
        'assistant': 'model',
        'instruction': 'system',
        'developer': 'system',
    }

    def __init__(
        self,
        tokenizer: PackedTokenizer,
        patch_size: int = 12,
        audio_pool_stride: int = 8,
        image_capab: Optional[bool] = None,
        audio_capab: Optional[bool] = None,
        video_capab: bool = False,
    ):
        '''
        Args:
            tokenizer: 已注入 chat template（A2 / chord 词表对应 codon.j2）的分词器。
            patch_size: 视觉塔的图像 patch 尺寸，决定 <|modality_image_pad|> 展开成几个占位符
                （DINOv3 ViT-B/16 用 16，MotifV1 用 12）。
            audio_pool_stride: 音频塔在 encoder 输出之后的池化步长，决定
                <|modality_audio_pad|> 展开成几个占位符（Whisper Tiny + pool 8 -> 187）。
            image_capab / audio_capab: 渲染模板时的模态能力开关。None（默认）表示按当前消息
                是否带了该模态自动决定：带了就渲染成占位符，没带就不影响。
                codon.j2 里这两个变量默认 false，必须传 true 才会渲染
                <|modality_image_start|>... 而不是 "Your image modality is unsupported" 文本。
            video_capab: 视频能力开关，默认 False（模型侧没有视频塔）。
        '''
        self.tokenizer = tokenizer
        self.patch_size = patch_size
        self.audio_pool_stride = max(1, int(audio_pool_stride))
        self.image_capab = image_capab
        self.audio_capab = audio_capab
        self.video_capab = video_capab
        self.messages: list[Message] = []

        self._tokens = dict(_A1_TOKEN_NAMES)
        self._ids: dict[str, Optional[int]] = {}
        self._maybe_use_angle_tokens()
        self._resolve_specials()

        self.policy: dict[str, MaskPolicy] = {
            'system': 'all',
            'user': 'all',
            'environment': 'all',        # 环境反馈不进 loss（codon.j2 的 <|environment|>）
            'tool': 'all',
            'model': 'content',
            'fim': 'fim',
        }

    def _maybe_use_angle_tokens(self) -> None:
        '''A2 词表（<|im_start|> 风格）探测：切换 token 名并从 codon.j2 注入空模板。'''
        if self.tokenizer.token_to_id('<|im_start|>') is None:
            return                       # A1 方括号风格，保持默认
        self._tokens = dict(_ANGLE_TOKEN_NAMES)
        template = getattr(self.tokenizer, 'template', '')
        if not (template or '').strip():
            try:
                from codon.res import LM
                with open(LM['jinja'], encoding='utf-8') as f:
                    self.tokenizer.set_chat_template(f.read())
            except Exception:
                pass

    def _resolve_specials(self) -> None:
        self._ids = {k: self.tokenizer.token_to_id(v) for k, v in self._tokens.items()}

    # ---- 特殊 token 查询（屏蔽两种词表风格的差异）----
    @property
    def tokens(self) -> dict:
        '''逻辑名 -> 当前词表风格下的 token 文本。'''
        return dict(self._tokens)

    def _token_key(self, name: str) -> Optional[str]:
        '''把各种写法归一到逻辑名（见 `_match_token_key`）。'''
        return _match_token_key(self._tokens, name)

    def token_name(self, name: str) -> Optional[str]:
        '''逻辑名 / A1 写法 / A2 写法 -> 当前词表风格下的 token 文本（没有则 None）。'''
        key = self._token_key(name)
        return self._tokens.get(key) if key is not None else None

    def token_id(self, name: str) -> Optional[int]:
        '''
        逻辑名 / A1 写法 / A2 写法 -> token id（词表里没有则 None）。

        A1 词表（`[im_end]`）与 A2 / chord 词表（`<|im_end|>`）可以混着传，
        内部按当前词表的实际风格解析，例如：

            session.token_id('im_end')          # 两种风格都可用
            session.token_id('[im_end]')        # A1 写法
            session.token_id('<|im_end|>')      # A2 写法
            session.token_id('thought_end')     # 别名 -> cot_end
        '''
        key = self._token_key(name)
        if key is not None:
            return self._ids.get(key)
        return self.tokenizer.token_to_id(name)     # 词表里恰好就有这个 token 名

    def special_ids(self) -> set:
        '''当前词表风格下已解析出的特殊 token id 集合（用于过滤生成文本里的结构 token）。'''
        return {token_id for token_id in self._ids.values() if token_id is not None}

    def set_token(self, mapping: dict) -> 'Session':
        self._tokens.update(mapping)
        self._resolve_specials()
        return self

    def set_policy(self, role: str, policy: MaskPolicy) -> 'Session':
        self.policy[self._ROLE_ALIAS.get(role, role)] = policy
        return self

    def _encode_message(self, message: dict) -> list[int]:
        msg = copy.deepcopy(message)
        role = self._ROLE_ALIAS.get(msg.get('role', 'user'), msg.get('role', 'user'))
        msg['role'] = role
        if role == 'model':
            msg.setdefault('thought', None)
            msg.setdefault('reasoning_content', None)

        # codon.j2 的模态能力开关默认 false：不显式传 true 就只会渲染
        # "Your image/audio modality is unsupported" 文本，占位符根本不会出现。
        # 逐帧视频（video + frames）复用图像塔，因此也要算进 has_image。
        content = msg.get('content')
        items = content if isinstance(content, list) else []

        def _is_type(it, *names):
            return isinstance(it, dict) and str(it.get('type', '')).lower() in names

        has_image = any(
            _is_type(it, 'image') or (_is_type(it, 'video') and it.get('frames'))
            for it in items
        )
        has_audio = any(_is_type(it, 'audio') for it in items)

        return self.tokenizer.apply_chat_template(
            [msg],
            add_generation_prompt=False,
            tokenize=True,
            image_capab=has_image if self.image_capab is None else bool(self.image_capab),
            audio_capab=has_audio if self.audio_capab is None else bool(self.audio_capab),
            video_capab=bool(self.video_capab),
        )

    def _apply_policy(self, msg: Message, policy: MaskPolicy) -> None:
        if isinstance(policy, (list, tuple)):
            if len(policy) != len(msg):
                raise ValueError(f'explicit mask length {len(policy)} != message length {len(msg)}')
            msg.ignore_mask = [bool(x) for x in policy]
            return

        if policy == 'all':
            msg.mask_all()
        elif policy == 'none':
            msg.unmask_all()
        elif policy == 'content':
            # 只对 model 回复部分算 loss：从 thought_start（有思考时）或 role 标签
            # 之后开始 unmask 到消息末尾（含 im_end），排除 role 标签本身入 loss。
            msg.mask_all()
            anchor = self._ids.get('cot_start')
            idx = msg.find(anchor) if anchor is not None else -1
            if idx < 0:
                role_id = self._ids.get(msg.role or '')
                idx = msg.find(role_id) if role_id is not None else -1
                if idx >= 0:
                    idx += 1             # 跳过 role 标签，从回复正文起算
            if idx >= 0:
                msg.unmask_between(idx, len(msg), include_boundaries=(True, True))
        elif policy == 'thought':
            msg.mask_all()
            s_id, e_id = self._ids.get('cot_start'), self._ids.get('cot_end')
            if s_id is None or e_id is None:
                return
            s = msg.find(s_id)
            e = msg.find(e_id, max(s, 0))
            if s >= 0 and e > s:
                msg.unmask_between(s, e, include_boundaries=(False, False))
        elif policy == 'answer':
            msg.mask_all()
            cot_end_id = self._ids.get('cot_end')
            im_end_id = self._ids.get('im_end')
            
            cot_end_idx = msg.find(cot_end_id) if cot_end_id is not None else -1
            im_end_idx = msg.find_last(im_end_id) if im_end_id is not None else -1
            if im_end_idx >= 0:
                model_id = self._ids.get('model')
                fallback_idx = msg.find(model_id) if model_id is not None else 0
                start_idx = cot_end_idx if cot_end_idx >= 0 else fallback_idx
                if start_idx >= 0 and im_end_idx > start_idx:
                    msg.unmask_between(start_idx, im_end_idx, include_boundaries=(False, True))
        elif policy == 'fim':
            msg.mask_all()
            mid_id = self._ids.get('fim_mid')
            end_id = self._ids.get('im_end')
            if mid_id is None or end_id is None:
                return
            mid_idx = msg.find(mid_id)
            end_idx = msg.find_last(end_id)
            if mid_idx >= 0 and end_idx > mid_idx:
                msg.unmask_between(mid_idx, end_idx, include_boundaries=(True, True))
        else:
            raise ValueError(f'unknown mask policy: {policy!r}')

    def _image_patch_count(self, img: torch.Tensor) -> int:
        '''图像 -> 视觉塔的 patch token 数（(h // patch_size) * (w // patch_size)）。'''
        h, w = img.shape[-2], img.shape[-1]
        return (h // self.patch_size) * (w // self.patch_size)

    def _expand_image_patches(self, msg: Message) -> None:
        '''
        扫描 Token 序列，在 [image_start] / [image_end] 之间填入动态计算的 [image_patch] 数量，
        并确保插入的 Patch 的 ignore_mask 为 True (避免对图像特征计算语言模型 loss)

        支持两种模板形态：
            <|modality_image_start|><|modality_image_end|>                       (中间空隙插入)
            <|modality_image_start|><|modality_image_pad|><|modality_image_end|> (codon.j2，把单个 pad 展开)
        '''
        start_id = self._ids.get('image_start')
        end_id = self._ids.get('image_end')
        patch_id = self._ids.get('image_patch') or self._ids.get('pad')

        if start_id is None or end_id is None or patch_id is None:
            return

        new_ids = []
        new_mask = []
        img_idx = 0

        i = 0
        n = len(msg.ids)
        while i < n:
            # codon.j2：start + 单个 pad + end -> 把 pad 展开成 N 个
            if (msg.ids[i] == start_id and i + 2 < n
                    and msg.ids[i + 1] == patch_id and msg.ids[i + 2] == end_id):
                new_ids.append(start_id)
                new_mask.append(msg.ignore_mask[i])

                num_patches = 1
                if img_idx < len(msg.images):
                    num_patches = self._image_patch_count(msg.images[img_idx])
                    img_idx += 1

                new_ids.extend([patch_id] * num_patches)
                new_mask.extend([True] * num_patches)

                new_ids.append(end_id)
                new_mask.append(msg.ignore_mask[i + 2])
                i += 3
            # 旧模板：start 紧接 end -> 在中间插入 N 个
            elif msg.ids[i] == start_id and i + 1 < n and msg.ids[i+1] == end_id:
                new_ids.append(start_id)
                new_mask.append(msg.ignore_mask[i])

                if img_idx < len(msg.images):
                    num_patches = self._image_patch_count(msg.images[img_idx])

                    new_ids.extend([patch_id] * num_patches)
                    new_mask.extend([True] * num_patches)
                    img_idx += 1

                new_ids.append(end_id)
                new_mask.append(msg.ignore_mask[i+1])
                i += 2
            else:
                new_ids.append(msg.ids[i])
                new_mask.append(msg.ignore_mask[i])
                i += 1

        msg.ids = new_ids
        msg.ignore_mask = new_mask

    def _audio_token_count(self, mel: torch.Tensor) -> int:
        '''
        mel [num_mel_bins, frames] -> 音频塔输出的 token 数。

        与 `WhisperTinyAudioEncoder` 的下采样链一致：conv2 (kernel=3, stride=2, padding=1)
        把 T 帧压成 (T-1)//2 + 1，再按 `audio_pool_stride` 做一次非重叠平均池化
        （stride=8 时 30s / 3000 帧 -> 1500 -> 187）。
        '''
        frames = mel.shape[-1]
        encoder_len = (frames - 1) // 2 + 1
        return max(1, encoder_len // self.audio_pool_stride)

    def _expand_audio_patches(self, msg: Message) -> None:
        '''
        把模板渲染出的单个 <|modality_audio_pad|> 展开成 N 个占位符，与
        `_expand_image_patches` 对称（N 由 mel 长度与 `audio_pool_stride` 决定）。

        codon.j2 的音频形态是 <|modality_audio_start|><|modality_audio_pad|><|modality_audio_end|>
        三连 token；展开的占位符 ignore_mask 置 True，避免对音频特征计算语言模型 loss。
        '''
        start_id = self._ids.get('audio_start')
        end_id = self._ids.get('audio_end')
        patch_id = self._ids.get('audio_patch')

        if start_id is None or end_id is None or patch_id is None:
            return                       # A1 词表没有 audio_patch token，音频占位符不展开

        new_ids = []
        new_mask = []
        aud_idx = 0

        i = 0
        n = len(msg.ids)
        while i < n:
            if (msg.ids[i] == start_id and i + 2 < n
                    and msg.ids[i + 1] == patch_id and msg.ids[i + 2] == end_id):
                new_ids.append(start_id)
                new_mask.append(msg.ignore_mask[i])

                count = 1
                if aud_idx < len(msg.audios):
                    count = self._audio_token_count(msg.audios[aud_idx])
                    aud_idx += 1

                new_ids.extend([patch_id] * count)
                new_mask.extend([True] * count)

                new_ids.append(end_id)
                new_mask.append(msg.ignore_mask[i + 2])
                i += 3
            else:
                new_ids.append(msg.ids[i])
                new_mask.append(msg.ignore_mask[i])
                i += 1

        msg.ids = new_ids
        msg.ignore_mask = new_mask

    @staticmethod
    def _vector_rows(value: Any) -> int:
        '''
        连续值张量 -> 占位符行数（`[D]` / 标量算 1 行，`[T, D]` 取 T）。

        与图像按 (H//patch)*(W//patch) 数 patch、音频按 mel 帧数算 token 同理：
        占位符数量必须等于投影塔输出的行数。
        '''
        if not isinstance(value, torch.Tensor):
            return 1
        if value.dim() < 2:
            return 1
        return max(1, int(value.shape[0]))

    def _expand_pad_blocks(
        self,
        msg: Message,
        start_id: Optional[int],
        end_id: Optional[int],
        patch_id: Optional[int],
        counts: Sequence[int],
    ) -> int:
        '''
        把 `start + 单个 pad + end` 三连展开成 counts[i] 个 pad（第 i 个三连配第 i 个输入）。

        与 `_expand_image_patches` 的区别：只认三连形态（codon.j2 对连续值模态统一渲染三连），
        每个三连的 pad 个数由调用方按张量形状算好。展开出来的 pad 的 ignore_mask 置 True
        —— 它们是「被写入特征的位置」，不是要被预测的目标。

        Returns:
            int: 实际消耗掉的 counts 个数（三连比 counts 多时，多出来的按 1 个 pad 处理）。
        '''
        if start_id is None or end_id is None or patch_id is None:
            return 0

        new_ids: list[int] = []
        new_mask: list[bool] = []
        used = 0
        i, n = 0, len(msg.ids)
        while i < n:
            if (msg.ids[i] == start_id and i + 2 < n
                    and msg.ids[i + 1] == patch_id and msg.ids[i + 2] == end_id):
                count = max(0, int(counts[used])) if used < len(counts) else 1
                used += 1
                new_ids.append(start_id)
                new_mask.append(msg.ignore_mask[i])
                new_ids.extend([patch_id] * count)
                new_mask.extend([True] * count)
                new_ids.append(end_id)
                new_mask.append(msg.ignore_mask[i + 2])
                i += 3
            else:
                new_ids.append(msg.ids[i])
                new_mask.append(msg.ignore_mask[i])
                i += 1

        msg.ids = new_ids
        msg.ignore_mask = new_mask
        return used

    def _expand_vector_blocks(self, msg: Message, name: str, blocks: Sequence[Any]) -> None:
        '''按 `_PAD_BLOCK_MODALITIES[name]` 展开三连（action / proprio / tactile 走这里）。'''
        start_key, end_key, patch_key = _PAD_BLOCK_MODALITIES[name]
        counts = [self._vector_rows(item) for item in blocks]
        used = self._expand_pad_blocks(
            msg,
            self._ids.get(start_key),
            self._ids.get(end_key),
            self._ids.get(patch_key),
            counts,
        )
        if used != len(counts):
            raise ValueError(
                f'{name}: 模板渲染出 {used} 个占位块，但给了 {len(counts)} 个输入'
            )

    def _pop_gen_prompt(self) -> Optional[Message]:
        if self.messages and self.messages[-1].role == '__gen_prompt__':
            return self.messages.pop()
        return None
    
    def _append(self, msg: Message) -> None:
        gen = self._pop_gen_prompt()
        self.messages.append(msg)
        if gen is not None:
            self.messages.append(gen)

    @staticmethod
    def _action_steps(item: dict) -> list:
        '''
        codon.j2 里 `action_chunk` 的 step 归一：list / tuple 视为多个 step，
        其余（含张量 / numpy 数组这类 array-like）视为一个 block。

        必须与 `codon.j2` 的 `raw_steps` 分支保持一字不差，否则模板渲染出的三连数与张量数会对不上。
        '''
        raw = item['actions'] if 'actions' in item else item.get('action')
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)):
            return list(raw)
        return [raw]

    @staticmethod
    def _frame_image(frame: Any) -> Optional[torch.Tensor]:
        '''逐帧视频的帧载荷 -> 图像张量（帧可以是张量本身，或带 image / image_url 的 mapping）。'''
        if isinstance(frame, torch.Tensor):
            return frame
        if isinstance(frame, dict):
            for key in ('image', 'image_url', 'pixel_values'):
                value = frame.get(key)
                if isinstance(value, torch.Tensor):
                    return value
                if isinstance(value, dict) and isinstance(value.get('url'), torch.Tensor):
                    return value['url']
        return None

    def add_message(
        self, message: dict,
        mask: Optional[MaskPolicy] = None,
    ) -> 'Session':
        if 'tool_calls' in message.keys() and isinstance(message['tool_calls'], str):
            message['tool_calls'] = json.loads(message['tool_calls'])
        role = self._ROLE_ALIAS.get(message.get('role', 'user'), message.get('role', 'user'))

        images = []
        audios = []
        actions = []
        proprios = []
        tactiles = []
        content = message.get('content')
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get('type', '')).lower()
                if item_type == 'image':
                    if isinstance(item.get('image'), torch.Tensor):
                        images.append(item['image'])
                elif item_type == 'audio':
                    # mel 谱：'audio' 或 'mel' 键都接受
                    audio = item.get('audio', item.get('mel'))
                    if isinstance(audio, torch.Tensor):
                        audios.append(audio)
                elif item_type == 'video':
                    # 逐帧视频：帧按模板渲染顺序进 images，复用同一套图像占位符展开
                    for frame in item.get('frames') or []:
                        frame_image = self._frame_image(frame)
                        if frame_image is not None:
                            images.append(frame_image)
                elif item_type == 'action':
                    value = item.get('action')
                    if isinstance(value, torch.Tensor):
                        actions.append(value)
                elif item_type in ('action_chunk', 'actions', 'chunk'):
                    for step in self._action_steps(item):
                        if isinstance(step, torch.Tensor):
                            actions.append(step)
                elif item_type == 'proprio':
                    value = item.get('proprio', item.get('proprioception'))
                    if isinstance(value, torch.Tensor):
                        proprios.append(value)
                elif item_type in ('tactile', 'touch'):
                    value = item.get('tactile', item.get('touch'))
                    if isinstance(value, torch.Tensor):
                        tactiles.append(value)

        ids = self._encode_message(message)
        msg = Message(
            ids=ids, role=role,
            images=images, audios=audios,
            actions=actions, proprios=proprios, tactiles=tactiles,
        )
        self._apply_policy(msg, mask if mask is not None else self.policy.get(role, 'all'))

        if images: self._expand_image_patches(msg)
        if audios: self._expand_audio_patches(msg)
        if actions: self._expand_vector_blocks(msg, 'action', actions)
        if proprios: self._expand_vector_blocks(msg, 'proprio', proprios)
        if tactiles: self._expand_vector_blocks(msg, 'tactile', tactiles)

        self._append(msg)
        return self

    def add_messages(self, messages: list[dict]) -> 'Session':
        for m in messages:
            self.add_message(m)
        return self

    def _effort_key(self, effort: Optional[str] = None) -> str:
        '''档位名 -> token 键（'xhigh' -> 'effort_high'），非法档位直接报错。'''
        return f'effort_{resolve_effort(effort)}'

    def add_generation_prompt(
        self,
        enable_thinking: bool = False,
        disable_thinking: bool = False,
        effort: Optional[str] = None,
    ) -> 'Session':
        '''
        追加生成提示 <|im_start|><|model|>。

        Args:
            enable_thinking: 打开思考段，会在 <|thought_start|> 后接思考强度 token。
            disable_thinking: 直接给空思考段 <|thought_start|><|thought_end|>，不带强度 token。
            effort: 思考强度，'minimal' / 'low' / 'medium' / 'high' / 'xhigh' / 'max' / 'ultra'，
                收敛到三个 token：minimal|low -> <|effort_low|>，medium|high|xhigh -> <|effort_high|>，
                max|ultra -> <|effort_max|>；None 表示默认档位 EFFORT_DEFAULT。
                若词表里没有对应 token：显式指定的档位会报错，默认档位则静默省略（兼容旧词表）。
        '''
        if enable_thinking and disable_thinking:
            raise ValueError('enable_thinking and disable_thinking cannot both be True')
        self._pop_gen_prompt()
        parts = ['im_start', 'model']
        if enable_thinking:
            parts.append('cot_start')
            effort_key = self._effort_key(effort)
            if self._ids.get(effort_key) is not None:
                parts.append(effort_key)
            elif effort is not None:
                raise ValueError(
                    f'effort token {self._tokens.get(effort_key, effort_key)!r} not found in vocab'
                )
        elif disable_thinking:
            parts.extend(['cot_start', 'cot_end'])
        ids: list[int] = []
        for key in parts:
            tid = self._ids.get(key)
            if tid is None:
                raise ValueError(f'token {self._tokens.get(key, key)!r} not found in vocab')
            ids.append(tid)
        msg = Message(ids=ids, role='__gen_prompt__')
        msg.mask_all()
        self.messages.append(msg)
        return self

    def pop(self, index: int = -1) -> Message:
        return self.messages.pop(index)

    def clear(self) -> 'Session':
        self.messages.clear()
        return self

    def pad_to(
        self,
        length: int,
        side: Literal['left', 'right'] = 'right',
    ) -> 'Session':
        current_len = len(self)
        if current_len >= length:
            return self
        pad_id = self._ids.get('pad')
        if pad_id is None:
            raise ValueError('pad token not found in tokenizer')
        delta = length - current_len
        pad_msg = Message(
            ids=[pad_id] * delta,
            ignore_mask=[True] * delta,
            role='__padding__',
        )
        if side == 'right':
            self.messages.append(pad_msg)
        elif side == 'left':
            self.messages.insert(0, pad_msg)
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        return self

    @property
    def input_ids(self) -> list[int]:
        out: list[int] = []
        for m in self.messages:
            out.extend(m.ids)
        return out

    @property
    def ignore_mask(self) -> list[bool]:
        out: list[bool] = []
        for m in self.messages:
            out.extend(m.ignore_mask)
        return out

    @property
    def labels(self) -> list[int]:
        return [-100 if ig else tid for tid, ig in zip(self.input_ids, self.ignore_mask)]

    def to_tensors(
        self,
        device: Union[str, torch.device] = 'cpu',
        pad_to: Optional[int] = None,
        batch_dim: bool = False,
    ) -> dict[str, torch.Tensor]:
        ids = self.input_ids
        lbl = self.labels
        att = [1] * len(ids)

        patch_id = self._ids.get('image_patch')
        audio_patch_id = self._ids.get('audio_patch')

        if pad_to is not None and len(ids) < pad_to:
            pad_id_token = self._ids.get('pad') or 0
            delta = pad_to - len(ids)
            ids += [pad_id_token] * delta
            lbl += [-100] * delta
            att += [0] * delta

        t = lambda xs: torch.tensor(xs, dtype=torch.long, device=device)
        
        patch_indices = [idx for idx, tid in enumerate(ids) if patch_id is not None and tid == patch_id]
        audio_patch_indices = [idx for idx, tid in enumerate(ids) if audio_patch_id is not None and tid == audio_patch_id]

        def block_indices(name: str) -> list[int]:
            block_patch_id = self._ids.get(f'{name}_pad')
            if block_patch_id is None:
                return []
            return [idx for idx, tid in enumerate(ids) if tid == block_patch_id]

        action_patch_indices = block_indices('action')
        proprio_patch_indices = block_indices('proprio')
        tactile_patch_indices = block_indices('tactile')

        out = {
            'input_ids': t(ids), 
            'labels': t(lbl), 
            'attention_mask': t(att),
            'image_patch_indices': t(patch_indices),
            'audio_patch_indices': t(audio_patch_indices),
            'action_patch_indices': t(action_patch_indices),
            'proprio_patch_indices': t(proprio_patch_indices),
            'tactile_patch_indices': t(tactile_patch_indices),
        }

        all_images = []
        all_audios = []
        all_actions = []
        all_proprios = []
        all_tactiles = []
        for m in self.messages:
            all_images.extend(m.images)
            all_audios.extend(m.audios)
            all_actions.extend(m.actions)
            all_proprios.extend(m.proprios)
            all_tactiles.extend(m.tactiles)
        
        out['images'] = [img.to(device) for img in all_images]
        out['audios'] = [mel.to(device) for mel in all_audios]
        out['actions'] = [item.to(device) for item in all_actions]
        out['proprios'] = [item.to(device) for item in all_proprios]
        out['tactiles'] = [item.to(device) for item in all_tactiles]

        if batch_dim:
            out['input_ids'] = out['input_ids'].unsqueeze(0)
            out['labels'] = out['labels'].unsqueeze(0)
            out['attention_mask'] = out['attention_mask'].unsqueeze(0)
            out['image_patch_indices'] = out['image_patch_indices'].unsqueeze(0)
            out['audio_patch_indices'] = out['audio_patch_indices'].unsqueeze(0)
            out['action_patch_indices'] = out['action_patch_indices'].unsqueeze(0)
            out['proprio_patch_indices'] = out['proprio_patch_indices'].unsqueeze(0)
            out['tactile_patch_indices'] = out['tactile_patch_indices'].unsqueeze(0)

        return out

    def decode(self, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(self.input_ids, skip_special_tokens=skip_special_tokens)

    def __len__(self) -> int:
        return sum(len(m) for m in self.messages)

    def __iter__(self):
        return iter(self.messages)

    def __getitem__(self, idx: int) -> Message:
        return self.messages[idx]