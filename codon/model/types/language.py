from codon import *
from codon.model.cache import ModelCache
from codon.model.sampler import Sampler

import inspect


#: 音频能力允许声明的子类型（canonical 值）：
#:   'speech'  -> 说话声（人声 / 语音）
#:   'music'   -> 音乐
#:   'general' -> 通用（不区分内容，覆盖上面两类）
AUDIO_SUBTYPES: Tuple[str, ...] = ('speech', 'music', 'general')

#: 子类型别名 -> canonical 值；便于配置里直接写中文（说话声 / 音乐 / 通用）。
_AUDIO_SUBTYPE_ALIASES: Dict[str, str] = {
    'speech': 'speech', 'voice': 'speech', '说话声': 'speech', '语音': 'speech',
    'music': 'music', '音乐': 'music',
    'general': 'general', 'generic': 'general', '通用': 'general',
}


def normalize_audio_subtypes(subtypes: Union[str, Iterable[str], None]) -> Tuple[str, ...]:
    '''把声明的音频子类型规整成去重、按 `AUDIO_SUBTYPES` 顺序排列的 tuple。

    接受单个字符串（`'speech'`）或任意字符串可迭代（`['音乐', 'speech']`），
    `None` / 空集返回 `()`。声明中含 'general'（通用）时收敛为 `('general',)`，
    因为它已经覆盖其余子类型。未知子类型抛 ValueError。
    '''
    if subtypes is None:
        return ()
    if isinstance(subtypes, str):
        subtypes = (subtypes,)
    if not isinstance(subtypes, Iterable):
        raise TypeError(
            f'audio_subtypes must be a str or an iterable of str, '
            f'got {type(subtypes).__name__}'
        )

    resolved: List[str] = []
    for subtype in subtypes:
        if not isinstance(subtype, str):
            raise TypeError(
                f'audio subtype must be a str, got {type(subtype).__name__}'
            )
        key = subtype.strip().lower()
        if key not in _AUDIO_SUBTYPE_ALIASES:
            raise ValueError(
                f'unknown audio subtype {subtype!r}; '
                f'expected one of {AUDIO_SUBTYPES} (aliases: speech/说话声/语音, '
                f'music/音乐, general/通用)'
            )
        canonical = _AUDIO_SUBTYPE_ALIASES[key]
        if canonical not in resolved:
            resolved.append(canonical)

    if 'general' in resolved:
        return ('general',)
    return tuple(name for name in AUDIO_SUBTYPES if name in resolved)


@dataclass
class CausalLanguageModelOutput:
    '''
    Output of causal language model.

    Attributes:
        logits (torch.Tensor): Prediction logits.
        past_key_values (ModelCache, optional): The updated ModelCache container.
        aux_loss (torch.Tensor, optional): Auxiliary loss.
        attentions (list, optional): List of attention weights.
        hidden_states (tuple, optional): Tuple of hidden states.
    '''
    logits: torch.Tensor
    past_key_values: Optional[ModelCache] = None
    aux_loss: Optional[torch.Tensor] = None
    attentions: Optional[List[torch.Tensor]] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None


@dataclass
class ModelMeta:
    '''
    模型能力元信息（capability flags），用于对外声明这个 CausalLanguageModel 支持什么。

    默认全部 False：一个未声明的子类会被当成「纯文本、无思考、无工具、无音频」的基础模型，
    下游（服务层、chat 管线、评测）可以据此决定是否启用对应能力，而不是靠猜。

    Attributes:
        supports_image (bool): 是否支持图片输入（多模态）。
        supports_thinking (bool): 是否支持思考（CoT / reasoning 段落）。
        supports_tool (bool): 是否支持工具调用（function calling）。
        supports_audio (bool): 是否支持音频（多模态）。
        audio_subtypes (Tuple[str, ...]): 音频子类型，可选值见 `AUDIO_SUBTYPES`：
            'speech'（说话声）/ 'music'（音乐）/ 'general'（通用）。不开启音频时为 ()；
            开启音频但未指定子类型时按 ('general',) 处理；'general' 覆盖其余子类型。
    '''
    supports_image: bool = False
    supports_thinking: bool = False
    supports_tool: bool = False
    supports_audio: bool = False
    audio_subtypes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.audio_subtypes = normalize_audio_subtypes(self.audio_subtypes)
        if not self.supports_audio:
            # 「不支持音频」优先：未开启 supports_audio 时子类型一律清空，
            # 既不会出现自相矛盾的 meta，也让子类用 supports_audio = False 就能单独关掉音频。
            self.audio_subtypes = ()
        elif not self.audio_subtypes:
            # 只声明 supports_audio = True 时按「通用」处理。
            self.audio_subtypes = ('general',)

    def to_dict(self) -> Dict[str, Any]:
        '''展开为普通 dict，便于写入 API 响应。'''
        return {
            'supports_image': self.supports_image,
            'supports_thinking': self.supports_thinking,
            'supports_tool': self.supports_tool,
            'supports_audio': self.supports_audio,
            'audio_subtypes': list(self.audio_subtypes),
        }

    def supports_audio_subtype(self, subtype: str) -> bool:
        '''是否支持给定音频子类型：'speech' / 'music' / 'general'（也接受中文别名）。

        声明了 'general'（通用）即视为覆盖 'speech' / 'music'。
        '''
        if not self.supports_audio:
            return False
        resolved = normalize_audio_subtypes(subtype)
        if not resolved:
            return False
        return 'general' in self.audio_subtypes or resolved[0] in self.audio_subtypes


#: ModelMeta 里所有布尔能力开关：子类可用同名类属性声明（如 `supports_audio = True`）。
#: 由默认值类型推导，新增 bool 能力时无需再改 `_sync_meta`。
_BOOL_FIELDS: Tuple[str, ...] = tuple(
    name
    for name, field_info in ModelMeta.__dataclass_fields__.items()
    if isinstance(field_info.default, bool)
)


class CausalLanguageModel(BasicModel):
    '''
    Causal 语言模型基类，提供优化的自回归生成管线（generate / compute_perplexity）。

    ## 能力元信息（meta）

    子类通过类属性声明自身能力，默认全部 False：

        class MotifA1(CausalLanguageModel):
            supports_thinking = True          # 也可以整体覆写 meta = ModelMeta(...)

        class WhistleA1(CausalLanguageModel):
            supports_audio = True             # 音频能力（多模态）
            audio_subtypes = ('speech',)      # 可选子类型：说话声 / 音乐 / 通用

    也可在实例上读取：`model.meta` / `model.supports_thinking` / `model.supports('image')` /
    `model.supports_audio` / `model.audio_subtypes` / `model.supports_audio_subtype('music')`。

    ## 音频子类型（audio_subtypes）

    `audio_subtypes` 只在 `supports_audio = True` 时有意义，可写单个字符串或字符串集合，
    取值（含中文别名）为 'speech' / 说话声、'music' / 音乐、'general' / 通用：

        supports_audio = True                       # -> ('general',) 默认通用
        supports_audio = True; audio_subtypes = 'music'        # -> ('music',)
        supports_audio = True; audio_subtypes = ('speech', 'music')   # -> ('speech', 'music')
        supports_audio = False                      # -> ()，子类型被清空（关掉音频即忽略子类型）

    未声明 `supports_audio` 的子类 audio_subtypes 恒为 ()，'general' 覆盖其余子类型。

    ## forward 契约（子类必须遵守）

    `generate()` 会以关键字方式调用子类的 `forward`：

        outputs = self.forward(input_ids=..., start_pos=..., past_key_values=...)

    因此任何 CausalLanguageModel 子类的 `forward` **必须以 `input_ids` 为第一参数**，
    并接受 `start_pos` 与 `past_key_values` 关键字，返回 `CausalLanguageModelOutput`
    （含 `logits`，形状 [batch, seq, vocab]）。本类在子类定义时即校验该约定，
    签名不含 `input_ids`（或没有 `**kwargs` 兜底）的子类会在 import 时直接报错，
    而不是等到 generate 运行到一半才暴露。
    '''

    # 能力元信息默认值；子类可用 `supports_xxx = True` 或整体覆写 `meta` 覆盖。
    meta: ModelMeta = ModelMeta()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._sync_meta()
        cls._validate_forward_contract()

    @classmethod
    def _sync_meta(cls) -> None:
        '''把子类声明的能力标记（`supports_*` / `audio_subtypes`）合并进 `meta`。

        未覆写任何标记时直接复用父类实例（无额外开销）；一旦覆写就新建 ModelMeta，
        避免修改父类共享的可变对象。
        '''
        declared = getattr(cls, 'meta', None)
        if declared is not None and not isinstance(declared, ModelMeta):
            raise TypeError(
                f'{cls.__name__}.meta must be a ModelMeta, got {type(declared).__name__}'
            )

        overridden = {
            name: cls.__dict__[name]
            for name in ModelMeta.__dataclass_fields__
            if name in cls.__dict__
        }
        for name in _BOOL_FIELDS:
            if name in overridden and not isinstance(overridden[name], bool):
                raise TypeError(
                    f'{cls.__name__}.{name} must be a bool, got {type(overridden[name]).__name__}'
                )

        if overridden:
            base = declared if declared is not None else ModelMeta()
            values = {name: getattr(base, name) for name in ModelMeta.__dataclass_fields__}
            values.update(overridden)
            try:
                cls.meta = ModelMeta(**values)
            except (TypeError, ValueError) as exc:
                raise type(exc)(f'{cls.__name__}: {exc}') from exc
        elif declared is None:
            cls.meta = ModelMeta()

        # 让类属性与规范化后的 meta 保持一致（'music' -> ('music',)），
        # 这样实例上读 `model.audio_subtypes` 拿到的永远是规整结果。
        cls.audio_subtypes = cls.meta.audio_subtypes

    @classmethod
    def _validate_forward_contract(cls) -> None:
        forward = cls.__dict__.get('forward')
        if forward is None:
            return  # 允许中间抽象层暂不实现；真正实例化前必须补全
        try:
            params = inspect.signature(forward).parameters
        except (TypeError, ValueError):
            return  # C 扩展/不可检视的 callable，跳过静态校验
        names = set(params)
        has_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
        if 'input_ids' not in names and not has_kwargs:
            raise TypeError(
                f'{cls.__name__}.forward must accept `input_ids` as its first '
                f'parameter (CausalLanguageModel contract used by generate()). '
                f'Got signature: {inspect.signature(forward)}'
            )

    # ---- 能力查询 ----
    @property
    def supports_image(self) -> bool:
        '''是否支持图片输入。'''
        return self.meta.supports_image

    @property
    def supports_thinking(self) -> bool:
        '''是否支持思考（CoT / reasoning 段落）。'''
        return self.meta.supports_thinking

    @property
    def supports_tool(self) -> bool:
        '''是否支持工具调用。'''
        return self.meta.supports_tool

    @property
    def supports_audio(self) -> bool:
        '''是否支持音频（多模态）。'''
        return self.meta.supports_audio

    @property
    def audio_subtypes(self) -> Tuple[str, ...]:
        '''音频子类型：('speech',) / ('music',) / ('general',)；未开启音频时为 ()。'''
        return self.meta.audio_subtypes

    def supports(self, capability: str) -> bool:
        '''按名称查询能力：'image' / 'thinking' / 'tool' / 'audio'（也接受带 supports_ 前缀的写法）。'''
        name = capability[9:] if capability.startswith('supports_') else capability
        field_name = f'supports_{name}'
        if field_name not in ModelMeta.__dataclass_fields__:
            expected = ', '.join(
                repr(field[len('supports_'):]) for field in _BOOL_FIELDS
            )
            raise ValueError(
                f'unknown capability {capability!r}; expected one of {expected}'
            )
        return bool(getattr(self.meta, field_name))

    def supports_audio_subtype(self, subtype: str) -> bool:
        '''按名称查询音频子类型：'speech'（说话声）/ 'music'（音乐）/ 'general'（通用）。'''
        return self.meta.supports_audio_subtype(subtype)


    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        sampler: Optional[Sampler] = None,
        temperature: float = 0.7,
        eos_token_id: Optional[int] = None,
        past_key_values: Optional[ModelCache] = None
    ) -> torch.Tensor:
        '''
        Generate text tokens autoregressively using a prefill-decode pipeline.

        Args:
            input_ids (torch.Tensor): Input prompt token IDs with shape [batch, seq_len].
            max_new_tokens (int): Maximum number of new tokens to generate.
            sampler (Sampler, optional): Instance of Sampler. If None, default Sampler(0.7) is used.
            temperature (float): Sampling temperature for the default Sampler. Only used if `sampler` is None.
            eos_token_id (int, optional): End-of-sequence token ID.
            past_key_values (ModelCache, optional): Cache container to reuse states across decode steps.

        Returns:
            torch.Tensor: Generated token IDs with shape [batch, seq_len + num_generated].
        '''
        self.eval()
        if sampler is None:
            sampler = Sampler(temperature=temperature)

        generated = input_ids.clone()
        
        if past_key_values is None: past_key_values = ModelCache().to(self.device)

        with torch.no_grad():
            # 1. Prefill 
            outputs = self.forward(
                input_ids=input_ids,
                start_pos=0,
                past_key_values=past_key_values,
            )
            
            logits = outputs.logits[:, -1, :]
            next_token = sampler(logits)
            generated = torch.cat([generated, next_token], dim=-1)

            # 2. Decode
            for _ in range(max_new_tokens - 1):
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

                current_pos = past_key_values.seq_length

                outputs = self.forward(
                    input_ids=next_token,
                    start_pos=current_pos,
                    past_key_values=past_key_values,
                )

                logits = outputs.logits[:, -1, :]
                next_token = sampler(logits, input_ids=generated)
                generated = torch.cat([generated, next_token], dim=-1)

            return generated

    def compute_perplexity(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        '''
        Compute perplexity from logits and target tokens.

        Args:
            logits (torch.Tensor): Model output logits with shape [batch, seq_len, vocab_size].
            targets (torch.Tensor): Target token IDs with shape [batch, seq_len].

        Returns:
            torch.Tensor: Perplexity value (lower is better).
        '''
        batch_size, seq_len, vocab_size = logits.shape

        logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
        targets_flat = targets.reshape(batch_size * seq_len)

        loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')
        perplexity = torch.exp(loss)

        return perplexity