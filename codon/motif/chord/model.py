from codon import *

from codon.model.cache import ModelCache, build_cache
from codon.model.sampler import Sampler
from codon.model.types.language import CausalLanguageModel, CausalLanguageModelOutput
from codon.block.embedding import InterleavedRotaryEmbedding

from .config import MotifChordConfig
from .vision import MotifChordVisionProjector
from .audio import MotifChordAudioProjector
from .vector import MotifChordVectorProjector
from .decoder import HybridDecoder
from .utils import make_layer_pattern


class MotifChord(CausalLanguageModel):
    '''
    GDN + Gated GQA 混合主干，外挂 DINOv3 视觉塔与 Whisper 音频塔。

    ## 多模态：codon.j2 占位符约定

    `codon/res/codon.j2` 把模态渲染成序列内的占位符：

        <|modality_image_start|><|modality_image_pad|>...<|modality_image_end|>
        <|modality_audio_start|><|modality_audio_pad|>...<|modality_audio_end|>
        <|action_start|><|action_pad|>...<|action_end|>            （逐帧视频复用图像形态）
        <|proprio_start|><|proprio_pad|>...<|proprio_end|>
        <|tactile_start|><|tactile_pad|>...<|tactile_end|>

    `Session._expand_image_patches` / `_expand_audio_patches` / `_expand_vector_blocks`
    按图像尺寸、mel 长度、向量行数把中间的 pad 展开成 N 个占位符，
    `Session.to_tensors()` 给出它们在序列里的绝对位置（`image_patch_indices` /
    `audio_patch_indices` / `action_patch_indices` / `proprio_patch_indices` /
    `tactile_patch_indices`）。本模型的 forward 只做**就地写入**：

        x[b, image_patch_indices[b]] = 视觉特征
        x[b, audio_patch_indices[b]] = 音频特征
        x[b, action_patch_indices[b]] = 动作特征（MotifChordVectorProjector）

    序列长度不变，因此文本 token 的 RoPE 位置与不带模态时一致；decode 阶段
    （`past_key_values` 非空且不再传模态）复用 prefill 写好的模态位置。

    与旧接口的区别：不再接受 `pixel_values` / `mel` 前缀拼接参数 —— 那种写法会让
    模板渲染出的 pad token 仍走 `token_emb`、模态特征落在序列开头，与 codon.j2 的
    占位符语义错位。

    ## 与 ModelCache 的兼容性

    - 占位符是**就地写入**（序列长度不变），所以 `ModelCache.seq_length` 就是真实位置：
      decode 用 `start_pos = cache.seq_length` 续位置，不会因为模态而漂移，也不需要
      「减去模态长度」这类补偿。
    - 混合主干里 GDN 用 steps 计数（`GatedDeltaAttentionLayerCache._seq_len`）、
      GQA 用 KV 长度（`KVLayerCache.k.shape[2]`），两者在 prefill / decode 之后始终一致。
    - 模态只在 prefill 注入；传非空的 `past_key_values` 即表示增量 prefill（新图 / 新音频），
      `generate` 会从 cache 长度续上位置而不是把位置重置为 0。
    - decode 步不重传模态，也不会重跑视觉 / 音频塔：`forward(images=None, audios=None)`
      直接复用 prefill 写好的 key/value。
    '''

    supports_image = True
    supports_audio = True
    supports_thinking = True
    audio_subtypes = ('speech', 'music')

    def __init__(self, config: MotifChordConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.model_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.position_emb = InterleavedRotaryEmbedding(
            model_dim=config.model_dim // config.num_heads,
            num_axes=1,
        )

        layer_pattern = make_layer_pattern(config.num_layers)
        self.decoder = nn.ModuleList([
            HybridDecoder(
                config=config,
                layer_type=layer_pattern[idx],
                rope_emb=self.position_emb,
                idx=idx,
            )
            for idx in range(config.num_layers)
        ])

        self.norm = nn.RMSNorm(config.model_dim)
        self.proj_out = nn.Linear(config.model_dim, config.vocab_size, bias=False)

        if config.tie_weights:
            self.proj_out.weight = self.token_emb.weight

        self.vision = MotifChordVisionProjector(config)
        self.audio = MotifChordAudioProjector(config)
        # 连续向量模态：动作 / 本体感觉 / 触觉（无可冻结编码器，纯投影塔）
        self.action_proj = MotifChordVectorProjector(config, config.action_dim)
        self.proprio_proj = MotifChordVectorProjector(config, config.proprio_dim)
        self.tactile_proj = MotifChordVectorProjector(config, config.tactile_dim)

    # ---- 多模态占位符参数（Session 侧展开占位符时用，见 codon.utils.media.modal_config）----
    @property
    def image_patch_size(self) -> int:
        return int(self.config.vision_patch_size)

    @property
    def audio_pool_stride(self) -> int:
        return int(self.config.audio_pool_stride)

    @property
    def audio_num_mel_bins(self) -> int:
        return int(self.config.audio_num_mel_bins)

    # ---- 占位符工具（codon.j2 多模态约定）----

    @staticmethod
    def _scan_patch_indices(input_ids: torch.Tensor, patch_id: int) -> torch.Tensor:
        '''
        从 input_ids 里扫出占位符 token 的绝对位置。

        Returns:
            torch.Tensor: [B, P] 位置矩阵；某行占位符较少时用 -1 补齐
                （各行占位符数量允许不同，注入前会过滤掉 -1）。
        '''
        rows = [
            (input_ids[b] == patch_id).nonzero(as_tuple=False).flatten()
            for b in range(input_ids.shape[0])
        ]
        width = max((row.numel() for row in rows), default=0)
        indices = input_ids.new_full((len(rows), width), -1)
        for b, row in enumerate(rows):
            indices[b, :row.numel()] = row
        return indices

    @staticmethod
    def _per_batch_items(items, batch_size: int) -> Optional[list]:
        '''
        规整模态输入为「每 batch 一行」的嵌套列表。

        单个 Tensor / 扁平 list 按 batch 广播（每行同样的模态），嵌套 list 视为
        每行各自的模态。None 或空列表返回 None（表示没有模态输入）。
        '''
        if items is None:
            return None
        if isinstance(items, torch.Tensor):
            items = [items]
        items = list(items)
        if not items:
            return None
        if isinstance(items[0], (list, tuple)):
            per_batch = [list(group) for group in items]
        else:
            per_batch = [list(items) for _ in range(batch_size)]

        if len(per_batch) != batch_size:
            raise ValueError(
                f'modality batch size {len(per_batch)} != input_ids batch size {batch_size}'
            )
        return per_batch

    def _inject_modality(
        self,
        x: torch.Tensor,
        projector: BasicModel,
        items,
        indices: Optional[torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        '''
        把模态塔的输出写进占位符位置：`x[b, indices[b]] = 特征`。

        占位符数量与特征行数必须相等，不做静默截断（数量不匹配直接报错，
        因为静默截断会留下未被替换的 pad token，训练时表现为莫名其妙的 CE loss）。
        '''
        per_batch = self._per_batch_items(items, x.shape[0])
        if per_batch is None:
            if indices is not None and bool((indices >= 0).any()):
                raise ValueError(
                    f'{name}: 序列里有 {name} 占位符，但没有提供 {name}s'
                )
            return x
        if indices is None:
            raise ValueError(
                f'{name}: 需要同时提供 {name}s 与 {name}_patch_indices（或 {name}_patch_id）'
            )

        for b in range(x.shape[0]):
            idx = indices[b].to(device=x.device, dtype=torch.long)
            idx = idx[idx >= 0]
            group = per_batch[b]

            if idx.numel() == 0:
                if group:
                    raise ValueError(
                        f'{name}: batch {b} 给了 {len(group)} 个输入，但序列里没有对应占位符'
                    )
                continue

            feats = [projector(item.unsqueeze(0)).squeeze(0) for item in group]
            flat = torch.cat(feats, dim=0) if feats else x.new_zeros(0, x.shape[-1])

            if flat.shape[0] != idx.numel():
                raise ValueError(
                    f'{name}: batch {b} 占位符数量 {idx.numel()} != 特征行数 {flat.shape[0]}'
                )
            if int(idx.max()) >= x.shape[1]:
                raise ValueError(
                    f'{name}: batch {b} 占位符索引越界 (max {int(idx.max())} >= seq_len {x.shape[1]})'
                )

            x[b, idx] = flat.to(x.dtype)

        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[List[torch.Tensor]] = None,
        image_patch_indices: Optional[torch.Tensor] = None,
        image_patch_id: Optional[int] = None,
        audios: Optional[List[torch.Tensor]] = None,
        audio_patch_indices: Optional[torch.Tensor] = None,
        audio_patch_id: Optional[int] = None,
        actions: Optional[List[torch.Tensor]] = None,
        action_patch_indices: Optional[torch.Tensor] = None,
        action_patch_id: Optional[int] = None,
        proprios: Optional[List[torch.Tensor]] = None,
        proprio_patch_indices: Optional[torch.Tensor] = None,
        proprio_patch_id: Optional[int] = None,
        tactiles: Optional[List[torch.Tensor]] = None,
        tactile_patch_indices: Optional[torch.Tensor] = None,
        tactile_patch_id: Optional[int] = None,
        mask: torch.Tensor = None,
        start_pos: Union[int, torch.Tensor] = 0,
        past_key_values: Optional[ModelCache] = None,
        output_attentions: bool = False,
    ) -> CausalLanguageModelOutput:
        '''
        Args:
            input_ids: [B, L] Session + codon.j2 渲染出的 token 序列（含模态占位符）。
            images: 图像张量列表，或每 batch 一行的嵌套列表。只在 prefill 阶段需要。
            image_patch_indices: [B, P] 图像占位符的绝对位置（`Session.to_tensors()` 直接给出）。
            image_patch_id: 图像占位符 token id；给了它就不必再传 indices，forward 自己扫描。
            audios: mel 谱 [num_mel_bins, T] 列表，形态与 images 一致。
            audio_patch_indices: [B, P] 音频占位符的绝对位置。
            audio_patch_id: 音频占位符 token id，同上。
            actions: 动作块 `[T, action_dim]`（或 `[action_dim]`）列表，形态与 images 一致；
                第 i 个块对应序列里第 i 个 `<|action_start|><|action_pad|><|action_end|>` 三连，
                占用 T 个 pad 位置。
            action_patch_indices / action_patch_id: 动作占位符位置 / token id，同上。
            proprios / proprio_patch_indices / proprio_patch_id: 本体感觉，约定与 action 相同。
            tactiles / tactile_patch_indices / tactile_patch_id: 触觉读数，约定与 action 相同。
            mask: attention mask（0/1，0 表示 padding）。注意 GDN 层不消费 attention_mask，
                带 padding 的 batch 请用 pack 过的序列。
            start_pos: 起始位置；有 cache 时等于 cache 已有的序列长度。
            past_key_values: 增量解码缓存。
            output_attentions: 是否返回每层注意力权重。
        '''
        if image_patch_indices is None and image_patch_id is not None:
            image_patch_indices = self._scan_patch_indices(input_ids, image_patch_id)
        if audio_patch_indices is None and audio_patch_id is not None:
            audio_patch_indices = self._scan_patch_indices(input_ids, audio_patch_id)

        x = self.token_emb(input_ids)

        if images is not None or image_patch_indices is not None:
            x = self._inject_modality(x, self.vision, images, image_patch_indices, 'image')
        if audios is not None or audio_patch_indices is not None:
            x = self._inject_modality(x, self.audio, audios, audio_patch_indices, 'audio')

        for name, projector, items, indices, patch_id in (
            ('action', self.action_proj, actions, action_patch_indices, action_patch_id),
            ('proprio', self.proprio_proj, proprios, proprio_patch_indices, proprio_patch_id),
            ('tactile', self.tactile_proj, tactiles, tactile_patch_indices, tactile_patch_id),
        ):
            if indices is None and patch_id is not None:
                indices = self._scan_patch_indices(input_ids, patch_id)
            if items is not None or indices is not None:
                x = self._inject_modality(x, projector, items, indices, name)

        # 模态特征与 token 嵌入一起过 dropout（注入之后才 dropout，避免模态特征被漏掉）
        x = self.dropout(x)

        all_attentions = [] if output_attentions else None

        for i, layer in enumerate(self.decoder):
            layer_past = None
            if isinstance(past_key_values, ModelCache):
                if past_key_values[i] is None:
                    past_key_values[i] = build_cache(layer.attn)
                layer_past = past_key_values[i]

            out = layer(
                hidden_states=x,
                attention_mask=mask,
                output_attentions=output_attentions,
                position_emb=self.position_emb,
                embedding_start=start_pos,
                past_key_value=layer_past,
            )
            x = out.output

            if output_attentions:
                all_attentions.append(out.attention_weights)

        x = self.norm(x)
        logits = self.proj_out(x)

        return CausalLanguageModelOutput(
            logits=logits,
            past_key_values=past_key_values,
            aux_loss=None,
            attentions=all_attentions,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        images: Optional[List[torch.Tensor]] = None,
        image_patch_indices: Optional[torch.Tensor] = None,
        image_patch_id: Optional[int] = None,
        audios: Optional[List[torch.Tensor]] = None,
        audio_patch_indices: Optional[torch.Tensor] = None,
        audio_patch_id: Optional[int] = None,
        actions: Optional[List[torch.Tensor]] = None,
        action_patch_indices: Optional[torch.Tensor] = None,
        action_patch_id: Optional[int] = None,
        proprios: Optional[List[torch.Tensor]] = None,
        proprio_patch_indices: Optional[torch.Tensor] = None,
        proprio_patch_id: Optional[int] = None,
        tactiles: Optional[List[torch.Tensor]] = None,
        tactile_patch_indices: Optional[torch.Tensor] = None,
        tactile_patch_id: Optional[int] = None,
        mask: torch.Tensor = None,
        max_new_tokens: int = 100,
        sampler: Optional[Sampler] = None,
        temperature: float = 0.7,
        eos_token_id: Optional[int] = None,
        past_key_values: Optional[ModelCache] = None,
        constraint=None,
    ) -> torch.Tensor:
        '''
        多模态自回归生成：模态只在 prefill 注入，decode 只喂新 token
        （`start_pos` 取 `past_key_values.seq_length`）。

        sampler / temperature / eos_token_id / constraint 的语义与
        `CausalLanguageModel.generate` 完全一致。

        传入非空的 `past_key_values` 即表示「续写」：prefill 从 cache 已有长度续上位置，
        因此可以在多轮对话里对新的图片 / 音频增量 prefill（此时 `mask` 传对应新 chunk 的，
        或直接 None —— 历史 token 对增量段都是可见的）。
        '''
        self.eval()
        if sampler is None:
            sampler = Sampler(temperature=temperature)
        if constraint is not None:
            sampler = sampler.with_constraint(constraint, eos_token_id=eos_token_id)
            if eos_token_id is None:
                eos_token_id = constraint.eos_token_id

        generated = input_ids.clone()

        if past_key_values is None:
            past_key_values = ModelCache().to(self.device)

        # 续写兼容：调用方传了非空 cache 时从 cache 已有长度续上位置（多图 / 多轮增量 prefill），
        # 空 cache 就是 0，与 `CausalLanguageModel.generate` 的行为一致。
        prefill_start = past_key_values.seq_length

        with torch.no_grad():
            # 1. Prefill（模态在这里注入）
            outputs = self.forward(
                input_ids=input_ids,
                images=images,
                image_patch_indices=image_patch_indices,
                image_patch_id=image_patch_id,
                audios=audios,
                audio_patch_indices=audio_patch_indices,
                audio_patch_id=audio_patch_id,
                actions=actions,
                action_patch_indices=action_patch_indices,
                action_patch_id=action_patch_id,
                proprios=proprios,
                proprio_patch_indices=proprio_patch_indices,
                proprio_patch_id=proprio_patch_id,
                tactiles=tactiles,
                tactile_patch_indices=tactile_patch_indices,
                tactile_patch_id=tactile_patch_id,
                mask=mask,
                start_pos=prefill_start,
                past_key_values=past_key_values,
            )

            logits = outputs.logits[:, -1, :]
            next_token = sampler(logits)
            generated = torch.cat([generated, next_token], dim=-1)

            # 2. Decode（不再传模态，位置从 cache 长度续上）
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
