from codon import *
from codon.config import configclass, field
from codon.block.attention import GatedDeltaAttention, MultiHeadAttention
from codon.block.embedding import InterleavedFourierRotaryEmbedding, RotaryEmbedding
from codon.block import MLP
from codon.impl import MobileNetV3
from codon.model.cache import (
    BasicLayerCache,
    GatedDeltaAttentionLayerCache,
    ModelCache,
    build_cache,
)
from codon.model.types.language import CausalLanguageModel, CausalLanguageModelOutput


class Block(BasicModel):
    def __init__(
        self,
        model_dim: int = 768,
        num_heads: Optional[int] = None,
        attn_type: Literal['mha', 'gdn'] = 'gdn'
    ):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads or (model_dim // 64)
        self.attn_type = attn_type.lower()

        self.norm1 = nn.RMSNorm(model_dim)
        if self.attn_type == 'gdn':
            self.attn = GatedDeltaAttention(hidden_size=model_dim, num_heads=self.num_heads)
        elif self.attn_type == 'mha':
            self.attn = MultiHeadAttention(hidden_size=model_dim, num_heads=self.num_heads, dropout=0.0, bias=False)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type!r}. Use 'mha' or 'gdn'.")
        self.mlp = MLP.SwiGLU(model_dim)
        self.norm2 = nn.RMSNorm(model_dim)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            output_attentions=False,
            position_emb=None,
            embedding_start=0,
            embedding_pos=None,
            past_key_value=None
        ):
        '''
        Returns:
            (hidden_states, past_key_value, payload)

            payload 仅在 past_key_value 为 None 且注意力为循环式机制（如 GDN）时非空，
            用于让上层决定是否把「全序列前向算出的状态」转成可复用的层缓存。
        '''
        x = self.norm1(hidden_states)
        attn_out = self.attn(
            hidden_states=x,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            position_emb=position_emb,
            embedding_start=embedding_start,
            embedding_pos=embedding_pos,
            past_key_value=past_key_value,
        )
        hidden_states = hidden_states + attn_out.output

        x = self.norm2(hidden_states)
        hidden_states = hidden_states + self.mlp(x)

        return hidden_states, attn_out.past_key_value, attn_out.payload


@configclass
class LoopedConfig:
    model_dim: int = 768
    num_heads: int = 12
    recurrence: int = 2
    prelude_attn_types: List[Literal['mha', 'gdn']] = field(default=['gdn'])
    body_attn_types: List[Literal['mha', 'gdn']] = field(default=['gdn', 'gdn'])
    coda_attn_types: List[Literal['mha', 'gdn']] = field(default=['gdn'])
    use_residual_scaling: bool = True
    dropout: float = 0.0
    max_len: int = 131072
    rope_base: int = 500000
    # MHA 的位置编码：'fourier_interleaved' = InterleavedFourierRotaryEmbedding(num_axes=2)
    # （FoPE + 交错多轴），'rope' = 普通 RotaryEmbedding。
    pos_emb_type: Literal['fourier_interleaved', 'rope'] = 'fourier_interleaved'
    pos_num_axes: int = 2
    fope_sigma: Optional[float] = None          # None -> 由 num_params 拟合（无则 0.3）
    fope_train_len_per_axis: Optional[List[int]] = None


class Looped(BasicModel):
    '''
    循环深度模型（prelude → body × recurrence → coda），支持可选的推理缓存。

    ## Cache 契约

    `past_key_values` 为 `codon.model.cache.ModelCache`，按「block 槽位」索引：

        prelude: 0 .. P-1
        body:    P .. P+B-1
        coda:    P+B .. P+B+C-1

    槽位在整个 `forward` 调用中保持不变，因此 `ModelCache.seq_length` 可跨 decode 步
    单调增长，`generate()` / `chat()` 等推理管线可直接使用。

    ## 为什么要求 recurrence == 1

    `recurrence > 1` 时 body 必须对**整段序列**重复前向 R 次，而第 r 轮的注意力
    KV / 递归状态会覆盖第 r-1 轮的状态：GatedDeltaNet 的 delta-rule 是
    S ← S·exp(g) + kᵀ(β(v − kᵀS))，第二轮读到的 S 已不是第一轮的 S。
    缓存只保存最终状态，无法在 decode 时重放「同一 token 被 body 处理 R 次」，
    因此增量解码不可能与一次性前向等价（实测偏差 ~0.1–0.8，而输出尺度仅 ~4）。

    与其返回一个静默错误的结果，这里在 `recurrence > 1` 且传入缓存时直接报错。
    需要缓存推理时把 `recurrence` 设为 1（此时 prelude/body/coda 各跑一次，
    等价于标准堆叠 Transformer，缓存可精确复现）。
    '''

    def __init__(self, config: LoopedConfig):
        super().__init__()
        self.config = config

        self.prelude = nn.ModuleList([
            Block(config.model_dim, num_heads=config.num_heads, attn_type=t)
            for t in config.prelude_attn_types
        ])
        self.body = nn.ModuleList([
            Block(config.model_dim, num_heads=config.num_heads, attn_type=t)
            for t in config.body_attn_types
        ])
        self.coda = nn.ModuleList([
            Block(config.model_dim, num_heads=config.num_heads, attn_type=t)
            for t in config.coda_attn_types
        ])

    # ---- cache 槽位 ----
    @property
    def _body_offset(self) -> int:
        return len(self.prelude)

    @property
    def _coda_offset(self) -> int:
        return len(self.prelude) + len(self.body)

    def create_cache(self, device: Optional[torch.device] = None) -> ModelCache:
        '''构造与当前模型结构匹配的空 ModelCache（不预先分配层缓存，按需在 forward 中创建）。'''
        if self.config.recurrence != 1:
            raise ValueError(
                f'Looped 仅在 recurrence == 1 时支持增量缓存，当前 recurrence={self.config.recurrence}。'
                f'body 需要重复前向时缓存无法重放每轮状态，请改用 recurrence=1。'
            )
        return ModelCache().to(device if device is not None else self.device)

    def _check_cacheable(self, past_key_values) -> None:
        if isinstance(past_key_values, ModelCache) and self.config.recurrence != 1:
            raise ValueError(
                f'Looped 仅在 recurrence == 1 时支持 past_key_values，当前 recurrence={self.config.recurrence}。'
                f'循环前向的每轮状态会互相覆盖，缓存无法在 decode 时重放，'
                f'增量解码将与一次性前向不一致。'
            )

    @staticmethod
    def _payload_to_cache(cache: BasicLayerCache, payload: dict) -> BasicLayerCache:
        '''把「全序列前向」返回的循环状态 payload 落到层缓存上，供后续 decode 复用。'''
        if isinstance(cache, GatedDeltaAttentionLayerCache):
            # payload 只含状态本身，不含 token 数；seq_length 由后续 update 的 steps 推进。
            cache.update(payload['state'], payload['conv_state'])
            return cache
        raise TypeError(f'无法把 payload 转为缓存: 不支持的缓存类型 {type(cache).__name__}')

    def _run(
        self,
        x,
        blocks,
        layer_offset,
        past_key_values,
        attention_mask,
        output_attentions,
        position_emb,
        embedding_start,
        embedding_pos,
    ):
        '''跑一组 block；`past_key_values` 非 None 时逐槽位建/取缓存并回写。'''
        for i, block in enumerate(blocks):
            layer_past = None
            if isinstance(past_key_values, ModelCache):
                idx = layer_offset + i
                if past_key_values[idx] is None:
                    past_key_values[idx] = build_cache(block.attn)
                layer_past = past_key_values[idx]

            x, layer_kv, payload = block(
                x,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                position_emb=position_emb,
                embedding_start=embedding_start,
                embedding_pos=embedding_pos,
                past_key_value=layer_past,
            )

            if isinstance(past_key_values, ModelCache):
                if layer_kv is not None:
                    past_key_values[idx] = layer_kv
                elif payload is not None:
                    past_key_values[idx] = self._payload_to_cache(past_key_values[idx], payload)

        return x

    def forward(
        self,
        x,
        attention_mask=None,
        output_attentions=False,
        position_emb=None,
        embedding_start=0,
        embedding_pos=None,
        past_key_values: Optional[ModelCache] = None,
        return_all_loop_outputs=False,
    ):
        self._check_cacheable(past_key_values)

        x = self._run(
            x, self.prelude, 0, past_key_values,
            attention_mask, output_attentions, position_emb, embedding_start, embedding_pos,
        )

        loop_outputs = [] if return_all_loop_outputs else None

        for r in range(self.config.recurrence):
            x_before_body = x

            x = self._run(
                x, self.body, self._body_offset, past_key_values,
                attention_mask, output_attentions, position_emb, embedding_start, embedding_pos,
            )

            if self.config.use_residual_scaling:
                x = x_before_body + (x - x_before_body) / self.config.recurrence

            if return_all_loop_outputs:
                loop_outputs.append(x)

        x = self._run(
            x, self.coda, self._coda_offset, past_key_values,
            attention_mask, output_attentions, position_emb, embedding_start, embedding_pos,
        )

        if return_all_loop_outputs:
            return x, loop_outputs

        return x


class LoopedLM(CausalLanguageModel):
    '''
    把 `Looped` 包成完整的 CausalLanguageModel：吃 input_ids、出 logits，
    因此可以直接使用 `codon.utils.generate.generate()` / `chat()`。

    与 `MotifA1` 一致：`past_key_values` 由外部传入并原地更新，`start_pos` 用于位置偏移。

    位置编码（仅作用于 attn_type='mha' 的 block，GDN 走递归状态、不吃位置编码）：
    `config.pos_emb_type='fourier_interleaved'` 时使用
    `InterleavedFourierRotaryEmbedding(model_dim=head_dim, num_axes=2)`（FoPE + 交错双轴）。
    该模块要求显式传入 positions，因此这里构造二维位置 [t, t]（一维序列在两条轴上的
    对角映射），并把 positions 通过 `embedding_pos` 透传到注意力层；
    此时 `embedding_start` 传 0，避免模块内部再叠加一次偏移。
    '''

    def __init__(self, config: LoopedConfig, vocab_size: int = 2 ** 13, tie_weights: bool = True):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        head_dim = config.model_dim // config.num_heads
        if config.pos_emb_type == 'fourier_interleaved':
            if head_dim % config.pos_num_axes != 0:
                raise ValueError(
                    f'head_dim={head_dim} 必须能被 pos_num_axes={config.pos_num_axes} 整除'
                )
            self.position_emb = InterleavedFourierRotaryEmbedding(
                model_dim=head_dim,
                max_len=config.max_len,
                base=config.rope_base,
                num_axes=config.pos_num_axes,
                sigma=config.fope_sigma,
                train_len_per_axis=config.fope_train_len_per_axis,
            )
        elif config.pos_emb_type == 'rope':
            self.position_emb = RotaryEmbedding(
                head_dim, max_len=config.max_len, base=config.rope_base
            )
        else:
            raise ValueError(f"Unknown pos_emb_type: {config.pos_emb_type!r}")

        self.token_emb = nn.Embedding(vocab_size, config.model_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.looped = Looped(config)
        self.norm = nn.RMSNorm(config.model_dim)
        self.proj_out = nn.Linear(config.model_dim, vocab_size, bias=False)

        if tie_weights:
            self.proj_out.weight = self.token_emb.weight

        self.apply(self._init_weights)

    @property
    def _needs_positions(self) -> bool:
        '''交错多轴编码要求显式 positions（num_axes > 1 时模块自身无法自动生成）。'''
        return getattr(self.position_emb, 'num_axes', 1) > 1

    def _build_positions(self, x, start_pos, seq_len):
        '''构造 [B, L, num_axes] 的绝对位置；一维序列映射到各轴的相同取值。'''
        return self.build_positions(x.shape[0], seq_len, start_pos, device=x.device)

    @staticmethod
    def _init_weights(module):
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                torch.nn.init.zeros_(module.weight[module.padding_idx])

    def create_cache(self, device: Optional[torch.device] = None) -> ModelCache:
        return self.looped.create_cache(device)

    def build_positions(self, batch_size: int, seq_len: int, start_pos, device=None) -> torch.Tensor:
        '''公开的位置构造入口（供多模态侧按需覆盖部分位置），返回 [B, L, num_axes]。'''
        if isinstance(start_pos, torch.Tensor):
            start_pos = int(start_pos.reshape(-1)[0].item())
        idx = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=torch.long)
        num_axes = getattr(self.position_emb, 'num_axes', 1)
        return idx.view(1, seq_len, 1).expand(batch_size, seq_len, num_axes)

    def embed_with_positions(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        start_pos: Union[int, torch.Tensor] = 0,
    ):
        '''
        返回 `(嵌入, 实际使用的位置张量)`。

        多轴编码（InterleavedFourierRotaryEmbedding）必须显式传 positions，所以这里
        把「按需构造」的逻辑集中在一处，调用方拿到返回的 positions 再透传给注意力层，
        避免 forward 与多模态路径各自构造一遍、又不小心传成 None。
        '''
        x = self.dropout(self.token_emb(input_ids))
        if positions is None and self._needs_positions:
            positions = self.build_positions(
                x.shape[0], input_ids.shape[1], start_pos, device=x.device
            )
        return x, positions

    def embed(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        start_pos: Union[int, torch.Tensor] = 0,
    ) -> torch.Tensor:
        '''
        只做 token 嵌入 + dropout，返回 [B, L, model_dim]。

        多模态模型用它在序列中替换图像占位符位置，再调用 `looped` 跑主干；
        需要位置张量时用 `embed_with_positions()`。
        '''
        return self.embed_with_positions(input_ids, positions=positions, start_pos=start_pos)[0]

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor = None,
        start_pos: Union[int, torch.Tensor] = 0,
        past_key_values: Optional[ModelCache] = None,
        output_attentions: bool = False,
        return_all_loop_outputs: bool = False,
        positions: Optional[torch.Tensor] = None,
    ) -> CausalLanguageModelOutput:
        x, positions = self.embed_with_positions(input_ids, positions=positions, start_pos=start_pos)

        embedding_start = 0 if positions is not None else start_pos

        x = self.looped(
            x,
            attention_mask=mask,
            output_attentions=output_attentions,
            position_emb=self.position_emb,
            embedding_start=embedding_start,
            embedding_pos=positions,
            past_key_values=past_key_values,
            return_all_loop_outputs=return_all_loop_outputs,
        )

        if return_all_loop_outputs:
            x, loop_outputs = x
        else:
            loop_outputs = None

        x = self.norm(x)
        logits = self.proj_out(x)

        return CausalLanguageModelOutput(
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=tuple(loop_outputs) if loop_outputs is not None else None,
        )


class LoopedVLM(CausalLanguageModel):
    '''
    LoopedLM + MobileNetV3 的多模态封装。

    结构与 test_mbv3.py / vl.py 的思路一致，但视觉特征直接来自 MobileNetV3 的空间特征图：

        MobileNetV3(return_features=True)(image)  ->  [B, C, gh, gw]
        展平 + 转置                               ->  [B, gh*gw, C]
        投影                                      ->  [B, gh*gw, model_dim]
        写入 input_ids 中占位符位置                ->  x[b, patch_indices] = 视觉特征

    位置编码：MobileNetV3 的输出是二维网格，因此图像 patch 用真实的 (row, col)
    作为二维位置，文本 token 用 (t, t)（与 LoopedLM 一维序列的约定一致）。
    patch 数必须等于占位符数量，否则报错（不做静默截断）。

    推理缓存：图像只在 prefill 阶段注入，decode 阶段 `images=None` 即可；
    KV cache 复用 prefill 时算好的图像位置。
    '''

    supports_image = True

    def __init__(
        self,
        config: LoopedConfig,
        vocab_size: int = 2 ** 13,
        backend: str = 'small',
        image_size: int = 224,
        vision_pretrained: Optional[str] = None,
        freeze_vision: bool = True,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.config = config
        self.image_size = image_size

        self.language = LoopedLM(config, vocab_size=vocab_size, tie_weights=tie_weights)
        self.vision = MobileNetV3(backend=backend, return_features=True)

        vision_dim = self.vision.conv2.out_channels          # small=576, large=960
        self.vision_proj = nn.Linear(vision_dim, config.model_dim)
        self.norm = self.language.norm                        # 复用 LM 的输出 norm
        self.proj_out = self.language.proj_out

        if vision_pretrained is not None:
            self.vision.load_pretrained(vision_pretrained)

        if freeze_vision:
            # 视觉塔默认冻结：特征在前向里 no_grad 计算，训练只更新 vision_proj + 语言塔，
            # 显存与算力开销大幅下降（8G 卡上的关键一项）。
            self.vision.requires_grad_(False)
            self.vision.eval()

    # ---- 视觉侧 ----
    @torch.no_grad()
    def encode_image(self, image: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        '''[B, C, H, W] -> [B, gh*gw, model_dim]，返回二维网格顺序的 patch 特征。

        Args:
            dtype: 视觉塔的执行 dtype（默认取视觉塔自身 dtype）。
                   视觉塔是 `no_grad` 的，不受 autocast 保护，训练时（外层 autocast 会把
                   conv 权重按 autocast dtype 转换）必须把权重与输入显式对齐，
                   否则报 "Input type (torch.FloatTensor) and weight type (CUDABFloat16Type)"。
        '''
        vision = self.vision
        target = dtype or next(vision.parameters()).dtype
        # 视觉塔必须与输入同 device；dtype 也要对齐（no_grad 区不受 autocast 保护）
        if next(vision.parameters()).dtype != target or next(vision.parameters()).device != image.device:
            # device 必须一起给：只给 dtype 会把参数搬到 CPU
            vision = vision.to(device=image.device, dtype=target)

        feats = vision(image.to(target))                      # [B, C, gh, gw]
        b, c, gh, gw = feats.shape
        feats = feats.flatten(2).transpose(1, 2)              # [B, gh*gw, C]
        # 投影层要跟视觉特征同 device/dtype（它在 no_grad 区里，同样不受 autocast 管理）
        return self.vision_proj.to(device=feats.device, dtype=feats.dtype)(feats)

    def _build_vlm_positions(
        self,
        batch_size: int,
        seq_len: int,
        patch_indices: torch.Tensor,
        grids: Optional[List[Tuple[int, int]]],
        patch_counts: Optional[List[Optional[int]]],
        start_pos,
        device,
    ) -> torch.Tensor:
        '''文本位置 (t, t)，图像 patch 位置 (row, col)。

        Args:
            grids: 一个样本内每张图的 (gh, gw)；用于生成二维位置模板。
            patch_counts: 每个 batch 行实际填入的 patch 总数（无图的行是 None）。
        '''
        pos = self.language.build_positions(batch_size, seq_len, start_pos, device=device)

        if not grids:
            return pos

        base = int(start_pos) if not isinstance(start_pos, torch.Tensor) else int(start_pos.reshape(-1)[0])
        patch_pos = torch.cat([
            torch.tensor(
                [(base + (k // gw), base + (k % gw)) for k in range(gh * gw)],
                device=device, dtype=torch.long,
            )
            for gh, gw in grids
        ], dim=0)                                             # [P_total, 2]，行主序
        template_len = patch_pos.shape[0]

        pos = pos.clone()
        seq_len = pos.shape[1]
        for b in range(batch_size):
            idx = patch_indices[b]
            idx = idx[idx >= 0]
            if idx.numel() == 0:
                continue
            expected = patch_counts[b] if patch_counts else template_len
            if expected is None or idx.numel() != expected:
                raise ValueError(
                    f'batch {b}: 占位符数量 {idx.numel()} != 视觉特征行数 {expected} '
                    f'(grids {grids})'
                )
            if int(idx.max()) >= seq_len:
                raise ValueError(
                    f'batch {b}: 占位符索引越界 (max {int(idx.max())} >= seq_len {seq_len})'
                )
            # 模板按单样本的图数生成；若该行图数不同，重复/截取以对齐
            reps = (idx.numel() + template_len - 1) // template_len
            pos[b, idx] = patch_pos.repeat(reps, 1)[:idx.numel()]
        return pos

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[List[torch.Tensor]] = None,
        image_patch_indices: Optional[torch.Tensor] = None,
        image_patch_id: Optional[int] = None,
        mask: torch.Tensor = None,
        start_pos: Union[int, torch.Tensor] = 0,
        past_key_values: Optional[ModelCache] = None,
        output_attentions: bool = False,
        return_all_loop_outputs: bool = False,
        positions: Optional[torch.Tensor] = None,
    ) -> CausalLanguageModelOutput:
        '''
        Args:
            input_ids: [B, L]，图像占位符 token 的位置由 image_patch_indices 指定。
            images: 图像 Tensor 列表，或 [B] 个「每 batch 图像列表」。仅 prefill 阶段需要。
            image_patch_indices: [B, P] 占位符在序列中的绝对位置（Session.to_tensors() 可直接给出）。
            image_patch_id: 占位符 token id。给了它就不必再传 image_patch_indices——
                            直接从 input_ids 里扫出位置，训练管线只喂 batch 也能用。
        '''
        if image_patch_indices is None and image_patch_id is not None:
            image_patch_indices = torch.stack([
                (input_ids[b] == image_patch_id).nonzero(as_tuple=False).flatten()
                for b in range(input_ids.shape[0])
            ])

        grid = None
        grids = None
        patch_embeds = None
        if images is not None and image_patch_indices is not None:
            batch_size = input_ids.shape[0]
            if isinstance(images[0], list):
                images_by_batch = images
            else:
                images_by_batch = [list(images) for _ in range(batch_size)]

            # 视觉塔在 no_grad 下跑，不受外层 autocast 保护：训练时统一按语言塔的
            # autocast dtype 执行，避免 conv 权重被 autocast 转 bf16 后输入还是 fp32。
            vision_dtype = None
            if torch.is_autocast_enabled():
                vision_dtype = torch.get_autocast_dtype('cuda' if input_ids.is_cuda else 'cpu')
            elif self.language.token_emb.weight.dtype != torch.float32:
                vision_dtype = self.language.token_emb.weight.dtype

            patch_embeds = []
            grids = []
            patch_counts = []
            for b in range(batch_size):
                if not images_by_batch[b]:
                    patch_embeds.append(None)
                    patch_counts.append(None)
                    continue
                feats = []
                for img in images_by_batch[b]:
                    emb = self.encode_image(img.unsqueeze(0), dtype=vision_dtype).squeeze(0)
                    feats.append(emb)
                    if not grids:
                        # 二维位置模板只需一个样本的图数（各图同尺寸）
                        grids.append(self._infer_grid(emb.shape[0]))
                merged = torch.cat(feats, dim=0)
                patch_embeds.append(merged)
                patch_counts.append(merged.shape[0])

        if positions is None:
            if grids:
                positions = self._build_vlm_positions(
                    input_ids.shape[0], input_ids.shape[1], image_patch_indices, grids,
                    patch_counts, start_pos, input_ids.device,
                )
            elif self.language._needs_positions:
                # 纯文本路径：多轴编码仍要求显式 positions，用 (t, t)
                positions = self.language.build_positions(
                    input_ids.shape[0], input_ids.shape[1], start_pos, device=input_ids.device
                )

        x, positions = self.language.embed_with_positions(
            input_ids, positions=positions, start_pos=start_pos
        )

        if grids:
            for b in range(input_ids.shape[0]):
                emb = patch_embeds[b]
                if emb is None:
                    continue
                idx = image_patch_indices[b]
                idx = idx[idx >= 0]
                if idx.numel() != emb.shape[0]:
                    raise ValueError(
                        f'batch {b}: 占位符数量 {idx.numel()} != 视觉特征行数 {emb.shape[0]}'
                    )
                x[b, idx] = emb.to(x.dtype)

        embedding_start = 0 if positions is not None else start_pos

        x = self.language.looped(
            x,
            attention_mask=mask,
            output_attentions=output_attentions,
            position_emb=self.language.position_emb,
            embedding_start=embedding_start,
            embedding_pos=positions,
            past_key_values=past_key_values,
            return_all_loop_outputs=return_all_loop_outputs,
        )

        if return_all_loop_outputs:
            x, loop_outputs = x
        else:
            loop_outputs = None

        logits = self.proj_out(self.norm(x))

        return CausalLanguageModelOutput(
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=tuple(loop_outputs) if loop_outputs is not None else None,
        )

    @staticmethod
    def _infer_grid(num_patches: int) -> Tuple[int, int]:
        '''从 patch 数反推二维网格（MobileNetV3 的输出是正方形特征图）。'''
        side = int(math.isqrt(num_patches))
        if side * side != num_patches:
            raise ValueError(f'patch 数 {num_patches} 不是完全平方数，无法推断二维网格')
        return side, side

    def create_cache(self, device: Optional[torch.device] = None) -> ModelCache:
        return self.language.create_cache(device)


if __name__ == '__main__':
    model = Looped(LoopedConfig())
    print(model.count_params(human_readable=True))
