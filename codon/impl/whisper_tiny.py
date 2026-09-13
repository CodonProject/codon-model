from codon import *
from codon.ops import AttentionOutput


class WhisperEncoderAttention(BasicModel):
    '''
    Whisper encoder 的自注意力（双向、k_proj 无 bias、6 头）。

    与 codon 的 `MultiHeadAttention` 差别：
      - 非因果（encoder 是双向的）；
      - q/v/o 三个投影带 bias，**k_proj 不带 bias**（Whisper 的原始设计，
        encoder 与 decoder 的 self_attn / encoder_attn 都是如此，已核对权重文件）；
      - 无 QK-norm、无 gate、无 RoPE（位置信息由 `embed_positions` 加法注入）。

    投影命名沿用 codon 的 `proj_*` 习惯（对齐 motif_a1 的 `proj_out`）。

    Attributes:
        proj_q (nn.Linear): Query 投影（带 bias）。
        proj_k (nn.Linear): Key 投影（无 bias）。
        proj_v (nn.Linear): Value 投影（带 bias）。
        proj_o (nn.Linear): 输出投影（带 bias）。
    '''
    def __init__(self, model_dim: int = 384, num_heads: int = 6, dropout: float = 0.0):
        super().__init__()
        assert model_dim % num_heads == 0, 'model_dim must be divisible by num_heads'

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.dropout = dropout
        self.scale = self.head_dim ** -0.5

        self.proj_q = nn.Linear(model_dim, model_dim, bias=True)
        self.proj_k = nn.Linear(model_dim, model_dim, bias=False)
        self.proj_v = nn.Linear(model_dim, model_dim, bias=True)
        self.proj_o = nn.Linear(model_dim, model_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        '''
        Args:
            hidden_states: [B, L, D]。
            attention_mask: 加法式掩码（0 / -inf），形状可广播到 [B, 1, L, L]；None 表示全可见。
            output_attentions: 是否返回注意力权重。
        '''
        b, l, _ = hidden_states.shape

        def split(x):
            return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        q = split(self.proj_q(hidden_states))
        k = split(self.proj_k(hidden_states))
        v = split(self.proj_v(hidden_states))

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(b, l, self.model_dim)

        return AttentionOutput(
            output=self.proj_o(out),
            attention_weights=attn_weights if output_attentions else None,
        )


class WhisperEncoderMLP(BasicModel):
    '''
    Whisper encoder 的前馈层：Linear -> GELU -> Linear，两层都带 bias。

    Attributes:
        proj_fc1 (nn.Linear): 升维投影（384 -> 1536）。
        proj_fc2 (nn.Linear): 降维投影（1536 -> 384）。
    '''
    def __init__(self, model_dim: int = 384, ffn_dim: int = 1536, dropout: float = 0.0):
        super().__init__()
        self.proj_fc1 = nn.Linear(model_dim, ffn_dim, bias=True)
        self.proj_fc2 = nn.Linear(ffn_dim, model_dim, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj_fc2(self.act(self.proj_fc1(x))))


class WhisperEncoderLayer(BasicModel):
    '''
    Whisper encoder 单层：pre-LN 自注意力 + pre-LN 前馈。

    Attributes:
        attn_norm (nn.LayerNorm): 注意力前的归一化。
        attn (WhisperEncoderAttention): 双向自注意力。
        fn_norm (nn.LayerNorm): 前馈前的归一化。
        mlp (WhisperEncoderMLP): 前馈网络。
    '''
    def __init__(
        self,
        model_dim: int = 384,
        num_heads: int = 6,
        ffn_dim: int = 1536,
        dropout: float = 0.0,
        idx: Union[int, str] = None,
    ):
        super().__init__()
        self.idx = str(idx) if idx is not None else safecode()

        self.attn_norm = nn.LayerNorm(model_dim)
        self.attn = WhisperEncoderAttention(model_dim, num_heads, dropout)
        self.fn_norm = nn.LayerNorm(model_dim)
        self.mlp = WhisperEncoderMLP(model_dim, ffn_dim, dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        x = self.attn_norm(hidden_states)
        attn_out = self.attn(x, attention_mask=attention_mask,
                             output_attentions=output_attentions)
        hidden_states = hidden_states + attn_out.output

        hidden_states = hidden_states + self.mlp(self.fn_norm(hidden_states))
        return hidden_states, attn_out.attention_weights


class WhisperTinyAudioEncoder(BasicModel):
    '''
    Whisper-tiny 音频编码器（80 mel -> 384 dim，4 层，6 头，8.21M 参数）。

    结构（与 openai/whisper 完全一致，已与 transformers 官方实现逐元素对拍，误差 ~1e-5）：

        mel [B, 80, T]                      T = 3000（固定 30 秒 @100Hz）
          conv1 Conv1d(80 -> 384, k=3, s=1, p=1) + bias, GELU
          conv2 Conv1d(384 -> 384, k=3, s=2, p=1) + bias, GELU
          transpose -> [B, T/2, 384]
          + embed_positions (1500, 384)     可学习绝对位置
          layers × 4（pre-LN，双向注意力）
          norm (LayerNorm)
        输出 [B, T/2, 384]

    可选 `pool_stride`：在编码器输出后加一层无重叠平均池化做时间降采样，
    用于压低喂给语言模型的 token 数（stride=8 -> 30 秒音频 187 个 token）。
    池化发生在 `embed_positions` 之后，不改变编码器本身的权重。

    Attributes:
        conv1 (nn.Conv1d): mel -> model_dim。
        conv2 (nn.Conv1d): 时间下采样 2x。
        embed_positions (nn.Embedding): 可学习绝对位置（1500）。
        layers (nn.ModuleList): WhisperEncoderLayer × num_layers。
        norm (nn.LayerNorm): 最终归一化。
    '''
    __remote_resource__ = {
        'repo': 'CodonProject/Whisper-Tiny-Encoder',
        'files': ['whisper_tiny_encoder.safetensors'],
        'repo_type': 'model',
    }

    def __init__(
        self,
        num_mel_bins: int = 80,
        model_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 6,
        ffn_dim: int = 1536,
        max_source_positions: int = 1500,
        dropout: float = 0.0,
        pool_stride: int = 1,
    ):
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.max_source_positions = max_source_positions
        self.pool_stride = max(1, int(pool_stride))

        self.conv1 = nn.Conv1d(num_mel_bins, model_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(model_dim, model_dim, kernel_size=3, stride=2, padding=1)
        self.act = nn.GELU()

        self.embed_positions = nn.Embedding(max_source_positions, model_dim)

        self.layers = nn.ModuleList([
            WhisperEncoderLayer(model_dim, num_heads, ffn_dim, dropout, idx=i)
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(model_dim)

    @property
    def downsample_factor(self) -> int:
        '''**相对编码器输出**的降采样倍率 = pool_stride。

        注意完整链路是 mel(100Hz) -> conv2(2x) -> pool(pool_stride)，
        即相对 mel 帧率的总倍率是 `2 * pool_stride`：
        pool_stride=1 时 30 秒音频输出 1500 个 token，pool_stride=8 时输出 187 个。
        '''
        return self.pool_stride

    def forward(
        self,
        mel: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        '''
        Args:
            mel: [B, num_mel_bins, T]。与官方一致时 T = 3000（30 秒），
                 短音频应 pad 到 3000 并丢弃 pad 段的输出。
            attention_mask: 加法式掩码，形状 [B, 1, 1, T_out] 或可广播的形式。
            output_attentions: 是否收集每层注意力权重。

        Returns:
            (hidden_states [B, T/2/pool_stride, model_dim], attentions or None)
        '''
        x = self.act(self.conv1(mel))
        x = self.act(self.conv2(x))
        x = x.permute(0, 2, 1) # [B, T/2, D]

        seq_len = x.shape[1]
        if seq_len > self.max_source_positions:
            raise ValueError(
                f'序列长度 {seq_len} 超过 embed_positions 的 {self.max_source_positions}'
            )
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.embed_positions(positions).to(x.dtype)

        all_attentions = [] if output_attentions else None
        for layer in self.layers:
            x, attn = layer(x, attention_mask=attention_mask,
                            output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attn)

        x = self.norm(x)

        if self.pool_stride > 1:
            x = x.transpose(1, 2)                           # [B, D, L]
            x = F.avg_pool1d(x, kernel_size=self.pool_stride, stride=self.pool_stride)
            x = x.transpose(1, 2)                           # [B, L/stride, D]

        return x, all_attentions

    def _load_remote(self, local_paths: list, **kwargs):
        '''从 `__remote_resource__` 下载后加载（键名已是 codon 命名，直接 load）。'''
        if not local_paths:
            raise ValueError('no local file downloaded')
        target = local_paths[0]
        for path in local_paths:
            if os.path.basename(path) == self.__remote_resource__['files'][0]:
                target = path
                break
        return self.load(target, strict=True)
