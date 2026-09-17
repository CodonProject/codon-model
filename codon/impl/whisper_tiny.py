from codon import *
from codon.ops import AttentionOutput


class WhisperEncoderAttention(BasicModel):
    '''
    Self-attention used by the Whisper encoder (bidirectional, no bias on ``k_proj``, 6 heads).

    Attributes:
        proj_q (nn.Linear): Query projection (with bias).
        proj_k (nn.Linear): Key projection (without bias).
        proj_v (nn.Linear): Value projection (with bias).
        proj_o (nn.Linear): Output projection (with bias).
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
            hidden_states (torch.Tensor): Input states of shape [B, L, D].
            attention_mask (Optional[torch.Tensor], optional): Additive mask (0 / -inf) that
                broadcasts to [B, 1, L, L]; None means every position is visible. Defaults to None.
            output_attentions (bool, optional): Whether to also return the attention weights.
                Defaults to False.

        Returns:
            AttentionOutput: The projected attention output, together with the attention weights
                when ``output_attentions`` is True.
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
    Feed-forward layer of the Whisper encoder: Linear -> GELU -> Linear, both with bias.

    Attributes:
        proj_fc1 (nn.Linear): Expansion projection (384 -> 1536).
        proj_fc2 (nn.Linear): Contraction projection (1536 -> 384).
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
    A single Whisper encoder layer: pre-LN self-attention followed by a pre-LN feed-forward.

    Attributes:
        attn_norm (nn.LayerNorm): Normalization applied before attention.
        attn (WhisperEncoderAttention): Bidirectional self-attention.
        fn_norm (nn.LayerNorm): Normalization applied before the feed-forward.
        mlp (WhisperEncoderMLP): The feed-forward network.
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
    Whisper-tiny audio encoder (80 mel -> 384 dim, 4 layers, 6 heads, 8.21M parameters).

    The optional ``pool_stride`` appends a non-overlapping average pool after the encoder output
    to downsample time, which keeps the number of tokens fed to the language model small
    (stride=8 -> 187 tokens for 30 seconds of audio). The pooling happens after
    ``embed_positions`` and does not change the encoder weights themselves.

    Attributes:
        conv1 (nn.Conv1d): mel -> model_dim.
        conv2 (nn.Conv1d): 2x temporal downsampling.
        embed_positions (nn.Embedding): Learnable absolute positions (1500).
        layers (nn.ModuleList): WhisperEncoderLayer x num_layers.
        norm (nn.LayerNorm): Final normalization.
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
        '''Downsampling factor **relative to the encoder output**, equal to pool_stride.

        Note that the full chain is mel (100Hz) -> conv2 (2x) -> pool (pool_stride), so the
        overall factor relative to the mel frame rate is ``2 * pool_stride``: with
        pool_stride=1, 30 seconds of audio yields 1500 tokens; with pool_stride=8 it yields 187.
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
            mel (torch.Tensor): Input mel spectrogram of shape [B, num_mel_bins, T]. To match the
                official model, T = 3000 (30 seconds); shorter audio should be padded to 3000 and
                the outputs of the padded region discarded.
            attention_mask (Optional[torch.Tensor], optional): Additive mask of shape
                [B, 1, 1, T_out], or anything broadcastable to it. Defaults to None.
            output_attentions (bool, optional): Whether to collect the attention weights of every
                layer. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Optional[List[torch.Tensor]]]: ``hidden_states`` of shape
                [B, T/2/pool_stride, model_dim] and the per-layer attention weights, or None when
                ``output_attentions`` is False.
        '''
        x = self.act(self.conv1(mel))
        x = self.act(self.conv2(x))
        x = x.permute(0, 2, 1) # [B, T/2, D]

        seq_len = x.shape[1]
        if seq_len > self.max_source_positions:
            raise ValueError(
                f'sequence length {seq_len} exceeds the {self.max_source_positions} positions of embed_positions'
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
        if not local_paths:
            raise ValueError('no local file downloaded')
        target = local_paths[0]
        for path in local_paths:
            if os.path.basename(path) == self.__remote_resource__['files'][0]:
                target = path
                break
        return self.load(target, strict=True)
