from codon import *
from codon.utils.theta import validate_rope_config
from codon.block.mlp   import MLP


class BasicEmbedding(BasicModel):
    '''
    Base class for Positional Embeddings.
    '''

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None, start_pos: int = 0, *args, **kwargs) -> torch.Tensor:
        '''
        Forward pass for positional embedding.

        Args:
            x (torch.Tensor): Input tensor.
            positions (torch.Tensor, optional): Position indices. Defaults to None.
            start_pos (int, optional): Starting position. Defaults to 0.

        Returns:
            torch.Tensor: Output tensor with positional information.
        '''
        raise NotImplementedError


class SinusoidalEmbedding(BasicEmbedding):
    '''
    Sinusoidal absolute positional encoding.

    Implements the standard sinusoidal positional encoding proposed in "Attention Is All You Need".
    Uses sine and cosine functions of different frequencies:
        PE(pos, 2i) = sin(pos / base^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / base^(2i/d_model))

    Attributes:
        model_dim (int): The dimension of the model.
        max_len (int): Maximum sequence length.
        base (int): Base for computing frequencies.
        pe (torch.Tensor): Buffer containing the positional encodings. Shape: [1, max_len, model_dim].
    '''

    def __init__(self, model_dim: int, max_len: int = 131072, base: int = 500000):
        '''
        Initializes the absolute positional encoding module.

        Args:
            model_dim (int): The dimension of the model.
            max_len (int, optional): Maximum sequence length. Defaults to 131072.
            base (int, optional): Base for computing frequencies. Defaults to 500000.
        '''
        super().__init__()

        self.model_dim = model_dim
        self.max_len = max_len
        self.base = base

        config = validate_rope_config(self.max_len, self.base)
        if not config.is_passed:
            print(f'Sinusoidal validation failed: {config.info}. Suggested base: {config.suggested_base}')
        
        pe = torch.zeros(max_len, model_dim)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(base) / model_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None, start_pos: int = 0, *args, **kwargs) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor. Shape: [Batch_Size, Seq_Len, model_dim].
            positions (torch.Tensor, optional): Explicit position indices. Shape: [Batch_Size, Seq_Len].
                                                If provided, retrieves embeddings for these indices.
            start_pos (int, optional): Starting position index. Used if positions is None. Defaults to 0.

        Returns:
            torch.Tensor: Tensor with positional encoding added.
        '''
        if positions is not None:
            # pe: [1, max_len, dim] -> [max_len, dim] -> [Batch, Seq_Len, Dim]
            pe = self.pe.squeeze(0)[positions]
        else:
            seq_len = x.size(1)
            pe = self.pe[:, start_pos : start_pos + seq_len, :]

        return x + pe


class TimestepEmbedding(BasicEmbedding):
    '''
    Base class for timestep embeddings.
    '''

    def get_embedding(self, timesteps: Union[torch.Tensor, int]) -> torch.Tensor:
        '''
        Get the embedding for the given timesteps.

        Args:
            timesteps (Union[torch.Tensor, int]): Input timesteps. Can be a tensor or an integer.
        
        Returns:
            torch.Tensor: Embedding for the given timesteps. Shape: [Batch_Size, dim].
        '''
        raise NotImplementedError


class TimestepSinusoidalEmbedding(TimestepEmbedding):
    '''
    Sinusoidal embedding for timesteps.
    '''

    def __init__(self, model_dim: int, max_period: int = 10000):
        '''
        Initializes the timestep sinusoidal embedding module.

        Args:
            model_dim (int): The dimension of the model.
            max_period (int, optional): Maximum period for the sinusoidal functions. Defaults to 10000.
        '''
        super().__init__()
        self.model_dim = model_dim
        self.max_period = max_period

    def get_embedding(self, timesteps: Union[torch.Tensor, int]) -> torch.Tensor:
        '''
        Forward pass for timestep sinusoidal embedding.

        Args:
            timesteps (Union[torch.Tensor, int]): Input timesteps. Can be a tensor or an integer.

        Returns:
            torch.Tensor: Sinusoidal embedding for the given timesteps. Shape: [Batch_Size, dim].
        '''
        if isinstance(timesteps, int):
            timesteps = torch.tensor([timesteps], dtype=torch.float32)
        else:
            timesteps = timesteps.float()
        
        half_dim = self.model_dim // 2
        exponent = -math.log(self.max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        freqs = torch.exp(exponent)
        arg = timesteps[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)

        if self.model_dim % 2:
            emb = torch.cat([emb, torch.zeros(timesteps.size(0), 1)], dim=-1)
        
        return emb
    
    def forward(self, x: torch.Tensor, timesteps: Union[torch.Tensor, int], *args, **kwargs) -> torch.Tensor:
        emb = self.get_embedding(timesteps)
        return x + emb


class TimestepMLPEmbedding(TimestepEmbedding):

    def __init__(
        self,
        model_dim: int,
        embed_dim: int = 512,
        max_period: int = 10000

    ):
        '''
        Initializes the timestep MLP embedding module.

        Args:
            model_dim (int): The dimension of the model.
            embed_dim (int, optional): Dimension of the intermediate embedding. Defaults to 512.
            max_period (int, optional): Maximum period for the sinusoidal functions. Defaults to 10000.
        '''
        super().__init__()
        self.model_dim  = model_dim
        self.embed_dim  = embed_dim
        self.max_period = max_period

        self.sinusoidal_embedding = TimestepSinusoidalEmbedding(self.embed_dim, max_period)

        self.mlp = MLP(
            in_dim=self.embed_dim,
            hidden_dim=self.model_dim,
            out_dim=self.model_dim,
            activation='silu',
        )
    
    def get_embedding(self, timesteps: Union[torch.Tensor, int]) -> torch.Tensor:
        '''
        Forward pass for timestep MLP embedding.

        Args:
            timesteps (Union[torch.Tensor, int]): Input timesteps. Can be a tensor or an integer.
        
        Returns:
            torch.Tensor: MLP embedding for the given timesteps. Shape: [Batch_Size, model_dim].
        '''
        sinusoidal_emb = self.sinusoidal_embedding(timesteps)
        mlp_emb = self.mlp(sinusoidal_emb)
        return mlp_emb
    
    def forward(self, x: torch.Tensor, timesteps: Union[torch.Tensor, int], *args, **kwargs) -> torch.Tensor:
        emb = self.get_embedding(timesteps)
        return x + emb


class BasicRotaryEmbedding(BasicEmbedding):
    '''
    Base class for Rotary Positional Embeddings.

    Attributes:
        model_dim (int): The dimension of the model.
        max_len (int): Maximum sequence length.
        base (int): Base for computing frequencies.
        cos_cached (torch.Tensor): Cached cosine values.
        sin_cached (torch.Tensor): Cached sine values.
    '''

    def __init__(self, model_dim: int, max_len: int = 131072, base: int = 500000) -> None:
        '''
        Initializes the BasicRotaryEmbedding module.

        Args:
            model_dim (int): The dimension of the model.
            max_len (int, optional): Maximum sequence length. Defaults to 131072.
            base (int, optional): Base for computing frequencies. Defaults to 500000.
        '''
        super().__init__()
        self.model_dim = model_dim
        self.max_len = max_len
        self.base = base

        config = validate_rope_config(self.max_len, self.base)
        if not config.is_passed:
            print(f'RoPE validation failed: {config.info}. Suggested base: {config.suggested_base}')

        inv_freq = 1.0 / (base ** (torch.arange(0, model_dim, 2).float() / model_dim))

        t = torch.arange(max_len, dtype=torch.float)

        freqs = torch.outer(t, inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Split the vector into two halves and rotate them: [-x2, x1].
        The split operation is performed on the last dimension (model_dim),
        regardless of whether the input is 3D or 4D.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Rotated tensor.
        '''
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(BasicRotaryEmbedding):
    '''
    Rotary Positional Embedding (RoPE).

    Attributes:
        model_dim (int): The dimension of the model.
        max_len (int): Maximum sequence length.
        base (int): Base for computing frequencies.
        cos_cached (torch.Tensor): Cached cosine values.
        sin_cached (torch.Tensor): Cached sine values.
    '''

    def __init__(self, model_dim: int, max_len: int = 131072, base: int = 500000) -> None:
        '''
        Initialize the RoPE module.

        Args:
            model_dim (int): The dimension of the model (or head_dim). Must be even.
            max_len (int, optional): Maximum sequence length for pre-computing position encodings. 
                                     Defaults to 131072.
            base (int, optional): Base for computing frequencies. Defaults to 500000.
        '''
        super().__init__(model_dim, max_len, base)

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None, start_pos: Union[int, torch.Tensor] = 0, *args, **kwargs) -> torch.Tensor:
        '''
        Apply rotary positional encoding.
        
        Automatically adapts to two types of inputs:
        1. [Batch, Seq_Len, Dim]
        2. [Batch, Head, Seq_Len, Head_Dim]

        Args:
            x (torch.Tensor): Input tensor.
            positions (torch.Tensor, optional): Explicit position indices. Shape: [Batch, Seq_Len].
                If provided, uses these indices to retrieve positional embeddings.
            start_pos (Union[int, torch.Tensor], optional): Starting position index for KV Cache inference.
                                       Used if positions is None. Defaults to 0.

        Returns:
            torch.Tensor: Tensor with positional information added.
        '''
        ndim = x.ndim
        seq_len = x.shape[-2]

        if positions is not None:
            # positions: [Batch, Seq_Len] -> cos/sin: [Batch, Seq_Len, Dim]
            cos = self.cos_cached[positions]
            sin = self.sin_cached[positions]
            
            if ndim == 4:
                # [Batch, Seq_Len, Dim] -> [Batch, 1, Seq_Len, Dim]
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
        else:
            if not isinstance(start_pos, torch.Tensor):
                start_pos_tensor = torch.tensor(start_pos, device=x.device, dtype=torch.long)
            else:
                start_pos_tensor = start_pos.to(device=x.device, dtype=torch.long)
            
            positions_idx = torch.arange(seq_len, device=x.device, dtype=torch.long) + start_pos_tensor
            cos = self.cos_cached[positions_idx]
            sin = self.sin_cached[positions_idx]
            
            shape = [1] * (ndim - 2) + [seq_len, -1]
            cos = cos.view(*shape)
            sin = sin.view(*shape)

        # Handle cases where hidden_dim is a multiple of model_dim (e.g., when attention is skipped)
        if x.shape[-1] > cos.shape[-1]:
            multiplier = x.shape[-1] // cos.shape[-1]
            cos = cos.repeat(*([1] * (cos.ndim - 1)), multiplier)
            sin = sin.repeat(*([1] * (sin.ndim - 1)), multiplier)

        # Cast cos/sin to match input dtype to keep mixed-precision graphs (e.g. ONNX FP16
        # exports) type-consistent. cos_cached/sin_cached are non-persistent buffers and
        # therefore not affected by `model.half()`.
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        return (x * cos) + (self._rotate_half(x) * sin)


# FoPE heuristic sigma anchors from Hua et al. ICML 2025 (Table 3): the optimal
# harmonic-noise std (Var_Freq sigma) measured per model scale via grid search.
_FOPE_SIGMA_ANCHORS = (
    (60e6, 0.3),
    (180e6, 0.4),
    (1.2e9, 0.6),
)


def fope_sigma(num_params: float) -> float:
    '''
    Heuristic std for FourierRotaryEmbedding ``sigma``, fitted to the paper's
    Table 3 anchors by piecewise-linear interpolation in log10(parameter count):

        (60M, 0.3), (180M, 0.4), (1.2B, 0.6)

    Parameter counts outside the anchor range are clamped to the nearest
    anchor's sigma. This is an empirical fit -- the paper provides no closed-form
    relationship between sigma and model scale -- so treat the result as an
    initialization and re-tune on your own data / length-generalization eval.

    Args:
        num_params (float): Total (trainable) parameter count of the model.

    Returns:
        float: Suggested sigma, bounded within [0.3, 0.6].
    '''
    x = math.log10(num_params)
    for (n0, s0), (n1, s1) in zip(_FOPE_SIGMA_ANCHORS, _FOPE_SIGMA_ANCHORS[1:]):
        x0, x1 = math.log10(n0), math.log10(n1)
        if x0 <= x <= x1:
            return s0 + (s1 - s0) * (x - x0) / (x1 - x0)
    if x < math.log10(_FOPE_SIGMA_ANCHORS[0][0]):
        return _FOPE_SIGMA_ANCHORS[0][1]
    return _FOPE_SIGMA_ANCHORS[-1][1]


class FourierRotaryEmbedding(RotaryEmbedding):
    '''
    Fourier Position Embedding (FoPE), from Hua et al. ICML 2025 (arXiv:2412.17739).

    FoPE is a drop-in RoPE replacement targeting length generalization. It makes
    two modifications to how each rotary dimension stores position:

    1. **Fourier Series (FS)**: Instead of treating each dimension as a *single*
       frequency, every adequately-trained frequency component is turned into a
       Fourier series whose dominant term is the original frequency plus weaker
       harmonic terms drawn from the other adequately-trained frequencies:
           h_m(n) = e^{i ω_m n} + Σ_k a_{k,m} e^{i ω_k n}
       The random coefficients a_{k,m} (std = ``sigma``) model the *Spectrum
       Damage* real linear/activation layers inject into hidden states.

    2. **Clip-to-Floor (CF)**: Frequencies whose period exceeds the training
       window (ω_m < 2π / train_len) never complete a full cycle during
       pre-training, so they are under-trained. They are replaced by the
       zero-frequency component (position-invariant: cos=1, sin=0), which carries
       no positional bias and therefore extrapolates cleanly to longer contexts.

    Implementation notes (geometry of this module):
      - The repository's RoPE cache stores cos/sin with two identical halves
        (pair index ``d`` rotates with ``d + head_dim//2``). FoPE only rewrites
        those cached cos/sin tables; ``RotaryEmbedding.forward``, its position
        indexing and the KV-cache path are reused verbatim.
      - Coefficient matrices are frozen (no gradients) and shared across heads /
        layers, unlike the per-head weights in the reference implementation. This
        is required by this codebase's GQA setup, where the embedding is shared
        between a ``num_heads``-head query and a ``num_kv_heads``-head key.
      - All tables/coefficients are non-persistent, so swapping this module in
        does **not** change ``state_dict`` keys or checkpoint layout.

    Both sub-methods are independently reachable:
        - ``sigma = 0.0`` and ``train_len=None``  -> identical to plain RoPE.
        - ``sigma = 0.0`` with a ``train_len``     -> CF only.
        - ``sigma > 0``  with ``train_len=None``  -> FS only.
        - ``sigma > 0``  with a ``train_len``     -> full FoPE.

    Attributes:
        sigma (float): Std of the Fourier-series harmonic coefficients. If not
            passed to ``__init__``, derived from ``num_params`` via
            :func:`fope_sigma` (or 0.3 when no parameter count is given).
        train_len (Optional[int]): Pre-training sequence window used for the
            floor-frequency clip (CF). ``None`` disables clipping.
        num_params (Optional[float]): Model parameter count used to fit ``sigma``.
        num_freqs_kept (int): Number of frequencies that pass the floor clip and
            are active after ``__init__``.
        sin_coef (torch.Tensor): Frozen sine-series coefficient matrix, shape
            ``[num_freqs_kept, num_freqs_kept]`` (non-persistent buffer).
        cos_coef (torch.Tensor): Frozen cosine-series coefficient matrix
            (non-persistent buffer).
        cos_cached (torch.Tensor): Fourier-series cosine table ``[max_len, model_dim]``
            (non-persistent buffer, supersedes the base RoPE table).
        sin_cached (torch.Tensor): Fourier-series sine table (non-persistent buffer).
    '''

    def __init__(
        self,
        model_dim: int,
        max_len: int = 131072,
        base: int = 500000,
        sigma: Optional[float] = None,
        train_len: Optional[int] = None,
        num_params: Optional[float] = None,
    ) -> None:
        '''
        Initialize the FoPE module.

        Args:
            model_dim (int): The dimension of the model (or head_dim). Must be even.
            max_len (int, optional): Maximum sequence length for pre-computing
                                     position encodings. Defaults to 131072.
            base (int, optional): Base for computing frequencies. Defaults to 500000.
            sigma (float, optional): Std of the Fourier-series harmonic coefficients
                                     (Var_Freq σ in the paper). ``0.0`` disables the
                                     Fourier Series, keeping only the dominant
                                     frequency. When ``None``, sigma is derived from
                                     ``num_params`` via :func:`fope_sigma`
                                     (log10-scaled fit of the paper's Table 3), or
                                     falls back to 0.3 if ``num_params`` is also
                                     ``None``. Defaults to None.
            train_len (int, optional): Pre-training sequence window used for the
                                       floor-frequency clip (CF). Frequencies with
                                       period > train_len are under-trained and are
                                       replaced by the zero-frequency component.
                                       ``None`` (default) disables clipping.
            num_params (float, optional): Total (trainable) parameter count of the
                                          model. Used only when ``sigma`` is not
                                          given, to pick sigma by model scale via
                                          :func:`fope_sigma`. Defaults to None.
        '''
        assert model_dim % 2 == 0, 'model_dim must be even'
        if sigma is not None:
            assert sigma >= 0.0, 'sigma must be non-negative'
        if num_params is not None:
            assert num_params > 0, 'num_params must be > 0 when provided'

        super().__init__(model_dim, max_len, base)

        # Manual sigma wins; otherwise fit sigma from the model's parameter count
        # (log10 piecewise fit of the paper's Table 3 anchors); fall back to the
        # paper's 60M-scale default 0.3 when neither is given.
        if sigma is None:
            sigma = fope_sigma(num_params) if num_params is not None else 0.3

        self.sigma = float(sigma)
        self.train_len = train_len
        self.num_params = num_params
        self.num_freqs_kept = model_dim // 2

        half = model_dim // 2

        # Base angular frequencies, one per rotary pair. Kept bit-identical to
        # BasicRotaryEmbedding so that sigma=0 / no-clip reproduces plain RoPE
        # exactly (including the reciprocal-form rounding).
        inv_freq = 1.0 / (base ** (torch.arange(0, model_dim, 2, dtype=torch.float) / model_dim))

        # CF: drop frequencies that cannot finish a cycle inside train_len.
        if train_len is not None:
            floor_freq = 2.0 * math.pi / float(train_len)
            keep_mask = inv_freq >= floor_freq
            if keep_mask.sum() == 0:
                print(f'[FourierRotaryEmbedding] train_len={train_len} clips every '
                      'frequency; keeping RoPE frequencies unclipped.')
                keep_mask = torch.ones_like(keep_mask)
            self.num_freqs_kept = int(keep_mask.sum())
            inv_freq = inv_freq[keep_mask]

        K = int(inv_freq.numel())

        # Fourier series coefficients: identity (dominant frequency) + σ noise.
        # Each column of the matrix is the harmonic weight vector of one pair.
        coef_s = (sigma / math.sqrt(K)) * torch.randn(K, K)
        coef_s = coef_s.fill_diagonal_(1.0)
        coef_c = (sigma / math.sqrt(K)) * torch.randn(K, K)
        coef_c = coef_c.fill_diagonal_(1.0)

        self.register_buffer('sin_coef', coef_s, persistent=False)
        self.register_buffer('cos_coef', coef_c, persistent=False)

        # Base per-frequency sin/cos over positions [0, max_len).
        seq = torch.arange(max_len, dtype=torch.float)
        ang = torch.outer(seq, inv_freq)                # [max_len, K]
        base_sin, base_cos = ang.sin(), ang.cos()

        # Mix into a Fourier series per pair -> [max_len, K].
        fourier_sin = base_sin @ coef_s
        fourier_cos = base_cos @ coef_c

        # Rebuild the half-length cache. Trailing pairs (under-trained freqs) get
        # the zero-frequency component: cos=1, sin=0 (position-invariant).
        pad = half - K
        if pad > 0:
            ones = torch.ones(max_len, pad, dtype=torch.float)
            zeros = torch.zeros(max_len, pad, dtype=torch.float)
            fourier_sin = torch.cat([fourier_sin, zeros], dim=-1)
            fourier_cos = torch.cat([fourier_cos, ones], dim=-1)

        # Two identical halves, matching the base RoPE table layout.
        cos_cached = torch.cat([fourier_cos, fourier_cos], dim=-1)
        sin_cached = torch.cat([fourier_sin, fourier_sin], dim=-1)

        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)


class InterleavedRotaryEmbedding(BasicRotaryEmbedding):
    '''
    Interleaved Multimodal Rotary Positional Embedding (MRoPE-Interleave).
    
    Supports multi-dimensional positions (e.g., 3D for video: time, height, width), 
    with frequency channels assigned in a rotating interleaved manner across dimensions.

    Attributes:
        model_dim (int): The dimension of the model.
        max_len (int): Maximum sequence length.
        base (int): Base for computing frequencies.
        cos_cached (torch.Tensor): Cached cosine values.
        sin_cached (torch.Tensor): Cached sine values.
        num_axes (int): Number of positional axes.
        axis_mask (torch.Tensor): Mask for assigning frequency channels to axes.
        interleave_idx (torch.Tensor): Indices for interleaving.
    '''

    def __init__(self, model_dim: int, max_len: int = 131072, base: int = 500000, num_axes: int = 3) -> None:
        '''
        Initializes the MRoPEInterleaved module.

        Args:
            model_dim (int): The dimension of the model. Must be even and divisible by num_axes.
            max_len (int, optional): Maximum sequence length for pre-computing position encodings.
                                     Defaults to 131072.
            base (int, optional): Base for computing frequencies. Defaults to 500000.
            num_axes (int, optional): Number of positional axes (e.g., 3 for time, height, width).
                                      Defaults to 3.
        '''
        assert model_dim % 2 == 0, 'model_dim must be even'
        assert model_dim % num_axes == 0, f'model_dim {model_dim} not divisible by num_axes {num_axes}'
        
        super().__init__(model_dim, max_len, base)
        
        self.num_axes = num_axes
        
        self.register_buffer(
            'axis_mask', 
            torch.arange(model_dim) % num_axes, 
            persistent=False
        )
        
        k = model_dim // num_axes
        idx = []
        for p in range(model_dim):
            j = p % num_axes
            i = p // num_axes
            pos_in_old = j * k + i
            idx.append(pos_in_old)
            
        self.register_buffer('interleave_idx', torch.tensor(idx, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None, start_pos: int = 0, *args, **kwargs) -> torch.Tensor:
        '''
        Apply multimodal rotary positional encoding.

        Args:
            x (torch.Tensor): Input tensor. Shape: [Batch, Seq_Len, Dim] or [Batch, Head, Seq_Len, Head_Dim].
            positions (torch.Tensor, optional): Position index tensor. Shape: [Batch, Seq_Len] or
                [Batch, Seq_Len, num_axes].
                If 2D tensor, it will be automatically expanded to [Batch, Seq_Len, num_axes].
                If None and num_axes=1, linear position indices will be automatically created.
            start_pos (int, optional): Starting position index. Defaults to 0.

        Returns:
            torch.Tensor: Tensor with positional information added.
        
        Raises:
            ValueError: If positions is None and num_axes > 1.
        '''
        ndim = x.ndim
        seq_len = x.shape[-2]
        batch_size = x.shape[0]
        
        if positions is None:
            if self.num_axes == 1:
                positions = torch.arange(0, seq_len, device=x.device, dtype=torch.long)
            else:
                raise ValueError('positions must be provided when num_axes > 1 (e.g. for vision/multimodal inputs)')
        
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.num_axes)
            
        if positions.ndim == 2:
            positions = positions.unsqueeze(-1).expand(-1, -1, self.num_axes)
            
        if positions.ndim == 3 and positions.shape[-1] == 1:
            positions = positions.expand(-1, -1, self.num_axes)
            
        batch_size = positions.shape[0]
        
        cos_list, sin_list = [], []
        
        for ax in range(self.num_axes):
            pos_ax = positions[..., ax]
            pos_ax = torch.clamp(pos_ax + start_pos, 0, self.max_len - 1).long()
            
            cos_full = self.cos_cached[pos_ax]
            sin_full = self.sin_cached[pos_ax]
            
            mask = (self.axis_mask == ax)
            cos_ax = cos_full[..., mask]
            sin_ax = sin_full[..., mask]
            
            cos_list.append(cos_ax)
            sin_list.append(sin_ax)
        
        cos_all = torch.cat(cos_list, dim=-1)
        sin_all = torch.cat(sin_list, dim=-1)
        
        cos_all = cos_all[..., self.interleave_idx]
        sin_all = sin_all[..., self.interleave_idx]
        
        if ndim == 4:
            shape = [batch_size, 1, seq_len, -1]
            cos_all = cos_all.view(*shape)
            sin_all = sin_all.view(*shape)

        # Handle cases where hidden_dim is a multiple of model_dim (e.g., when attention is skipped)
        if x.shape[-1] > cos_all.shape[-1]:
            multiplier = x.shape[-1] // cos_all.shape[-1]
            cos_all = cos_all.repeat(*([1] * (cos_all.ndim - 1)), multiplier)
            sin_all = sin_all.repeat(*([1] * (sin_all.ndim - 1)), multiplier)

        # Cast cos/sin to match input dtype to keep mixed-precision graphs (e.g. ONNX FP16
        # exports) type-consistent. cos_cached/sin_cached are non-persistent buffers and
        # therefore not affected by `model.half()`.
        cos_all = cos_all.to(x.dtype)
        sin_all = sin_all.to(x.dtype)

        return (x * cos_all) + (self._rotate_half(x) * sin_all)
