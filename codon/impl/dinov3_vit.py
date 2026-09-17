from codon import *
from safetensors.torch import load_file

from codon.block.activation import get_activation
from codon.block.embedding import BasicEmbedding
from codon.block.norm import LayerNorm
from codon.ops import apply_attention


dinov3_sizes = {
    'vits16': dict(embed_dim=384, depth=12, num_heads=6, ffn_ratio=4.0),
    'vitb16': dict(embed_dim=768, depth=12, num_heads=12, ffn_ratio=4.0),
    'vitl16': dict(embed_dim=1024, depth=24, num_heads=16, ffn_ratio=4.0),
    'vitso400m': dict(embed_dim=1152, depth=27, num_heads=18, ffn_ratio=3.777777778),
    'vith16plus': dict(embed_dim=1280, depth=32, num_heads=20, ffn_ratio=4.0),
    'vit7b16': dict(embed_dim=4096, depth=40, num_heads=32, ffn_ratio=3.0),
}


_DTYPE_ALIASES = {
    'float16': torch.float16, 'fp16': torch.float16, 'half': torch.float16,
    'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16,
    'float32': torch.float32, 'fp32': torch.float32, 'float': torch.float32,
}


def _parse_config_dtype(config: Dict[str, Any]) -> Optional[torch.dtype]:
    '''
    Parses the weight precision from a config.

    Args:
        config (Dict[str, Any]): The already loaded config.json.

    Returns:
        Optional[torch.dtype]: The parsed dtype; None when no valid field is present.
    '''
    for field in ('dtype', 'torch_dtype', 'weight_dtype'):
        raw = config.get(field)
        if isinstance(raw, str):
            parsed = _DTYPE_ALIASES.get(raw.lower().strip())
            if parsed is not None:
                return parsed
    return None


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    '''
    Half-split rotation for RoPE: [x1, x2] -> [-x2, x1] (split along the last dimension).

    Identical to `codon.block.embedding.BasicRotaryEmbedding._rotate_half`: dimension d within an
    axis is paired with dimension d + head_dim/2. It is kept as a separate copy here because the
    DINOv3 RoPE only applies to patch tokens and cannot reuse `RotaryEmbedding.forward` directly
    (that one rotates the whole sequence by token index).
    '''
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class DropPath(BasicModel):
    '''
    Stochastic Depth (DropPath): randomly drops the whole residual branch per sample during
    training and passes through during inference.

    Attributes:
        drop_prob (float): Drop probability.
    '''

    def __init__(self, drop_prob: float = 0.0):
        '''
        Args:
            drop_prob (float, optional): Drop probability. Defaults to 0.0.
        '''
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        return x.div(keep_prob) * random_tensor


class DINOv3ViTRope(BasicEmbedding):
    '''
    The 2D axial RoPE (Rotary Position Embedding) of DINOv3.

    Attributes:
        embed_dim (int): Model hidden dimension.
        num_heads (int): Number of attention heads.
        head_dim (int): Per-head dimension = embed_dim // num_heads.
        base (float): Base frequency of the RoPE (the upstream `rope_theta`).
        num_prefix_tokens (int): Length of the sequence prefix (CLS + storage token), which does
            not take part in the rotation.
        shift_coords (Optional[float]): Random coordinate shift magnitude used in training mode.
        jitter_coords (Optional[float]): Log-uniform coordinate jitter factor used in training mode.
        rescale_coords (Optional[float]): Log-uniform coordinate rescale factor used in training mode.
        inv_freq (torch.Tensor): Inverse frequency table [head_dim // 4], a non-persistent buffer.
    '''

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        base: float = 100.0,
        num_prefix_tokens: int = 1,
        shift_coords: Optional[float] = None,
        jitter_coords: Optional[float] = None,
        rescale_coords: Optional[float] = None,
    ):
        '''
        Args:
            embed_dim (int): Model hidden dimension.
            num_heads (int): Number of attention heads; `embed_dim` must be divisible by it.
            base (float, optional): Base frequency of the RoPE. Defaults to 100.0.
            num_prefix_tokens (int, optional): Number of prefix tokens excluded from the rotation.
                Defaults to 1 (CLS only).
            shift_coords (Optional[float], optional): Coordinate shift magnitude in training mode;
                None disables it. Defaults to None.
            jitter_coords (Optional[float], optional): Coordinate jitter factor in training mode;
                None disables it. Defaults to None.
            rescale_coords (Optional[float], optional): Coordinate rescale factor in training mode;
                None disables it. Defaults to None.
        '''
        super().__init__()

        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.base = float(base)
        self.num_prefix_tokens = num_prefix_tokens
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        inv_freq = 1.0 / self.base ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def patch_coords(self, num_patches_h: int, num_patches_w: int, device, dtype=torch.float32) -> torch.Tensor:
        '''
        Computes normalized patch center coordinates in [-1, +1].

        Args:
            num_patches_h (int): Patch grid height.
            num_patches_w (int): Patch grid width.
            device (torch.device): Target device.
            dtype (torch.dtype, optional): Coordinate precision. Defaults to torch.float32.

        Returns:
            torch.Tensor: (y, x) coordinates of shape [num_patches_h * num_patches_w, 2].
        '''
        coords_h = torch.arange(0.5, num_patches_h, dtype=dtype, device=device) / num_patches_h
        coords_w = torch.arange(0.5, num_patches_w, dtype=dtype, device=device) / num_patches_w
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1)
        coords = coords.flatten(0, 1)
        return 2.0 * coords - 1.0

    def augment_coords(self, coords: torch.Tensor) -> torch.Tensor:
        '''Training-mode coordinate augmentation: shift / jitter / rescale.

        Returns ``coords`` unchanged when all three options are None.
        '''
        if self.shift_coords is not None:
            shift_hw = torch.empty((1, 2), device=coords.device, dtype=coords.dtype).uniform_(
                -self.shift_coords, self.shift_coords
            )
            coords = coords + shift_hw
        if self.jitter_coords is not None:
            jitter_range = math.log(self.jitter_coords)
            jitter_hw = torch.empty((1, 2), device=coords.device, dtype=coords.dtype).uniform_(
                -jitter_range, jitter_range
            ).exp()
            coords = coords * jitter_hw
        if self.rescale_coords is not None:
            rescale_range = math.log(self.rescale_coords)
            rescale_hw = torch.empty(1, device=coords.device, dtype=coords.dtype).uniform_(
                -rescale_range, rescale_range
            ).exp()
            coords = coords * rescale_hw
        return coords

    def sincos(self, num_patches_h: int, num_patches_w: int, device, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Computes the cos / sin tables of the RoPE.

        Args:
            num_patches_h (int): Patch grid height.
            num_patches_w (int): Patch grid width.
            device (torch.device): Target device.
            batch_size (int, optional): Batch size; when > 1 every sample shares the same
                coordinates. Defaults to 1.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Both of shape [batch_size, 1, num_patches, head_dim].
        '''
        coords = self.patch_coords(num_patches_h, num_patches_w, device)
        if self.training:
            coords = self.augment_coords(coords)

        # [P, 2, head_dim/4] -> [P, head_dim/2] -> [P, head_dim]
        angles = 2 * math.pi * coords[:, :, None] * self.inv_freq[None, None, :]
        angles = angles.flatten(1, 2).tile(2)

        cos = angles.cos().view(1, 1, -1, self.head_dim)
        sin = angles.sin().view(1, 1, -1, self.head_dim)
        if batch_size > 1:
            cos = cos.expand(batch_size, -1, -1, -1)
            sin = sin.expand(batch_size, -1, -1, -1)
        return cos, sin

    def forward(self, x: torch.Tensor, cos: torch.Tensor = None, sin: torch.Tensor = None, *args, **kwargs) -> torch.Tensor:
        '''
        Applies RoPE to q / k of shape [B, H, L, D], rotating only the patch tokens after the
        prefix.

        Args:
            x (torch.Tensor): q or k, of shape [B, num_heads, L, head_dim].
            cos (torch.Tensor): Cosine table returned by ``sincos``, [B, 1, P, head_dim].
            sin (torch.Tensor): Sine table returned by ``sincos``, [B, 1, P, head_dim].

        Returns:
            torch.Tensor: The position-encoded result, with the same shape as ``x``.
        '''
        if cos is None or sin is None:
            return x
        prefix = self.num_prefix_tokens
        x_prefix, x_patch = x.split([prefix, x.shape[-2] - prefix], dim=-2)
        cos = cos.to(x_patch.dtype)
        sin = sin.to(x_patch.dtype)
        x_patch = (x_patch * cos) + (_rotate_half(x_patch) * sin)
        return torch.cat((x_prefix, x_patch), dim=-2)


class DINOv3ViTBlock(BasicModel):
    '''
    A single Transformer block of the DINOv3 ViT (pre-norm + Layer Scale).

    Attributes:
        norm1 (LayerNorm): LayerNorm before attention (eps=1e-5).
        gamma1 (nn.Parameter): Layer Scale of the attention branch.
        drop_path1 (nn.Module): DropPath of the attention branch (Identity when stochastic depth
            is disabled).
        q_proj (nn.Linear): Query projection.
        k_proj (nn.Linear): Key projection (the official DINOv3 has no bias here).
        v_proj (nn.Linear): Value projection.
        o_proj (nn.Linear): Output projection.
        norm2 (LayerNorm): LayerNorm before the MLP (eps=1e-5).
        up_proj (nn.Linear): MLP expansion projection.
        down_proj (nn.Linear): MLP contraction projection.
        act (nn.Module): GELU (the exact variant).
        gamma2 (nn.Parameter): Layer Scale of the MLP branch.
        drop_path2 (nn.Module): DropPath of the MLP branch.
    '''

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        key_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float = 1.0,
        norm_eps: float = 1e-5,
    ):
        '''
        Args:
            embed_dim (int, optional): Hidden dimension. Defaults to 768.
            num_heads (int, optional): Number of attention heads. Defaults to 12.
            ffn_ratio (float, optional): MLP hidden size as a multiple of the hidden dimension.
                Defaults to 4.0.
            qkv_bias (bool, optional): Whether the q / v projections carry a bias. Defaults to True.
            key_bias (bool, optional): Whether the k projection carries a bias. Defaults to False.
            proj_bias (bool, optional): Whether the output projection carries a bias.
                Defaults to True.
            ffn_bias (bool, optional): Whether the MLP carries a bias. Defaults to True.
            drop_path_rate (float, optional): Stochastic depth probability. Defaults to 0.0.
            layerscale_init (float, optional): Initial value of Layer Scale. Defaults to 1.0.
            norm_eps (float, optional): Eps of the LayerNorm. Defaults to 1e-5.
        '''
        super().__init__()

        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.layerscale_init = layerscale_init

        self.norm1 = LayerNorm(embed_dim, eps=norm_eps)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=key_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=proj_bias)
        self.gamma1 = nn.Parameter(layerscale_init * torch.ones(embed_dim))
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = LayerNorm(embed_dim, eps=norm_eps)
        hidden_dim = int(embed_dim * ffn_ratio)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=ffn_bias)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=ffn_bias)
        self.act = get_activation('gelu')
        self.gamma2 = nn.Parameter(layerscale_init * torch.ones(embed_dim))
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def attention(
        self,
        x: torch.Tensor,
        cos: torch.Tensor = None,
        sin: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        '''
        Bidirectional multi-head self-attention; RoPE is applied to the patch tokens only.

        Args:
            x (torch.Tensor): Normalized hidden states [B, L, D].
            cos (torch.Tensor, optional): RoPE cosine table [B, 1, P, head_dim]. Defaults to None.
            sin (torch.Tensor, optional): RoPE sine table [B, 1, P, head_dim]. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            output_attentions (bool, optional): Whether to return the attention weights.
                Defaults to False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: (attention output [B, L, D], attention
                weights or None).
        '''
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if cos is not None:
            num_patches = cos.shape[-2]
            num_prefix = seq_len - num_patches
            cos = cos.to(q.dtype)
            sin = sin.to(q.dtype)
            q_prefix, q_patch = q.split([num_prefix, num_patches], dim=-2)
            k_prefix, k_patch = k.split([num_prefix, num_patches], dim=-2)
            q = torch.cat((q_prefix, (q_patch * cos) + (_rotate_half(q_patch) * sin)), dim=-2)
            k = torch.cat((k_prefix, (k_patch * cos) + (_rotate_half(k_patch) * sin)), dim=-2)

        attn_output = apply_attention(
            q, k, v,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            is_causal=False,
        )

        output = attn_output.output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.o_proj(output), attn_output.attention_weights

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor = None,
        sin: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        output_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        '''
        Args:
            x (torch.Tensor): Hidden states [B, L, D].
            cos (torch.Tensor, optional): RoPE cosine table. Defaults to None.
            sin (torch.Tensor, optional): RoPE sine table. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            output_attentions (bool, optional): Whether to return the attention weights.
                Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple]: The hidden states; when ``output_attentions=True`` a tuple
                of (hidden states, attention weights) is returned instead.
        '''
        residual = x
        attn_out, attn_weights = self.attention(
            self.norm1(x), cos=cos, sin=sin,
            attention_mask=attention_mask, output_attentions=output_attentions,
        )
        x = residual + self.drop_path1(self.gamma1 * attn_out)

        residual = x
        mlp_out = self.down_proj(self.act(self.up_proj(self.norm2(x))))
        x = residual + self.drop_path2(self.gamma2 * mlp_out)

        if output_attentions:
            return x, attn_weights
        return x


class DINOv3ViT(BasicModel):
    '''
    codon implementation of the DINOv3 ViT (``facebook/dinov3-vit*``,
    https://github.com/facebookresearch/dinov3).

    The variant is selected through ``variant`` (vits16 / vitb16 / vitl16 / vitso400m / ...), or
    ``embed_dim`` / ``depth`` / ``num_heads`` may be passed directly. Note that the officially
    released DINOv3 weights all carry 4 register tokens, so ``n_storage_tokens=4`` is required
    when loading them.

    Attributes:
        patch_embed (nn.Conv2d): Patch embedding convolution (kernel_size = stride = patch_size).
        cls_token (nn.Parameter): CLS token [1, 1, D].
        mask_token (nn.Parameter): MAE-style mask token [1, 1, D].
        storage_tokens (nn.Parameter, optional): Register tokens [1, n, D]; not registered when
            n = 0.
        rope (DINOv3ViTRope): 2D axial RoPE (rotates the patch tokens only).
        blocks (nn.ModuleList): ``depth`` instances of ``DINOv3ViTBlock``.
        norm (LayerNorm): Final LayerNorm (eps=1e-5), applied to every token.
        head (nn.Identity): This implementation is a backbone only, so the head is the identity.
        embed_dim (int): Hidden dimension.
        n_blocks (int): Number of blocks.
        n_storage_tokens (int): Number of register/storage tokens.
        patch_size (int): Patch size.
        chunked_blocks (bool): Whether chunked (gradient checkpointing segmented) mode is used;
            False in this implementation.
        input_pad_size (int): Multiple the input must be padded to (= patch_size).
    '''

    variant = None

    def __init__(
        self,
        variant: Optional[str] = None,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        key_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float = 1.0,
        norm_eps: float = 1e-5,
        n_storage_tokens: int = 0,
        rope_base: float = 100.0,
        rope_shift_coords: Optional[float] = None,
        rope_jitter_coords: Optional[float] = None,
        rope_rescale_coords: Optional[float] = None,
        **ignored_kwargs,
    ):
        '''
        Args:
            variant (Optional[str], optional): Variant name; when given it overrides ``embed_dim``
                / ``depth`` / ``num_heads`` / ``ffn_ratio``. A subclass may also pin it with a
                class attribute. Defaults to None.
            img_size (int, optional): Training resolution, recorded only (the RoPE is computed
                dynamically from the input). Defaults to 224.
            patch_size (int, optional): Patch size. Defaults to 16.
            in_channels (int, optional): Number of input channels. Defaults to 3.
            embed_dim (int, optional): Hidden dimension. Defaults to 768.
            depth (int, optional): Number of blocks. Defaults to 12.
            num_heads (int, optional): Number of attention heads. Defaults to 12.
            ffn_ratio (float, optional): MLP hidden size ratio. Defaults to 4.0.
            qkv_bias (bool, optional): Whether the q / v projections carry a bias. Defaults to True.
            key_bias (bool, optional): Whether the k projection carries a bias. Defaults to False.
            proj_bias (bool, optional): Whether the output projection carries a bias.
                Defaults to True.
            ffn_bias (bool, optional): Whether the MLP carries a bias. Defaults to True.
            drop_path_rate (float, optional): Total stochastic depth probability, increasing
                linearly per block. Defaults to 0.0.
            layerscale_init (float, optional): Initial value of Layer Scale. Defaults to 1.0.
            norm_eps (float, optional): Eps of the LayerNorm. Defaults to 1e-5.
            n_storage_tokens (int, optional): Number of register tokens; 0 means none.
                Defaults to 0.
            rope_base (float, optional): Base frequency of the RoPE. Defaults to 100.0.
            rope_shift_coords (Optional[float], optional): Coordinate shift magnitude in training
                mode. Defaults to None.
            rope_jitter_coords (Optional[float], optional): Coordinate jitter factor in training
                mode. Defaults to None.
            rope_rescale_coords (Optional[float], optional): Coordinate rescale factor in training
                mode. Defaults to None.
            **ignored_kwargs: Other configuration entries (such as num_classes) are ignored
                outright, which makes it easy to reuse an external config.
        '''
        super().__init__()
        if len(ignored_kwargs) > 0:
            print(f'[DINOv3ViT] ignoring unrecognized arguments: {sorted(ignored_kwargs)}')

        variant = variant if variant is not None else self.variant
        if variant is not None:
            if variant not in dinov3_sizes:
                raise NotImplementedError(
                    f"didn't recognize dinov3 variant string: {variant!r}; "
                    f'choose from {list(dinov3_sizes)}'
                )
            size_dict = dinov3_sizes[variant]
            embed_dim = size_dict['embed_dim']
            depth = size_dict['depth']
            num_heads = size_dict['num_heads']
            ffn_ratio = size_dict['ffn_ratio']

        self.variant = variant
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.ffn_ratio = ffn_ratio
        self.n_storage_tokens = n_storage_tokens
        self.layerscale_init = layerscale_init

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        if n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim))

        self.rope = DINOv3ViTRope(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=rope_base,
            num_prefix_tokens=1 + n_storage_tokens,
            shift_coords=rope_shift_coords,
            jitter_coords=rope_jitter_coords,
            rescale_coords=rope_rescale_coords,
        )

        dp_rates = [float(rate) for rate in np.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            DINOv3ViTBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                key_bias=key_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path_rate=dp_rates[i],
                layerscale_init=layerscale_init,
                norm_eps=norm_eps,
            )
            for i in range(depth)
        ])

        self.norm = LayerNorm(embed_dim, eps=norm_eps)
        self.head = nn.Identity()

        self.chunked_blocks = False
        self.input_pad_size = patch_size

        self.init_weights()

    def init_weights(self) -> None:
        '''Initializes the parameters the ViT way: truncated normal (0.02) for the linear layers,
        LayerNorm weight/bias set to 1/0, and a constant Layer Scale.'''
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.trunc_normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)

        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.constant_(block.gamma1, block.layerscale_init)
            nn.init.constant_(block.gamma2, block.layerscale_init)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def prepare_tokens_with_masks(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        Splits the image into patches and prepends the CLS / storage tokens.

        Args:
            x (torch.Tensor): Images [N, C, H, W].
            masks (Optional[torch.Tensor], optional): MAE mask [N, HW]; the corresponding patch is
                replaced by ``mask_token`` wherever it is True. Defaults to None.

        Returns:
            torch.Tensor: [N, 1 + n_storage_tokens + HW, D].
        '''
        x = self.patch_embed(x)                    # [N, D, H', W']
        batch_size = x.shape[0]
        x = x.flatten(2).transpose(1, 2)           # [N, HW, D]

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype), x)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens.expand(batch_size, -1, -1)
        else:
            storage_tokens = torch.empty(batch_size, 0, self.embed_dim, dtype=x.dtype, device=x.device)

        return torch.cat([cls_token, storage_tokens, x], dim=1)

    def _decode_features(
        self,
        x: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        '''
        Runs the sequence forward pass plus normalization and produces the codon feature
        dictionary.

        Args:
            x (torch.Tensor): Images [N, C, H, W], or a large batch assembled from multiple crops.
            masks (Optional[torch.Tensor], optional): MAE mask. Defaults to None.
            output_attentions (bool, optional): Whether to return the attention weights.
                Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: See the description of ``forward_features``.
        '''
        num_patches_h = x.shape[-2] // self.patch_size
        num_patches_w = x.shape[-1] // self.patch_size
        num_patches = num_patches_h * num_patches_w
        num_prefix = 1 + self.n_storage_tokens

        x = self.prepare_tokens_with_masks(x, masks)
        cos, sin = self.rope.sincos(num_patches_h, num_patches_w, x.device, batch_size=x.shape[0])

        all_attentions = [] if output_attentions else None
        for block in self.blocks:
            if output_attentions:
                x, attn_weights = self.checkpoint(block, x, cos, sin, None, True)
                all_attentions.append(attn_weights)
            else:
                x = self.checkpoint(block, x, cos, sin, None, False)

        x_norm = self.norm(x)
        features = {
            'x_norm_clstoken': x_norm[:, 0],
            'x_storage_tokens': x_norm[:, 1: num_prefix],
            'x_norm_patchtokens': x_norm[:, num_prefix:],
            'x_norm_alltokens': x_norm,
            'x_prenorm': x,
            'x_prenorm_patchtokens': x[:, num_prefix:],
            'masks': masks,
        }
        if output_attentions:
            features['attentions'] = all_attentions
        return features

    def forward_features(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        masks: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        output_attentions: bool = False,
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        '''
        Extracts the backbone features.

        Args:
            x (Union[torch.Tensor, List[torch.Tensor]]): Images [N, C, H, W], or a list of image
                crops (every image in the list is forwarded independently, so different
                resolutions are allowed).
            masks (Optional[Union[torch.Tensor, List[torch.Tensor]]], optional): MAE mask whose
                type matches ``x``. Defaults to None.
            output_attentions (bool, optional): Whether to also include the per-layer attention
                weights. Defaults to False.

        Returns:
            Union[Dict, List[Dict]]: A single input returns one dictionary (a list input returns a
                list of the same length) with the keys:

                - `x_norm_clstoken`       [N, C]      CLS token after LayerNorm
                - `x_storage_tokens`      [N, R, C]   register tokens after LayerNorm (R = n_storage_tokens)
                - `x_norm_patchtokens`    [N, HW, C]  patch tokens after LayerNorm
                - `x_norm_alltokens`      [N, L, C]   full sequence after LayerNorm (CLS + register + patch)
                - `x_prenorm`             [N, L, C]   full sequence before normalization (as in DINOv3 upstream)
                - `x_prenorm_patchtokens` [N, HW, C]  patch tokens before normalization
                - `masks`                              the masks passed through
                - `attentions`                        present only when `output_attentions=True`
        '''
        if isinstance(x, torch.Tensor):
            return self._decode_features(x, masks, output_attentions=output_attentions)

        mask_list = [None] * len(x) if masks is None else masks
        return [
            self._decode_features(t_x, t_masks, output_attentions=output_attentions)
            for t_x, t_masks in zip(x, mask_list)
        ]

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        is_training: bool = False,
        masks: Optional[torch.Tensor] = None,
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]], torch.Tensor]:
        '''
        Args:
            x (Union[torch.Tensor, List[torch.Tensor]]): Images [N, C, H, W], or a list of images.
            is_training (bool, optional): True returns the dictionary from ``forward_features``,
                False returns only the CLS features. Defaults to False.
            masks (Optional[torch.Tensor], optional): Passed through to ``forward_features``.
                Defaults to None.

        Returns:
            Union[Dict, List[Dict], torch.Tensor]: The feature dictionary when training,
                otherwise the [N, embed_dim] CLS features.
        '''
        ret = self.forward_features(x, masks=masks)
        if is_training:
            return ret
        if isinstance(ret, list):
            raise ValueError(
                'forward(is_training=False) only supports a single image; for multi-crop input '
                'use forward_features(...) to get x_norm_clstoken per crop, or pass '
                'is_training=True'
            )
        return self.head(ret['x_norm_clstoken'])

    def _get_intermediate_layers_not_chunked(
        self, x: torch.Tensor, n: Union[int, Sequence] = 1
    ) -> List[torch.Tensor]:
        '''
        Runs the blocks one by one and collects the outputs of the requested layers (without the
        final norm).

        Args:
            x (torch.Tensor): Images [N, C, H, W].
            n (Union[int, Sequence], optional): Take the last n layers, or pass the layer indices
                directly. Defaults to 1.

        Returns:
            List[torch.Tensor]: One [N, L, D] tensor per layer.
        '''
        num_patches_h = x.shape[-2] // self.patch_size
        num_patches_w = x.shape[-1] // self.patch_size

        x = self.prepare_tokens_with_masks(x)
        cos, sin = self.rope.sincos(num_patches_h, num_patches_w, x.device, batch_size=x.shape[0])

        blocks_to_take = list(range(self.n_blocks - n, self.n_blocks)) if isinstance(n, int) else list(n)
        output = []
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin)
            if i in blocks_to_take:
                output.append(x)

        assert len(output) == len(blocks_to_take), (
            f'only {len(output)} / {len(blocks_to_take)} blocks found'
        )
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple:
        '''
        Returns intermediate-layer features.

        Args:
            x (torch.Tensor): Images [N, C, H, W].
            n (Union[int, Sequence], optional): Take the last n layers, or pass the layer indices
                directly. Defaults to 1.
            reshape (bool, optional): When True, patch tokens are restored to (N, C, H', W').
                Defaults to False.
            return_class_token (bool, optional): When True, every layer also carries the CLS token.
                Defaults to False.
            return_extra_tokens (bool, optional): When True, every layer also carries the register
                tokens. Defaults to False.
            norm (bool, optional): Whether to apply the final LayerNorm. Defaults to True.

        Returns:
            Tuple: One tensor per layer; a tuple in the order (patch, cls[, extra]) instead when
                ``return_class_token`` / ``return_extra_tokens`` is enabled.
        '''
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        num_prefix = 1 + self.n_storage_tokens

        if norm:
            outputs = [self.norm(out) for out in outputs]

        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1:num_prefix] for out in outputs]
        outputs = [out[:, num_prefix:] for out in outputs]

        if reshape:
            batch_size, _, height, width = x.shape
            outputs = [
                out.reshape(batch_size, height // self.patch_size, width // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]

        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        if return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        if not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        return tuple(zip(outputs, class_tokens, extra_tokens))

    def load_pretrained(
        self,
        path: str,
        strict: bool = True,
        remap: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> 'DINOv3ViT':
        '''
        Loads pretrained weights from a local file (.safetensors / .pth / .pt).

        Args:
            path (str): Path to the weight file.
            strict (bool, optional): When True, key names and shapes must match the model exactly.
                Defaults to True.
            remap (bool, optional): When True, ``remap_dinov3_keys`` is applied first to map the
                key names of transformers / official DINOv3 onto this implementation's naming.
                Defaults to True.
            dtype (Optional[torch.dtype], optional): Target precision of the weights; None follows
                the current parameter precision of the model. Defaults to None.

        Returns:
            DINOv3ViT: self.

        Raises:
            RuntimeError: If the weights are not a valid dict, if some shape does not match, or if
                a precision change is requested but the target is not a floating point type.
        '''
        if path.endswith('.safetensors'):
            tensors = dict(load_file(path, device='cpu'))
        else:
            raw = torch.load(path, map_location='cpu')
            if isinstance(raw, dict):
                for wrapper in ('model_state_dict', 'state_dict', 'model'):
                    sub = raw.get(wrapper)
                    if isinstance(sub, dict):
                        raw = sub
                        break
            if not isinstance(raw, dict):
                raise RuntimeError(f'{path} is not a valid PyTorch weight file (expected a dict)')
            tensors = {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}

        if remap:
            tensors = remap_dinov3_keys(tensors)

        ref = self.state_dict()
        extra = sorted(k for k in tensors if k not in ref)
        missing = sorted(k for k in ref if k not in tensors)

        for key in list(missing):
            if key.endswith('mask_token'):
                missing.remove(key)

        if strict and (extra or missing):
            raise RuntimeError(
                f'weights do not match {type(self).__name__}({self.variant}): '
                f'{len(extra)} extra / {len(missing)} missing -> '
                f'extra {extra[:5]}..., missing {missing[:5]}...'
            )

        if dtype is None:
            try:
                dtype = next(iter(self.parameters())).dtype
            except StopIteration:
                dtype = torch.float32

        for key in list(tensors):
            if key not in ref:
                tensors.pop(key)
                continue
            want = tuple(ref[key].shape)
            got = tuple(tensors[key].shape)
            if got != want:
                raise RuntimeError(f'shape mismatch for {key}: weights {got} vs model {want}')

            source_dtype = tensors[key].dtype
            if source_dtype == dtype:
                continue
            if not dtype.is_floating_point:
                raise RuntimeError(
                    f'cannot cast {key} from {source_dtype} to non-float {dtype}'
                )
            tensors[key] = tensors[key].to(dtype)

        self.load_state_dict(tensors, strict=False)
        return self

    __remote_resource__ = {
        'repo': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        'files': ['config.json', 'model.safetensors'],
        'repo_type': 'model',
    }
    @classmethod
    def from_remote(cls, dtype: Optional[torch.dtype] = None, **kwargs) -> 'DINOv3ViT':
        '''
        Pulls ``config.json`` and ``model.safetensors`` from a remote repository (ModelScope /
        HuggingFace, routed automatically), builds the model from the config and then transfers
        the weights.

        How ``dtype`` is handled: the ``dtype`` (or ``torch_dtype``) entry of the config decides
        the storage precision of the weights. The default is to "follow the config" -- for
        instance ``CodonProject/DINOv3-ViT-Base`` is fp16, so the model is cast to fp16 before
        loading and no VRAM is wasted; passing ``dtype=torch.float32`` explicitly instead
        promotes the fp16 weights back to fp32 for computation.

        Args:
            dtype (Optional[torch.dtype], optional): Target precision. None follows the ``dtype``
                / ``torch_dtype`` field of the config (fp32 when neither is present).
                Defaults to None.
            **kwargs: Passed through to ``cls.__init__`` (overriding the same-named fields
                derived from the config).

        Returns:
            DINOv3ViT: The instance with the weights loaded.
        '''
        import json

        from codon.builtin.repo import Repo

        cfg = getattr(cls, '__remote_resource__', None)
        if cfg is None:
            raise ValueError(f'No remote resource configured for {cls.__name__}.')

        repo = Repo(
            modelscope=cfg,
            huggingface=cfg,
            repo_type=cfg.get('repo_type', 'model'),
        )
        local_paths = repo.download_configured_files()
        by_name = {os.path.basename(p): p for p in local_paths}

        config = {}
        if 'config.json' in by_name:
            with open(by_name['config.json'], 'r', encoding='utf-8') as f:
                config = json.load(f)

        config_kwargs = {
            'img_size': config.get('image_size', 224),
            'patch_size': config.get('patch_size', 16),
            'in_channels': config.get('num_channels', 3),
            'embed_dim': config.get('hidden_size', 768),
            'depth': config.get('num_hidden_layers', 12),
            'num_heads': config.get('num_attention_heads', 12),
            'ffn_ratio': config.get('intermediate_size', 3072) / max(config.get('hidden_size', 768), 1),
            'qkv_bias': config.get('query_bias', True),
            'key_bias': config.get('key_bias', False),
            'proj_bias': config.get('proj_bias', True),
            'ffn_bias': config.get('mlp_bias', True),
            'n_storage_tokens': config.get('num_register_tokens', 0),
            'norm_eps': config.get('layer_norm_eps', 1e-5),
            'layerscale_init': config.get('layerscale_value', 1.0),
            'rope_base': config.get('rope_theta', 100.0),
            'rope_shift_coords': config.get('pos_embed_shift'),
            'rope_jitter_coords': config.get('pos_embed_jitter'),
            'rope_rescale_coords': config.get('pos_embed_rescale'),
        }
        config_kwargs.update(kwargs)

        weight_dtype = dtype if dtype is not None else _parse_config_dtype(config)

        model = cls(**config_kwargs)
        if weight_dtype is not None and weight_dtype != torch.float32:
            model = model.to(weight_dtype)

        weight_file = by_name.get('model.safetensors')
        if weight_file is None:
            raise FileNotFoundError(
                f'no model.safetensors in the remote repository, only found {sorted(by_name)}'
            )
        model.load_pretrained(weight_file, strict=True, dtype=weight_dtype)
        return model


class DINOv3ViT_Small(DINOv3ViT):
    '''DINOv3 ViT-S/16 (embed_dim=384, depth=12, num_heads=6).'''

    variant = 'vits16'

    __remote_resource__ = {
        'repo': 'facebook/dinov3-vits16-pretrain-lvd1689m',
        'files': ['config.json', 'model.safetensors'],
        'repo_type': 'model',
    }

    def __init__(self, **kwargs):
        super().__init__(variant='vits16', **kwargs)


class DINOv3ViT_Base(DINOv3ViT):
    '''
    DINOv3 ViT-B/16 (embed_dim=768, depth=12, num_heads=12).

    The default remote repository holds the fp16 weights converted by codon
    (``CodonProject/DINOv3-ViT-Base``, whose key names already use the codon layout and include 4
    register tokens), so ``DINOv3ViT_Base.from_remote()`` works out of the box:

        >>> model = DINOv3ViT_Base.from_remote().eval()          # fp16, follows the config
        >>> model = DINOv3ViT_Base.from_remote(dtype=torch.float32)   # fp32 also works

    To use Meta's original weights, simply point ``__remote_resource__`` at
    ``facebook/dinov3-vitb16-pretrain-lvd1689m`` (``remap_dinov3_keys`` converts the key names
    automatically).
    '''

    variant = 'vitb16'

    __remote_resource__ = {
        'repo': 'CodonProject/DINOv3-ViT-Base',
        'files': ['config.json', 'model.safetensors'],
        'repo_type': 'model',
    }

    __origin_resource__ = {
        'repo': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        'files': ['config.json', 'model.safetensors'],
        'repo_type': 'model',
    }

    def __init__(self, **kwargs):
        super().__init__(variant='vitb16', **kwargs)


class DINOv3ViT_Large(DINOv3ViT):
    '''DINOv3 ViT-L/16 (embed_dim=1024, depth=24, num_heads=16).'''

    variant = 'vitl16'

    __remote_resource__ = {
        'repo': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
        'files': ['config.json', 'model.safetensors'],
        'repo_type': 'model',
    }

    def __init__(self, **kwargs):
        super().__init__(variant='vitl16', **kwargs)


class DINOv3ViT_So400m(DINOv3ViT):
    '''DINOv3 ViT-SO400M/16 (embed_dim=1152, depth=27, num_heads=18).'''

    variant = 'vitso400m'

    __remote_resource__ = {
        'repo': 'facebook/dinov3-vitso400m-pretrain-lvd1689m',
        'files': ['config.json', 'model.safetensors'],
        'repo_type': 'model',
    }

    def __init__(self, **kwargs):
        super().__init__(variant='vitso400m', **kwargs)


class DINOv3ViT_HugePlus(DINOv3ViT):
    '''DINOv3 ViT-H+/16 (embed_dim=1280, depth=32, num_heads=20).'''

    variant = 'vith16plus'

    __remote_resource__ = {
        'repo': 'facebook/dinov3-vith16plus-pretrain-lvd1689m',
        'files': ['config.json', 'model.safetensors'],
        'repo_type': 'model',
    }

    def __init__(self, **kwargs):
        super().__init__(variant='vith16plus', **kwargs)


class DINOv3ViT_7B(DINOv3ViT):
    '''DINOv3 ViT-7B/16 (embed_dim=4096, depth=40, num_heads=32).'''

    variant = 'vit7b16'

    __remote_resource__ = {
        'repo': 'facebook/dinov3-vit7b16-pretrain-lvd1689m',
        'files': ['config.json', 'model.safetensors'],
        'repo_type': 'model',
    }

    def __init__(self, **kwargs):
        super().__init__(variant='vit7b16', **kwargs)


_STRIP_PREFIXES = ('module.', 'backbone.', 'model.')
_IGNORED_KEYS = ('embeddings.mask_token', 'mask_token')
_HEAD_PREFIXES = ('head.', 'classifier.')


def _strip_prefixes(key: str) -> str:
    '''Strips the prefixes left behind by DataParallel / top-level containers.'''
    while True:
        for prefix in _STRIP_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        else:
            return key


def _remap_one_key(key: str) -> Optional[str]:
    '''Maps one weight key onto the codon naming; returning None drops it.'''
    key = _strip_prefixes(key)

    if key.startswith(_IGNORED_KEYS) or key.startswith(_HEAD_PREFIXES):
        return None

    if key.startswith('embeddings.'):
        head, _, tail = key[len('embeddings.'):].partition('.')
        if head in ('patch_embeddings', 'patch_embed'):
            return 'patch_embed.' + tail
        if head == 'register_tokens':
            return 'storage_tokens'
        return head + ('.' + tail if tail else '')

    if key.startswith('layer.'):
        # layer.{i}.attention.{q,k,v,o}_proj.*  -> blocks.{i}.{q,k,v,o}_proj.*
        # layer.{i}.layer_scale{1,2}.lambda1    -> blocks.{i}.gamma{1,2}
        # layer.{i}.mlp.{up,down}_proj.*        -> blocks.{i}.{up,down}_proj.*
        _, index, kind, rest = key.split('.', 3)
        if kind == 'attention':
            return f'blocks.{index}.' + rest
        if kind.startswith('layer_scale'):
            scale_id = kind[len('layer_scale'):]
            return f'blocks.{index}.gamma{scale_id}'
        if kind == 'mlp':
            return f'blocks.{index}.' + rest
        return f'blocks.{index}.' + kind + ('.' + rest if rest else '')

    if key.startswith('blocks.'):
        return key.replace('.layer_scale1.', '.gamma1.').replace('.layer_scale2.', '.gamma2.')
    if key.startswith('norm.'):
        return key
    if key.startswith('patch_embed.'):
        return key
    if key.startswith('rope_embeddings.') or key.startswith('rope.'):
        return None
    if key.startswith('cls_token') or key.startswith('storage_tokens'):
        return key

    return key


def remap_dinov3_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    '''
    Maps the key names of common DINOv3 ViT weight files onto this implementation's codon naming.

    Args:
        state_dict (Dict[str, torch.Tensor]): The original weight dictionary.

    Returns:
        Dict[str, torch.Tensor]: A new dictionary with remapped key names (the input is not
            modified).
    '''
    remapped = {}
    for key, value in state_dict.items():
        new_key = _remap_one_key(key)
        if new_key is not None:
            remapped[new_key] = value
    return remapped
