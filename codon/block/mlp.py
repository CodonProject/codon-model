from codon import *
from codon.block.activation import SwiGLU, get_activation
from codon.block.modulation import Affine


class MLP(BasicModel):
    '''
    Multilayer Perceptron (MLP) module.

    Supports standard MLP and Gated MLP architectures.

    Attributes:
        fc1 (nn.Linear): First linear layer (used in standard MLP).
        fc2 (nn.Linear): Second linear layer (used in standard MLP).
        gate_proj (nn.Linear): Gating linear layer (used in Gated MLP).
        up_proj (nn.Linear): Up-projection linear layer (used in Gated MLP).
        down_proj (nn.Linear): Down-projection linear layer (used in Gated MLP).
        act (nn.Module): Activation function (SiLU).
        dropout (nn.Dropout): Dropout layer.
    '''
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        bias: bool = True,
        use_gate: bool = False,
        dropout: float = 0.0,
        act_layer: str = 'silu',
    ):
        '''
        Initialize the MLP module.

        Args:
            in_features (int): Dimension of input features.
            hidden_features (int): Dimension of hidden layer features.
            out_features (int, optional): Dimension of output features. If None, it defaults to in_features. Defaults to None.
            bias (bool, optional): Whether to use bias in linear layers. Defaults to True.
            use_gate (bool, optional): Whether to use the gating mechanism. Defaults to False.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            act_layer (str, optional): Activation function name (e.g. 'silu', 'gelu'). Defaults to 'silu'.
        '''
        super().__init__()
        
        out_features = out_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.bias = bias
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        
        act_layer_lower = act_layer.lower()
        self.act = get_activation(act_layer_lower)
        self.use_gate = use_gate
        
        if act_layer_lower == 'swiglu':
            self.gate_up_proj = nn.Linear(in_features, 2 * hidden_features, bias=bias)
            self.down_proj = nn.Linear(hidden_features, out_features, bias=bias)
        elif self.use_gate:
            self.gate_proj = nn.Linear(in_features, hidden_features, bias=bias)
            self.up_proj = nn.Linear(in_features, hidden_features, bias=bias)
            self.down_proj = nn.Linear(hidden_features, out_features, bias=bias)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
            self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        '''
        if isinstance(self.act, SwiGLU):
            x = self.gate_up_proj(x)
            x = self.act(x)
            x = self.down_proj(x)
            return x
        elif self.use_gate:
            return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
        else:
            x = self.fc1(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    @staticmethod
    def SwiGLU(
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> 'MLP':
        '''
        Factory method to create a SwiGLU MLP module.
        
        SwiGLU formulation:
            output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
            
        Args:
            in_features (int): Input dimension.
            hidden_features (int): Intermediate dimension (gate & up proj).
            out_features (int, optional): Output dimension. Defaults to in_features.
            bias (bool, optional): Whether to use bias. LLMs typically set False. Defaults to False.
            dropout (float, optional): Dropout rate. Usually 0.0 for SwiGLU. Defaults to 0.0.
            
        Returns:
            MLP: Configured SwiGLU module.
        '''
        if hidden_features is None:
            h = int(in_features * 8 / 3)
            hidden_features = (h + 127) // 128 * 128
            
        return MLP(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
            use_gate=True,
            dropout=dropout,
            act_layer='swiglu'
        )


class MLPMixer(nn.Module):
    def __init__(
        self,
        tokens_mlp_dim: int = 16,
        channels_mlp_dim: int = 1024,
        tokens_hidden_dim: int = 32,
        channels_hidden_dim: int = 1024,
        use_gate: bool = False,
        act_layer: str = 'gelu',
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(channels_mlp_dim)

        self.tokens_mlp = MLP(
            in_features=tokens_mlp_dim,
            hidden_features=tokens_hidden_dim,
            out_features=tokens_mlp_dim,
            bias=bias,
            use_gate=use_gate,
            dropout=dropout,
            act_layer=act_layer,
        )

        self.channels_mlp = MLP(
            in_features=channels_mlp_dim,
            hidden_features=channels_hidden_dim,
            out_features=channels_mlp_dim,
            bias=bias,
            use_gate=use_gate,
            dropout=dropout,
            act_layer=act_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: (bs, tokens, channels)
        '''
        residual = x
        out = self.ln(x)
        out = out.transpose(1, 2)
        out = self.tokens_mlp(out)
        out = out.transpose(1, 2)
        out = residual + out

        residual = out
        out = self.ln(out)
        out = self.channels_mlp(out)
        out = residual + out

        return out


class ResMLP(BasicModel):
    def __init__(
        self,
        image_size=14,
        patch_size=7,
        dim=128,
        depth=4,
        expansion_factor=4,
        use_swiglu=False,
        dropout=0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * 3

        self.embedding = nn.Linear(self.patch_dim, dim)

        self.layers = nn.ModuleList()
        for i in range(depth):
            token_mix_block = nn.ModuleDict({
                'affine': Affine(dim),
                'linear': nn.Linear(self.num_patches, self.num_patches, bias=False),
                'scale': nn.Parameter(torch.zeros(1, 1, dim).fill_(0.1 if i < 18 else 1e-5)),
            })
            self.layers.append(token_mix_block)

            channel_mix_block = nn.ModuleDict({
                'affine': Affine(dim),
                'mlp': MLP(
                    in_features=dim,
                    hidden_features=int(dim * expansion_factor),
                    out_features=dim,
                    use_gate=use_swiglu,
                    dropout=dropout,
                    act_layer='swiglu' if use_swiglu else 'silu',
                ),
                'scale': nn.Parameter(torch.zeros(1, 1, dim).fill_(0.1 if i < 18 else 1e-5)),
            })
            self.layers.append(channel_mix_block)

        self.final_affine = Affine(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        
        x = x.unfold(2, p, p).unfold(3, p, p)          # [B, C, nh, nw, p, p]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()   # [B, nh, nw, C, p, p]
        x = x.view(B, -1, self.patch_dim)              # [B, N, patch_dim]
        x = self.embedding(x)                          # [B, N, D]

        for block in self.layers:
            if 'linear' in block:               # token-mixing
                y = block['affine'](x)          # [B, N, D]
                y = y.transpose(1, 2)           # [B, D, N]
                y = block['linear'](y)          # [B, D, N]
                y = y.transpose(1, 2)           # [B, N, D]
                x = x + block['scale'] * y
            else:                               # channel-mixing
                y = block['affine'](x)          # [B, N, D]
                y = block['mlp'](y)             # [B, N, D]
                x = x + block['scale'] * y

        x = self.final_affine(x).mean(dim=1) # [B, D]
        return x



class SMLP(nn.Module):
    def __init__(
        self,
        h=224,
        w=224,
        c=3,
        expand_ratio=4,
        dropout=0.1,
        dynamic_res=True
    ):
        super().__init__()
        self.h = h
        self.w = w
        self.c = c
        self.dynamic_res = dynamic_res

        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)

        self.proj_h = nn.Linear(h, h)
        self.proj_w = nn.Linear(w, w)

        hidden_dim = c * expand_ratio
        self.channel_mlp = nn.Sequential(
            nn.Linear(c, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, c),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.dynamic_res and (H != self.h or W != self.w):
            x = F.interpolate(x, size=(self.h, self.w), mode='bilinear', align_corners=False)

        residual = x
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

        x = self.norm1(x)

        x_h = x.permute(0, 2, 1, 3)    # (B, W, H, C)
        x_h = self.proj_h(x_h)         # (B, W, H, C)
        x_h = x_h.permute(0, 2, 1, 3)  # (B, H, W, C)

        x_w = self.proj_w(x)           # (B, H, W, C)

        x = x_h + x_w

        x = x.permute(0, 3, 1, 2)    # (B, C, H, W)
        x = residual + x

        residual = x
        x = x.permute(0, 2, 3, 1)    # (B, H, W, C)
        x = self.norm2(x)
        x = self.channel_mlp(x)
        x = x.permute(0, 3, 1, 2)    # (B, C, H, W)
        x = residual + x

        return x