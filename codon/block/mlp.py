from codon import *
from codon.block.activation import SwiGLU


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
        if act_layer_lower == 'swiglu':
            self.act = SwiGLU(dim=-1)
            self.use_gate = True 
        elif act_layer_lower == 'silu':
            self.act = nn.SiLU()
            self.use_gate = use_gate
        elif act_layer_lower == 'gelu':
            self.act = nn.GELU()
            self.use_gate = use_gate
        elif act_layer_lower == 'relu':
            self.act = nn.ReLU()
            self.use_gate = use_gate
        else:
            raise NotImplementedError(f'Activation {act_layer} not implemented')
        
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