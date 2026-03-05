from codon.base import *


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
        dropout: float = 0.0
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
        '''
        super().__init__()
        
        out_features = out_features or in_features

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.bias = bias
        self.use_gate = use_gate
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        
        if use_gate:
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
        if self.use_gate:
            return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
        else:
            x = self.fc1(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            return x
