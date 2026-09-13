from codon import *
from codon.block.conv import CausalConv1d, calculate_causal_layer


class TemporalConvNet(BasicModel):
    '''
    Temporal Convolutional Network.
    
    Consists of a series of Causal Dilated Convolution layers.
    Supports manually specifying the number of channels for each layer.
    Use `TemporalConvNet.auto_build` for automatic construction based on the target receptive field.
    '''

    def __init__(
        self,
        in_channels: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        use_res: bool = True,
        norm: str = None,
        activation: str = 'leaky_relu',
        leaky_relu: float = 0.1,
        channel_first: bool = True
    ):
        '''
        Initializes the TCN module manually.

        Args:
            in_channels (int): Number of input channels.
            num_channels (List[int]): List of output channels for each layer.
            kernel_size (int, optional): Kernel size. Defaults to 3.
            dropout (float, optional): Dropout probability. Defaults to 0.2.
            use_res (bool, optional): Whether to use residual connections. Defaults to True.
            norm (str, optional): Normalization type (passed to CausalConv1d/ConvBlock). Defaults to None.
            activation (str, optional): Activation function type. Defaults to 'leaky_relu'.
            leaky_relu (float, optional): Negative slope for LeakyReLU. Defaults to 0.1.
            channel_first (bool, optional): Whether input is (Batch, Channels, Seq_Len). Defaults to True.
        '''
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.channel_first = channel_first
        
        self.network = CausalConv1d.manual_block(
            in_channels=in_channels,
            num_channels=num_channels,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
            leaky_relu=leaky_relu,
            use_res=use_res,
            dropout=dropout
        )
        self.out_channels = num_channels[-1]

    @staticmethod
    def auto_build(
        in_channels: int,
        out_channels: int,
        receptive_field: int,
        kernel_size: int = 3,
        dropout: float = 0.2,
        use_res: bool = True,
        norm: str = None,
        activation: str = 'leaky_relu',
        leaky_relu: float = 0.1,
        channel_first: bool = True
    ) -> 'TemporalConvNet':
        '''
        Automatically builds a TCN module based on the target receptive field.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Unified output channels for each layer.
            receptive_field (int): Target receptive field (time steps).
            kernel_size (int, optional): Kernel size. Defaults to 3.
            dropout (float, optional): Dropout probability. Defaults to 0.2.
            use_res (bool, optional): Whether to use residual connections. Defaults to True.
            norm (str, optional): Normalization type. Defaults to None.
            activation (str, optional): Activation function type. Defaults to 'leaky_relu'.
            leaky_relu (float, optional): Negative slope for LeakyReLU. Defaults to 0.1.
            channel_first (bool, optional): Whether input is (Batch, Channels, Seq_Len). Defaults to True.

        Returns:
            TemporalConvNet: An initialized TCN module.
        '''
        layers, _ = calculate_causal_layer(receptive_field, kernel_size)
        num_channels = [out_channels] * layers
        
        return TemporalConvNet(
            in_channels=in_channels,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            use_res=use_res,
            norm=norm,
            activation=activation,
            leaky_relu=leaky_relu,
            channel_first=channel_first
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor. Shape: [Batch, in_channels, Seq_Len] or [Batch, Seq_Len, in_channels] if channel_first=False.

        Returns:
            torch.Tensor: Output tensor.
        '''
        if not self.channel_first:
            x = x.transpose(1, 2)
            
        x = self.network(x)
        
        if not self.channel_first:
            x = x.transpose(1, 2)
            
        return x
