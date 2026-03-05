from codon.base import *
from typing     import List, Optional

from codon.block.conv import CausalConv1d

class TemporalConvNet(BasicModel):
    '''
    Temporal Convolutional Network (TCN).
    
    Consists of a series of Causal Dilated Convolution layers.
    Supports manually specifying the number of channels for each layer or automatically building based on the target receptive field.
    '''

    def __init__(
        self,
        in_channels: int,
        num_channels: Optional[List[int]] = None,
        out_channels: Optional[int] = None,
        step: Optional[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        use_res: bool = True,
        norm: str = None,
        activation: str = 'leaky_relu',
        leaky_relu: float = 0.1
    ):
        '''
        Initializes the TCN module.

        Args:
            in_channels (int): Number of input channels.
            num_channels (List[int], optional): List of output channels for each layer. If provided, `out_channels` and `step` are ignored.
            out_channels (int, optional): Unified output channels in auto-build mode.
            step (int, optional): Target receptive field (time steps) in auto-build mode.
            kernel_size (int, optional): Kernel size. Defaults to 3.
            dropout (float, optional): Dropout probability. Defaults to 0.2.
            use_res (bool, optional): Whether to use residual connections. Defaults to True.
            norm (str, optional): Normalization type (passed to CausalConv1d/ConvBlock). Defaults to None.
            activation (str, optional): Activation function type. Defaults to 'leaky_relu'.
            leaky_relu (float, optional): Negative slope for LeakyReLU. Defaults to 0.1.
        '''
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        
        if num_channels is not None:
            layers = []
            num_levels = len(num_channels)
            for i in range(num_levels):
                dilation_size = 2 ** i
                in_ch = in_channels if i == 0 else num_channels[i-1]
                out_ch = num_channels[i]
                
                layers.append(CausalConv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation_size,
                    norm=norm,
                    activation=activation,
                    leaky_relu=leaky_relu,
                    use_res=use_res,
                    dropout=dropout
                ))
            self.network = nn.Sequential(*layers)
            self.out_channels = num_channels[-1]
            
        elif step is not None and out_channels is not None:
            self.network = CausalConv1d.auto_block(
                in_channels=in_channels,
                out_channels=out_channels,
                step=step,
                kernel_size=kernel_size,
                norm=norm,
                activation=activation,
                leaky_relu=leaky_relu,
                use_res=use_res,
                dropout=dropout
            )
            self.out_channels = out_channels
        else:
            raise ValueError("Must provide either 'num_channels' (list) or both 'step' and 'out_channels' (int).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor. Shape: [Batch, in_channels, Seq_Len]

        Returns:
            torch.Tensor: Output tensor. Shape: [Batch, out_channels, Seq_Len]
        '''
        return self.network(x)
