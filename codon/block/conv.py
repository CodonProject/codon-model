import torch.nn.functional as F
import math

from codon.base import *
from typing     import Tuple, Union


def calculate_causal_layer(step: int, kernel_size: int = 3) -> Tuple[int, int]:
    '''
    Calculates the required number of layers and receptive field for causal convolution.

    Args:
        step (int): Target sequence length or number of time steps.
        kernel_size (int, optional): Kernel size. Defaults to 3.

    Returns:
        tuple[int, int]:
            - L (int): Required number of layers.
            - R (int): Final receptive field size.

    Raises:
        ValueError: If kernel_size <= 1.
    '''
    if kernel_size <= 1:
        raise ValueError('kernel_size must be greater than 1')
    L = math.ceil(math.log2((step - 1) / (kernel_size - 1) + 1))
    R = 1 + (kernel_size - 1) * (2 ** L - 1)
    return int(L), R


class CausalConv1d(BasicModel):
    '''
    Causal 1D Convolution layer.
    
    Implemented via Dilated Convolution and Causal Padding, ensuring the output 
    at the current time step depends only on current and past inputs, not the future.
    Commonly used for time-series data processing and waveform generation (e.g., WaveNet).

    Attributes:
        block (ConvBlock): The main convolution block.
        downsample (ConvBlock, optional): The downsampling layer for residual connection.
        padding (int): The amount of padding applied.
        use_res (bool): Whether to use residual connection.
    '''

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        dilation: int = 1, 
        norm: str = None,
        activation: str = 'leaky_relu',
        leaky_relu: float = 0.1, 
        use_res: bool = True, 
        dropout: float = 0.2
    ):
        '''
        Initializes the CausalConv1d module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int, optional): Kernel size. Defaults to 3.
            dilation (int, optional): Dilation factor. Defaults to 1.
            norm (str, optional): Normalization type. Defaults to None.
            activation (str, optional): Activation function type. Defaults to 'leaky_relu'.
            leaky_relu (float, optional): Negative slope for LeakyReLU (only valid when activation='leaky_relu'). Defaults to 0.1.
            use_res (bool, optional): Whether to use residual connection. Defaults to True.
            dropout (float, optional): Dropout probability. Defaults to 0.2.
        '''
        super().__init__()
        
        self.padding = (kernel_size - 1) * dilation
        self.use_res = use_res
        
        self.block = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            dim=1,
            norm=norm,
            activation=activation,
            dropout=dropout
        )
        
        if activation == 'leaky_relu' and isinstance(self.block.act, nn.LeakyReLU) and leaky_relu != 0.1:
            self.block.act = nn.LeakyReLU(leaky_relu, inplace=True)

        self.block.conv = nn.utils.parametrizations.weight_norm(self.block.conv)
        
        self.downsample = None
        if use_res and in_channels != out_channels:
            self.downsample = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                dim=1,
                norm=None,
                activation=None,
                dropout=0.0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor. Shape: [Batch, in_channels, Seq_Len]

        Returns:
            torch.Tensor: Output tensor. Shape: [Batch, out_channels, Seq_Len]
        '''
        residual = x
        x = F.pad(x, (self.padding, 0))
        x = self.block(x)
        
        if self.use_res:
            if self.downsample is not None:
                residual = self.downsample(residual)
            x = x + residual
            
        return x
    
    @staticmethod
    def auto_block(
        in_channels: int, 
        out_channels: int, 
        step: int, 
        kernel_size: int = 3, 
        norm: str = None,
        activation: str = 'leaky_relu',
        leaky_relu: float = 0.1, 
        use_res: bool = True, 
        dropout: float = 0.2
    ) -> nn.Sequential:
        '''
        Automatically builds multiple causal convolution blocks to cover the specified time steps.

        Automatically calculates the required number of layers and dilation factors based on the target step,
        constructing an nn.Sequential model. Dilation factors grow exponentially with layers (1, 2, 4, 8, ...).

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            step (int): Target time steps to cover (receptive field).
            kernel_size (int, optional): Kernel size. Defaults to 3.
            norm (str, optional): Normalization type. Defaults to None.
            activation (str, optional): Activation function type. Defaults to 'leaky_relu'.
            leaky_relu (float, optional): Negative slope for LeakyReLU. Defaults to 0.1.
            use_res (bool, optional): Whether to use residual connection. Defaults to True.
            dropout (float, optional): Dropout probability. Defaults to 0.2.

        Returns:
            nn.Sequential: A sequential model containing multiple CausalConv1d layers.
        '''
        layers, _ = calculate_causal_layer(step, kernel_size)
        model = []
        for i in range(layers):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else out_channels

            model.append(CausalConv1d(
                in_channels=in_ch, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                dilation=dilation, 
                norm=norm,
                activation=activation,
                leaky_relu=leaky_relu, 
                use_res=use_res, 
                dropout=dropout
            ))

        return nn.Sequential(*model)


class ConvBlock(BasicModel):
    '''
    General Convolution Block (Conv-Norm-Act-Dropout).

    Supports 1D, 2D, and 3D convolutions, as well as various normalization and activation function configurations.

    Attributes:
        conv (nn.Module): The convolution layer (Conv1d, Conv2d, or Conv3d).
        norm (nn.Module, optional): The normalization layer.
        act (nn.Module, optional): The activation function.
        dropout (nn.Dropout, optional): The dropout layer.
        pre_norm (bool): Whether to use Pre-Norm structure.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...], str] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = True,
        dim: int = 2,
        norm: str = 'batch',
        activation: str = 'relu',
        dropout: float = 0.0,
        pre_norm: bool = False,
    ):
        '''
        Initializes the ConvBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (Union[int, Tuple[int, ...]], optional): Kernel size. Defaults to 3.
            stride (Union[int, Tuple[int, ...]], optional): Stride. Defaults to 1.
            padding (Union[int, Tuple[int, ...], str], optional): Padding. Defaults to 0.
            dilation (Union[int, Tuple[int, ...]], optional): Dilation factor. Defaults to 1.
            groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
            bias (bool, optional): Whether to use bias. Defaults to True.
            dim (int, optional): Convolution dimension (1, 2, 3). Defaults to 2.
            norm (str, optional): Normalization type ('batch', 'group', 'layer', 'instance', None). Defaults to 'batch'.
            activation (str, optional): Activation function type ('relu', 'leaky_relu', 'gelu', 'silu', 'tanh', 'sigmoid', None). Defaults to 'relu'.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            pre_norm (bool, optional): Whether to use Pre-Norm (Norm-Conv-Act) structure. Defaults to False (Conv-Norm-Act).
        '''
        super().__init__()
        
        self.pre_norm = pre_norm
        
        if dim == 1:
            conv_class = nn.Conv1d
        elif dim == 2:
            conv_class = nn.Conv2d
        elif dim == 3:
            conv_class = nn.Conv3d
        else:
            raise ValueError(f'Unsupported dimension: {dim}')
            
        self.conv = conv_class(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias
        )
        
        self.norm = None
        if norm is not None:
            norm_channels = in_channels if pre_norm else out_channels
            if norm == 'batch':
                if dim == 1: self.norm = nn.BatchNorm1d(norm_channels)
                elif dim == 2: self.norm = nn.BatchNorm2d(norm_channels)
                elif dim == 3: self.norm = nn.BatchNorm3d(norm_channels)
            elif norm == 'group':
                num_groups = 32 if norm_channels % 32 == 0 else min(norm_channels, 8)
                self.norm = nn.GroupNorm(num_groups, norm_channels)
            elif norm == 'layer':
                self.norm = nn.GroupNorm(1, norm_channels) 
            elif norm == 'instance':
                if dim == 1: self.norm = nn.InstanceNorm1d(norm_channels)
                elif dim == 2: self.norm = nn.InstanceNorm2d(norm_channels)
                elif dim == 3: self.norm = nn.InstanceNorm3d(norm_channels)
            else:
                raise ValueError(f'Unsupported normalization: {norm}')

        self.act = None
        if activation is not None:
            if activation == 'relu':
                self.act = nn.ReLU(inplace=True)
            elif activation == 'leaky_relu':
                self.act = nn.LeakyReLU(0.1, inplace=True)
            elif activation == 'gelu':
                self.act = nn.GELU()
            elif activation == 'silu':
                self.act = nn.SiLU(inplace=True)
            elif activation == 'tanh':
                self.act = nn.Tanh()
            elif activation == 'sigmoid':
                self.act = nn.Sigmoid()
            else:
                raise ValueError(f'Unsupported activation: {activation}')

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        '''
        if self.pre_norm:
            if self.norm: x = self.norm(x)
            if self.act: x = self.act(x)
            x = self.conv(x)
            if self.dropout: x = self.dropout(x)
        else:
            x = self.conv(x)
            if self.norm: x = self.norm(x)
            if self.act: x = self.act(x)
            if self.dropout: x = self.dropout(x)
            
        return x


class DepthwiseSeparableConv(BasicModel):
    '''
    Depthwise Separable Convolution Block.
    
    Consists of a Depthwise Conv (channel-wise convolution) and a Pointwise Conv (1x1 convolution).
    Significantly reduces the number of parameters and computational cost while maintaining feature extraction capability.

    Attributes:
        depthwise (ConvBlock): The depthwise convolution block.
        pointwise (ConvBlock): The pointwise convolution block.
        use_res (bool): Whether to use residual connection.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...], str] = 1,
        dilation: Union[int, Tuple[int, ...]] = 1,
        bias: bool = False,
        dim: int = 2,
        norm: str = 'batch',
        activation: str = 'relu',
        dropout: float = 0.0,
        use_res: bool = True,
    ):
        '''
        Initializes DepthwiseSeparableConv.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size: Kernel size.
            stride: Stride.
            padding: Padding.
            dilation: Dilation factor.
            bias (bool): Whether to use bias.
            dim (int): Convolution dimension (1, 2, 3).
            norm (str): Normalization type.
            activation (str): Activation function type.
            dropout (float): Dropout probability.
            use_res (bool): Whether to use residual connection when input/output channels match and stride is 1.
        '''
        super().__init__()

        self.depthwise = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            dim=dim,
            norm=norm,
            activation=activation,
            dropout=0.0
        )

        self.pointwise = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            dim=dim,
            norm=norm,
            activation=activation,
            dropout=dropout
        )

        self.use_res = use_res and (in_channels == out_channels) and (stride == 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.
        '''
        identity = x
        
        out = self.depthwise(x)
        out = self.pointwise(out)

        if self.use_res:
            out += identity
            
        return out


class ResBasicBlock(BasicModel):
    '''
    Residual Basic Block.

    Consists of two convolution layers, supporting standard ResNet and Pre-activation ResNet variants.

    Attributes:
        conv1 (ConvBlock): The first convolution block.
        conv2 (ConvBlock): The second convolution block.
        downsample (ConvBlock, optional): The downsampling block.
        act (nn.Module, optional): The activation function (used in 'original' variant).
        variant (str): The variant type ('original' or 'pre_act').
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...], str] = 1,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = False,
        dim: int = 2,
        norm: str = 'batch',
        activation: str = 'relu',
        dropout: float = 0.0,
        variant: str = 'original',
    ):
        '''
        Initializes ResBasicBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (Union[int, Tuple[int, ...]], optional): Kernel size. Defaults to 3.
            stride (Union[int, Tuple[int, ...]], optional): Stride. Defaults to 1.
            padding (Union[int, Tuple[int, ...], str], optional): Padding. Defaults to 1.
            dilation (Union[int, Tuple[int, ...]], optional): Dilation factor. Defaults to 1.
            groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
            bias (bool, optional): Whether to use bias. Defaults to False.
            dim (int, optional): Convolution dimension (1, 2, 3). Defaults to 2.
            norm (str, optional): Normalization type. Defaults to 'batch'.
            activation (str, optional): Activation function type. Defaults to 'relu'.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            variant (str, optional): Variant type ('original', 'pre_act'). Defaults to 'original'.
        '''
        super().__init__()
        
        self.variant = variant
        self.activation = activation
        
        self.act = None
        if variant == 'original' and activation is not None:
            if activation == 'relu':
                self.act = nn.ReLU(inplace=True)
            elif activation == 'leaky_relu':
                self.act = nn.LeakyReLU(0.1, inplace=True)
            elif activation == 'gelu':
                self.act = nn.GELU()
            elif activation == 'silu':
                self.act = nn.SiLU(inplace=True)
            elif activation == 'tanh':
                self.act = nn.Tanh()
            elif activation == 'sigmoid':
                self.act = nn.Sigmoid()

        if variant == 'original':
            # Conv1: Conv-Norm-Act
            self.conv1 = ConvBlock(
                in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, dim, norm, activation, dropout, pre_norm=False
            )
            # Conv2: Conv-Norm
            self.conv2 = ConvBlock(
                out_channels, out_channels, kernel_size, 1, padding, dilation, groups, bias, dim, norm, activation=None, dropout=dropout, pre_norm=False
            )
        elif variant == 'pre_act':
            # Conv1: Norm-Act-Conv
            self.conv1 = ConvBlock(
                in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, dim, norm, activation, dropout, pre_norm=True
            )
            # Conv2: Norm-Act-Conv
            self.conv2 = ConvBlock(
                out_channels, out_channels, kernel_size, 1, padding, dilation, groups, bias, dim, norm, activation, dropout, pre_norm=True
            )
        else:
            raise ValueError(f'Unsupported variant: {variant}')

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            if variant == 'original':
                self.downsample = ConvBlock(
                    in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dim=dim, norm=norm, activation=None, bias=bias, pre_norm=False
                )
            elif variant == 'pre_act':
                self.downsample = ConvBlock(
                    in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dim=dim, norm=None, activation=None, bias=bias, pre_norm=False
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        '''
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.variant == 'original' and self.act is not None:
            out = self.act(out)

        return out
