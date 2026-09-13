from codon import *
from codon.block.conv import ConvBlock, ResBasicBlock


class ResNet(BasicModel):
    '''
    Residual Neural Network (ResNet).

    Supports 1D, 2D, and 3D data. Can be configured with different depths,
    channels, and structural variants.

    Attributes:
        conv1 (ConvBlock): The initial convolution layer.
        maxpool (nn.Module): The initial max pooling layer.
        stages (nn.Sequential): The sequential container of all residual stages.
        avgpool (nn.Module, optional): The global average pooling layer.
        fc (nn.Linear, optional): The final fully connected classification layer.
    '''

    def __init__(
        self,
        in_channels: int,
        layers: List[int],
        num_classes: int = 1000,
        dim: int = 2,
        base_channels: int = 64,
        norm: str = 'batch',
        activation: str = 'relu',
        dropout: float = 0.0,
        variant: str = 'original',
        include_top: bool = True,
        small_input: bool = False
    ) -> None:
        '''
        Initializes the ResNet model.

        Args:
            in_channels (int): Number of input channels.
            layers (List[int]): Number of residual blocks in each of the 4 stages.
            num_classes (int, optional): Number of output classes. Defaults to 1000.
            dim (int, optional): Dimensionality of the data (1, 2, or 3). Defaults to 2.
            base_channels (int, optional): Number of channels after the first convolution. Defaults to 64.
            norm (str, optional): Normalization type. Defaults to 'batch'.
            activation (str, optional): Activation function type. Defaults to 'relu'.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            variant (str, optional): Residual block variant ('original' or 'pre_act'). Defaults to 'original'.
            include_top (bool, optional): Whether to include the final pooling and fully connected layers. Defaults to True.
            small_input (bool, optional): If True, replaces the initial 7x7 conv and maxpool with a 3x3 conv. 
                                          Ideal for small images (e.g., 32x32 CIFAR) or small patches. Defaults to False.
        '''
        super().__init__()

        self.dim = dim
        self.include_top = include_top
        self.inplanes = base_channels
        self.small_input = small_input

        if self.small_input:
            self.conv1 = ConvBlock(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                dim=dim,
                norm=norm,
                activation=activation,
                dropout=dropout,
                pre_norm=False
            )
            self.maxpool = None
        else:
            self.conv1 = ConvBlock(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
                dim=dim,
                norm=norm,
                activation=activation,
                dropout=dropout,
                pre_norm=False
            )

            if dim == 1:
                self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            elif dim == 2:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            elif dim == 3:
                self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            else:
                raise ValueError(f'Unsupported dimension: {dim}')

        stages_list = []
        for i, blocks in enumerate(layers):
            stride = 1 if i == 0 else 2
            out_channels = base_channels * (2 ** i)
            stages_list.append(self._make_layer(
                out_channels, blocks, stride=stride, dim=dim, norm=norm, activation=activation, dropout=dropout, variant=variant
            ))
        
        self.stages = nn.Sequential(*stages_list)

        if self.include_top:
            if dim == 1:
                self.avgpool = nn.AdaptiveAvgPool1d(1)
            elif dim == 2:
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            elif dim == 3:
                self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

            final_channels = base_channels * (2 ** (len(layers) - 1)) if layers else base_channels
            self.fc = nn.Linear(final_channels, num_classes)

    def _make_layer(
        self,
        out_channels: int,
        blocks: int,
        stride: int,
        dim: int,
        norm: str,
        activation: str,
        dropout: float,
        variant: str
    ) -> nn.Sequential:
        '''
        Creates a sequential residual stage.

        Args:
            out_channels (int): Number of output channels for the blocks.
            blocks (int): Number of residual blocks in this stage.
            stride (int): Stride for the first block in the stage.
            dim (int): Dimensionality of the convolution.
            norm (str): Normalization type.
            activation (str): Activation function type.
            dropout (float): Dropout probability.
            variant (str): Residual block variant.

        Returns:
            nn.Sequential: A sequential container of the residual blocks.
        '''
        layers_list = []
        layers_list.append(ResBasicBlock(
            in_channels=self.inplanes,
            out_channels=out_channels,
            stride=stride,
            dim=dim,
            norm=norm,
            activation=activation,
            dropout=dropout,
            variant=variant
        ))
        self.inplanes = out_channels
        
        for _ in range(1, blocks):
            layers_list.append(ResBasicBlock(
                in_channels=self.inplanes,
                out_channels=out_channels,
                stride=1,
                dim=dim,
                norm=norm,
                activation=activation,
                dropout=dropout,
                variant=variant
            ))

        return nn.Sequential(*layers_list)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Defines the computation performed at every call.

        Args:
            input_tensor (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output features or logits.
        '''
        hidden = self.conv1(input_tensor)
        
        if self.maxpool is not None:
            hidden = self.maxpool(hidden)

        hidden = self.stages(hidden)

        if self.include_top:
            hidden = self.avgpool(hidden)
            hidden = torch.flatten(hidden, 1)
            output = self.fc(hidden)
            return output

        return hidden

    @staticmethod
    def auto_build(
        input_shape: Tuple[int, ...], 
        output_shape: Optional[Tuple[int, ...]] = None, 
        layers: Optional[List[int]] = None,
        base_channels: int = 64,
        norm: str = 'batch',
        activation: str = 'relu',
        dropout: float = 0.0,
        variant: str = 'original',
        depth_level: int = 1
    ) -> 'ResNet':
        '''
        Automatically builds a ResNet model based on input and output shapes.

        Args:
            input_shape (Tuple[int, ...]): Shape of the input data (without batch size). 
                                           E.g., (3, 224, 224) for 2D images.
            output_shape (Optional[Tuple[int, ...]], optional): Shape of the output targets. 
                                                                If provided, includes the classification head. Defaults to None.
            layers (Optional[List[int]], optional): List containing the number of blocks per stage. 
                                                    If None, it's dynamically calculated based on the spatial size. Defaults to None.
            base_channels (int, optional): Number of channels after the first convolution. Defaults to 64.
            norm (str, optional): Normalization type. Defaults to 'batch'.
            activation (str, optional): Activation function type. Defaults to 'relu'.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            variant (str, optional): Residual block variant. Defaults to 'original'.
            depth_level (int, optional): Level of network depth multiplier. Used when layers is None. Defaults to 1.

        Returns:
            ResNet: An initialized ResNet model.
        '''
        dim = len(input_shape) - 1
        in_channels = input_shape[0]

        small_input = False
        if dim > 0:
            spatial_size = min(input_shape[1:])
            if spatial_size <= 32:
                small_input = True

        if layers is None:
            if dim > 0:
                if small_input:
                    num_stages = max(1, int(math.log2(spatial_size)))
                else:
                    num_stages = max(1, int(math.log2(spatial_size)) - 2)
            else:
                num_stages = 4
            layers = [2 * max(1, depth_level)] * num_stages

        if output_shape is not None:
            num_classes = output_shape[-1]
            include_top = True
        else:
            num_classes = 1000
            include_top = False

        return ResNet(
            in_channels=in_channels,
            layers=layers,
            num_classes=num_classes,
            dim=dim,
            base_channels=base_channels,
            norm=norm,
            activation=activation,
            dropout=dropout,
            variant=variant,
            include_top=include_top,
            small_input=small_input
        )
