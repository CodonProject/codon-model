'''
nnUNet-style U-Net implementation supporting 1D/2D/3D data with optional PixelShuffle.

This module provides a flexible U-Net architecture that can be configured for coarse or fine
granularity, with optional pixel-shuffle-based up/down-sampling.
'''
from codon import *
from codon.block.conv import *
from codon.block.pixelshuffle import *


@dataclass
class UNetConfig:
    '''
    Configuration dataclass for the U-Net.

    Attributes:
        num_pooling (int): Number of down-sampling (and up-sampling) stages.
        patch_size (Union[Tuple[int, int], Tuple[int, int, int]]): Spatial size of the input patch.
            For 2D: (height, width); for 3D: (height, width, depth).
        stride (Union[Tuple[int, int], Tuple[int, int, int]]): Stride of the patch extraction.
        dim (int, optional): Dimensionality of the data (1, 2, or 3). Defaults to 2.
    '''
    num_pooling: int
    patch_size: Union[Tuple[int, int], Tuple[int, int, int]]
    stride: Union[Tuple[int, int], Tuple[int, int, int]]
    dim: int = 2


class _DownBlock(nn.Module):
    '''
    Single down-sampling block: two convolutions followed by down-sampling.

    This block applies two convolution layers (with optional normalisation and activation),
    then downsamples the feature map either via stride-2 convolution or UnPixelShuffle.

    Attributes:
        conv1 (ConvBlock): First convolution block.
        conv2 (ConvBlock): Second convolution block.
        downsample (nn.Module): Down-sampling layer (ConvBlock or UnPixelShuffleDownSample).
    '''

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        next_in_ch: int,
        use_pixel_shuffle: bool,
        dim: int = 2,
        norm: Optional[str] = None,
        activation: str = 'relu',
        dropout: float = 0.0
    ) -> None:
        '''
        Initialises the down-sampling block.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels after the two convolutions.
            next_in_ch (int): Number of channels after down-sampling (input to the next block).
            use_pixel_shuffle (bool): If True, use UnPixelShuffle for down-sampling;
                otherwise use stride-2 convolution.
            dim (int, optional): Data dimension. Defaults to 2.
            norm (Optional[str], optional): Normalisation type. Defaults to None.
            activation (str, optional): Activation function. Defaults to 'relu'.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        '''
        super().__init__()
        self.conv1 = ConvBlock(
            in_ch, out_ch, kernel_size=3, stride=1, padding=1,
            dim=dim, norm=norm, activation=activation, dropout=dropout
        )
        self.conv2 = ConvBlock(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1,
            dim=dim, norm=norm, activation=activation, dropout=dropout
        )
        if use_pixel_shuffle:
            self.downsample = UnPixelShuffleDownSample(
                out_ch, next_in_ch, downscale_factor=2,
                dim=dim, norm=norm, activation=activation, dropout=dropout
            )
        else:
            self.downsample = ConvBlock(
                out_ch, next_in_ch, kernel_size=3, stride=2, padding=1,
                dim=dim, norm=norm, activation=activation, dropout=dropout
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_ch, *spatial).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - skip: Feature map before down-sampling (used as skip connection).
                - down: Down-sampled feature map.
        '''
        skip = self.conv2(self.conv1(x))
        down = self.downsample(skip)
        return skip, down


class _UpBlock(nn.Module):
    '''
    Single up-sampling block: up-sampling, concatenation with skip, then two convolutions.

    Attributes:
        upsample (nn.Module): Up-sampling layer (PixelShuffleUpSample or Upsample+Conv).
        conv1 (ConvBlock): First convolution after concatenation.
        conv2 (ConvBlock): Second convolution.
    '''

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        skip_ch: int,
        use_pixel_shuffle: bool,
        dim: int = 2,
        norm: Optional[str] = None,
        activation: str = 'relu',
        dropout: float = 0.0
    ) -> None:
        '''
        Initialises the up-sampling block.

        Args:
            in_ch (int): Number of input channels from the lower level.
            out_ch (int): Number of channels after up-sampling (should match skip_ch).
            skip_ch (int): Number of channels in the skip connection.
            use_pixel_shuffle (bool): If True, use PixelShuffle for up-sampling;
                otherwise use interpolation + convolution.
            dim (int, optional): Data dimension. Defaults to 2.
            norm (Optional[str], optional): Normalisation type. Defaults to None.
            activation (str, optional): Activation function. Defaults to 'relu'.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        '''
        super().__init__()
        if use_pixel_shuffle:
            self.upsample = PixelShuffleUpSample(
                in_ch, out_ch, upscale_factor=2,
                dim=dim, norm=norm, activation=activation, dropout=dropout
            )
        else:
            if dim == 1:
                mode = 'linear'
            elif dim == 2:
                mode = 'bilinear'
            elif dim == 3:
                mode = 'trilinear'
            else:
                raise ValueError(f'Unsupported dim: {dim}')
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=mode, align_corners=True),
                ConvBlock(
                    in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                    dim=dim, norm=norm, activation=activation, dropout=dropout
                )
            )
        self.conv1 = ConvBlock(
            out_ch + skip_ch, out_ch, kernel_size=3, stride=1, padding=1,
            dim=dim, norm=norm, activation=activation, dropout=dropout
        )
        self.conv2 = ConvBlock(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1,
            dim=dim, norm=norm, activation=activation, dropout=dropout
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            x (torch.Tensor): Feature map from the lower level.
            skip (torch.Tensor): Skip connection feature map.

        Returns:
            torch.Tensor: Output feature map after up-sampling and convolutions.
        '''
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class nnUNetEncoder(BasicModel):
    '''
    Encoder of the nnUNet: a sequence of down-sampling blocks.

    Attributes:
        blocks (nn.ModuleList): List of _DownBlock modules.
    '''

    def __init__(
        self,
        in_channels: int,
        num_pooling: int,
        base_channels: int = 32,
        use_pixel_shuffle: bool = False,
        norm: Optional[str] = None,
        activation: str = 'relu',
        dropout: float = 0.0,
        dim: int = 2
    ) -> None:
        '''
        Initialises the encoder.

        Args:
            in_channels (int): Number of input channels.
            num_pooling (int): Number of down-sampling stages.
            base_channels (int, optional): Number of channels in the first stage,
                doubled each stage. Defaults to 32.
            use_pixel_shuffle (bool, optional): Whether to use UnPixelShuffle for down-sampling.
                Defaults to False.
            norm (Optional[str], optional): Normalisation type. Defaults to None.
            activation (str, optional): Activation function. Defaults to 'relu'.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            dim (int, optional): Data dimension. Defaults to 2.
        '''
        super().__init__()
        self.blocks = nn.ModuleList()
        in_ch = in_channels
        for i in range(num_pooling):
            out_ch = base_channels * (2 ** i)
            next_in_ch = base_channels * (2 ** (i + 1))
            block = _DownBlock(
                in_ch, out_ch, next_in_ch,
                use_pixel_shuffle, dim, norm, activation, dropout
            )
            self.blocks.append(block)
            in_ch = next_in_ch

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        '''
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, *spatial).

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]:
                - skips: List of skip connection tensors from each stage (shallow to deep).
                - last_feat: Final feature map after all down-sampling stages.
        '''
        skips = []
        for block in self.blocks:
            skip, x = block(x)
            skips.append(skip)
        return skips, x


class nnUNetBottom(BasicModel):
    '''
    Bottleneck of the nnUNet: two convolutions without changing spatial size.

    Attributes:
        conv1 (ConvBlock): First convolution block.
        conv2 (ConvBlock): Second convolution block.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        norm: Optional[str] = None,
        activation: str = 'relu',
        dropout: float = 0.0,
        dim: int = 2
    ) -> None:
        '''
        Initialises the bottleneck.

        Args:
            in_channels (int): Number of input channels.
            out_channels (Optional[int], optional): Number of output channels.
                If None, set to in_channels. Defaults to None.
            norm (Optional[str], optional): Normalisation type. Defaults to None.
            activation (str, optional): Activation function. Defaults to 'relu'.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            dim (int, optional): Data dimension. Defaults to 2.
        '''
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv1 = ConvBlock(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1,
            dim=dim, norm=norm, activation=activation, dropout=dropout
        )
        self.conv2 = ConvBlock(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1,
            dim=dim, norm=norm, activation=activation, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with same spatial size.
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class nnUNetDecoder(BasicModel):
    '''
    Decoder of the nnUNet: a sequence of up-sampling blocks that fuse skip connections.

    Attributes:
        blocks (nn.ModuleList): List of _UpBlock modules.
        final_conv (nn.Module): Final 1x1 convolution to produce the desired output channels.
    '''

    def __init__(
        self,
        in_channels: int,
        num_pooling: int,
        base_channels: int = 32,
        final_out_channels: Optional[int] = None,
        use_pixel_shuffle: bool = False,
        norm: Optional[str] = None,
        activation: str = 'relu',
        dropout: float = 0.0,
        dim: int = 2
    ) -> None:
        '''
        Initialises the decoder.

        Args:
            in_channels (int): Number of input channels from the bottleneck.
            num_pooling (int): Number of up-sampling stages (should match encoder).
            base_channels (int, optional): Base channel count (same as encoder). Defaults to 32.
            final_out_channels (Optional[int], optional): Desired output channels.
                If None, output channels equal the last layer's channels. Defaults to None.
            use_pixel_shuffle (bool, optional): Whether to use PixelShuffle for up-sampling.
                Defaults to False.
            norm (Optional[str], optional): Normalisation type. Defaults to None.
            activation (str, optional): Activation function. Defaults to 'relu'.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            dim (int, optional): Data dimension. Defaults to 2.
        '''
        super().__init__()
        self.blocks = nn.ModuleList()
        # Build from deepest to shallowest
        for i in range(num_pooling - 1, -1, -1):
            out_ch = base_channels * (2 ** i)
            skip_ch = out_ch
            block = _UpBlock(
                in_channels, out_ch, skip_ch,
                use_pixel_shuffle, dim, norm, activation, dropout
            )
            self.blocks.append(block)
            in_channels = out_ch

        # Final projection if needed
        if final_out_channels is not None and final_out_channels != in_channels:
            self.final_conv = ConvBlock(
                in_channels, final_out_channels, kernel_size=1, stride=1, padding=0,
                dim=dim, norm=None, activation=None, dropout=0.0
            )
        else:
            self.final_conv = nn.Identity()

    def forward(self, skips: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            skips (List[torch.Tensor]): List of skip connections from encoder (shallow to deep).
            x (torch.Tensor): Bottleneck feature map.

        Returns:
            torch.Tensor: Final output tensor.
        '''
        for block, skip in zip(self.blocks, reversed(skips)):
            x = block(x, skip)
        x = self.final_conv(x)
        return x


class nnUNet(BasicModel):
    '''
    Full nnUNet-style U-Net architecture supporting 1D, 2D, and 3D data.

    The network consists of an encoder (down-sampling path), a bottleneck,
    and a decoder (up-sampling path) with skip connections. The depth and width
    can be adjusted via num_pooling and base_channels. Optionally, PixelShuffle
    can be used for up/down-sampling instead of standard convolution+interpolation.

    Attributes:
        encoder (nnUNetEncoder): The encoder module.
        bottom (nnUNetBottom): The bottleneck module.
        decoder (nnUNetDecoder): The decoder module.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_pooling: int,
        base_channels: int = 32,
        use_pixel_shuffle: bool = False,
        norm: Optional[str] = None,
        activation: str = 'relu',
        dropout: float = 0.0,
        dim: int = 2
    ) -> None:
        '''
        Initialises the nnUNet.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_pooling (int): Number of down-sampling stages (network depth).
            base_channels (int, optional): Base number of channels (first stage).
                Doubled after each pooling. Defaults to 32.
            use_pixel_shuffle (bool, optional): If True, use PixelShuffle/UnPixelShuffle
                for up/down-sampling; otherwise use standard convolution+interpolation.
                Defaults to False.
            norm (Optional[str], optional): Normalisation type. Defaults to None.
            activation (str, optional): Activation function. Defaults to 'relu'.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            dim (int, optional): Data dimension (1, 2, or 3). Defaults to 2.
        '''
        super().__init__()
        self.encoder = nnUNetEncoder(
            in_channels, num_pooling, base_channels,
            use_pixel_shuffle, norm, activation, dropout, dim
        )
        bottom_in = base_channels * (2 ** num_pooling)
        self.bottom = nnUNetBottom(
            bottom_in, bottom_in, norm, activation, dropout, dim
        )
        self.decoder = nnUNetDecoder(
            bottom_in, num_pooling, base_channels, out_channels,
            use_pixel_shuffle, norm, activation, dropout, dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, *spatial).

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels, *spatial).
        '''
        skips, last_feat = self.encoder(x)
        bottom_feat = self.bottom(last_feat)
        out = self.decoder(skips, bottom_feat)
        return out

    @staticmethod
    def build_from_config(
        config: UNetConfig,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        use_pixel_shuffle: bool = False,
        norm: Optional[str] = None,
        activation: str = 'relu',
        dropout: float = 0.0
    ) -> 'nnUNet':
        '''
        Builds an nnUNet instance from a configuration object.

        Args:
            config (UNetConfig): Configuration containing num_pooling and dim.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            base_channels (int, optional): Base channel count. Defaults to 32.
            use_pixel_shuffle (bool, optional): Whether to use PixelShuffle. Defaults to False.
            norm (Optional[str], optional): Normalisation type. Defaults to None.
            activation (str, optional): Activation function. Defaults to 'relu'.
            dropout (float, optional): Dropout probability. Defaults to 0.0.

        Returns:
            nnUNet: Initialised nnUNet model.
        '''
        dim = getattr(config, 'dim', 2)
        return nnUNet(
            in_channels, out_channels, config.num_pooling,
            base_channels, use_pixel_shuffle, norm, activation, dropout, dim
        )


def compute_2d_config(image_size: Tuple[int, int]) -> UNetConfig:
    '''
    Automatically computes a suitable U-Net configuration for 2D images.

    The pooling count is chosen so that the smallest spatial dimension is reduced
    to at least 4, and the patch size is made divisible by 2**num_pooling.

    Args:
        image_size (Tuple[int, int]): (height, width) of the input image.

    Returns:
        UNetConfig: Configuration object with num_pooling, patch_size, stride, and dim=2.
    '''
    H, W = image_size
    max_pool_H = math.floor(math.log2(H / 4)) if H >= 8 else 0
    max_pool_W = math.floor(math.log2(W / 4)) if W >= 8 else 0
    d = min(max_pool_H, max_pool_W)
    d = max(1, d)
    divisor = 2 ** d
    patch_H = (H // divisor) * divisor
    patch_W = (W // divisor) * divisor
    patch_size = (patch_H, patch_W)
    stride = (patch_H // 2, patch_W // 2)
    return UNetConfig(
        num_pooling=d,
        patch_size=patch_size,
        stride=stride,
        dim=2
    )


def compute_3d_config(image_size: Tuple[int, int, int]) -> UNetConfig:
    '''
    Automatically computes a suitable U-Net configuration for 3D volumes.

    The pooling count is chosen based on the minimum spatial dimension,
    ensuring each dimension is reduced to at least 4.

    Args:
        image_size (Tuple[int, int, int]): (height, width, depth) of the input volume.

    Returns:
        UNetConfig: Configuration object with num_pooling, patch_size, stride, and dim=3.
    '''
    H, W, D = image_size
    max_pool_H = math.floor(math.log2(H / 4)) if H >= 8 else 0
    max_pool_W = math.floor(math.log2(W / 4)) if W >= 8 else 0
    max_pool_D = math.floor(math.log2(D / 4)) if D >= 8 else 0
    d = min(max_pool_H, max_pool_W, max_pool_D)
    d = max(1, d)
    divisor = 2 ** d
    patch_H = (H // divisor) * divisor
    patch_W = (W // divisor) * divisor
    patch_D = (D // divisor) * divisor
    patch_size = (patch_H, patch_W, patch_D)
    stride = (patch_H // 2, patch_W // 2, patch_D // 2)
    return UNetConfig(
        num_pooling=d,
        patch_size=patch_size,
        stride=stride,
        dim=3
    )


def build_coarse_2d(
    in_channels: int,
    out_channels: int,
    image_size: Tuple[int, int],
    **kwargs
) -> nnUNet:
    '''
    Builds a coarse-grained 2D U-Net with one fewer down-sampling stage.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        image_size (Tuple[int, int]): Input image size (height, width).
        **kwargs: Additional arguments passed to nnUNet.build_from_config.

    Returns:
        nnUNet: Coarse 2D nnUNet model.
    '''
    config = compute_2d_config(image_size)
    config.num_pooling = max(1, config.num_pooling - 1)
    return nnUNet.build_from_config(config, in_channels, out_channels, **kwargs)


def build_fine_2d(
    in_channels: int,
    out_channels: int,
    image_size: Tuple[int, int],
    **kwargs
) -> nnUNet:
    '''
    Builds a fine-grained 2D U-Net with one additional down-sampling stage (up to max 5).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        image_size (Tuple[int, int]): Input image size (height, width).
        **kwargs: Additional arguments passed to nnUNet.build_from_config.

    Returns:
        nnUNet: Fine 2D nnUNet model.
    '''
    config = compute_2d_config(image_size)
    config.num_pooling = min(5, config.num_pooling + 1)
    return nnUNet.build_from_config(config, in_channels, out_channels, **kwargs)


def build_coarse_3d(
    in_channels: int,
    out_channels: int,
    image_size: Tuple[int, int, int],
    **kwargs
) -> nnUNet:
    '''
    Builds a coarse-grained 3D U-Net with one fewer down-sampling stage.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        image_size (Tuple[int, int, int]): Input volume size (height, width, depth).
        **kwargs: Additional arguments passed to nnUNet.build_from_config.

    Returns:
        nnUNet: Coarse 3D nnUNet model.
    '''
    config = compute_3d_config(image_size)
    config.num_pooling = max(1, config.num_pooling - 1)
    return nnUNet.build_from_config(config, in_channels, out_channels, **kwargs)


def build_fine_3d(
    in_channels: int,
    out_channels: int,
    image_size: Tuple[int, int, int],
    **kwargs
) -> nnUNet:
    '''
    Builds a fine-grained 3D U-Net with one additional down-sampling stage (up to max 5).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        image_size (Tuple[int, int, int]): Input volume size (height, width, depth).
        **kwargs: Additional arguments passed to nnUNet.build_from_config.

    Returns:
        nnUNet: Fine 3D nnUNet model.
    '''
    config = compute_3d_config(image_size)
    config.num_pooling = min(5, config.num_pooling + 1)
    return nnUNet.build_from_config(config, in_channels, out_channels, **kwargs)