import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

from codon.base import BasicModel
from codon.block.conv import ConvBlock
from codon.ops.pixelshuffle import pixel_shuffle, unpixel_shuffle


class PixelShuffleUpSample(BasicModel):
    '''
    Pixel Shuffle Upsampling Module.

    Supports 1D, 2D, and 3D data. This module uses a convolution to increase
    the number of channels, followed by a reshaping operation (Pixel Shuffle)
    to move channel spatial information to the spatial dimensions, effectively
    upsampling the input tensor.

    Attributes:
        conv (ConvBlock): The convolution block that projects the input to the required number of channels.
        dim (int): The dimensionality of the input data (1, 2, or 3).
        upscale_factor (int): The factor by which to upsample the spatial dimensions.
        out_channels (int): The final number of output channels after upsampling.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale_factor: int,
        dim: int = 2,
        norm: Optional[str] = None,
        activation: str = 'relu',
        dropout: float = 0.0
    ) -> None:
        '''
        Initializes the PixelShuffleUpSample module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels after pixel shuffle.
            upscale_factor (int): Factor to increase spatial resolution by.
            dim (int, optional): Dimensionality of the data (1, 2, or 3). Defaults to 2.
            norm (Optional[str], optional): Normalization type. Defaults to None.
            activation (str, optional): Activation function type. Defaults to 'relu'.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        '''
        super().__init__()
        
        if dim not in (1, 2, 3):
            raise ValueError(f'Unsupported dimension: {dim}. Must be 1, 2, or 3.')

        self.dim = dim
        self.upscale_factor = upscale_factor
        self.out_channels = out_channels

        intermediate_channels = out_channels * (upscale_factor ** dim)

        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dim=dim,
            norm=norm,
            activation=activation,
            dropout=dropout
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Defines the computation performed at every call.

        Args:
            input_tensor (torch.Tensor): The input data with shape (batch_size, in_channels, *spatial_dims).

        Returns:
            torch.Tensor: The upsampled output data with shape (batch_size, out_channels, *upsampled_spatial_dims).
        '''
        hidden = self.conv(input_tensor)
        output = pixel_shuffle(hidden, self.upscale_factor, self.out_channels, self.dim)
        return output

    @staticmethod
    def auto_build(
        input_shape: Tuple[int, ...],
        output_shape: Optional[Tuple[int, ...]] = None,
        upscale_factor: Optional[int] = None,
        norm: Optional[str] = None,
        activation: str = 'relu',
        dropout: float = 0.0,
        depth_level: int = 1
    ) -> nn.Module:
        '''
        Automatically builds a PixelShuffleUpSample module or a Sequential of modules based on shapes.

        If the desired output spatial dimension is not an exact multiple of the input spatial dimension
        by the upscale factor, an AdaptiveAvgPool layer is appended to match the output shape exactly.

        Args:
            input_shape (Tuple[int, ...]): Shape of the input data (without batch size).
            output_shape (Optional[Tuple[int, ...]], optional): Shape of the desired output data. Defaults to None.
            upscale_factor (Optional[int], optional): Factor to increase spatial resolution by. Defaults to None.
            norm (Optional[str], optional): Normalization type. Defaults to None.
            activation (str, optional): Activation function type. Defaults to 'relu'.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            depth_level (int, optional): Level of network depth multiplier. Defaults to 1.

        Returns:
            nn.Module: An initialized PixelShuffleUpSample module or an nn.Sequential.
        '''
        dim = len(input_shape) - 1
        in_channels = input_shape[0]

        if output_shape is not None:
            out_channels = output_shape[0]
            if upscale_factor is None:
                upscale_factor = max(1, math.ceil(output_shape[1] / input_shape[1]))
        else:
            out_channels = in_channels
            if upscale_factor is None:
                upscale_factor = 2  # Default to 2x upsampling if neither output_shape nor factor is provided

        layers = []
        block = PixelShuffleUpSample(
            in_channels=in_channels,
            out_channels=out_channels,
            upscale_factor=upscale_factor,
            dim=dim,
            norm=norm,
            activation=activation,
            dropout=dropout
        )
        layers.append(block)

        for _ in range(max(0, depth_level - 1)):
            layers.append(ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dim=dim,
                norm=norm,
                activation=activation,
                dropout=dropout
            ))

        if output_shape is not None:
            expected_spatial_size = input_shape[1] * upscale_factor
            target_spatial_size = output_shape[1]
            if expected_spatial_size != target_spatial_size:
                pool_layer = None
                target_spatial = output_shape[1:]
                if dim == 1:
                    pool_layer = nn.AdaptiveAvgPool1d(target_spatial)
                elif dim == 2:
                    pool_layer = nn.AdaptiveAvgPool2d(target_spatial)
                elif dim == 3:
                    pool_layer = nn.AdaptiveAvgPool3d(target_spatial)

                if pool_layer is not None:
                    layers.append(pool_layer)

        if len(layers) == 1:
            return layers[0]
        else:
            return nn.Sequential(*layers)


class UnPixelShuffleDownSample(BasicModel):
    '''
    UnPixel Shuffle Downsampling Module (Space-to-Depth).

    Supports 1D, 2D, and 3D data. This module performs the inverse of PixelShuffleUpSample.
    It reshapes spatial information into the channel dimension (space-to-depth),
    followed by a convolution to reduce the number of channels, effectively
    downsampling the input tensor.

    Attributes:
        conv (ConvBlock): The convolution block that projects the expanded channels to the required number of channels.
        dim (int): The dimensionality of the input data (1, 2, or 3).
        downscale_factor (int): The factor by which to downsample the spatial dimensions.
        out_channels (int): The final number of output channels after downsampling.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downscale_factor: int,
        dim: int = 2,
        norm: Optional[str] = None,
        activation: str = 'relu',
        dropout: float = 0.0
    ) -> None:
        '''
        Initializes the UnPixelShuffleDownSample module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels after convolution.
            downscale_factor (int): Factor to decrease spatial resolution by.
            dim (int, optional): Dimensionality of the data (1, 2, or 3). Defaults to 2.
            norm (Optional[str], optional): Normalization type. Defaults to None.
            activation (str, optional): Activation function type. Defaults to 'relu'.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        '''
        super().__init__()

        if dim not in (1, 2, 3):
            raise ValueError(f'Unsupported dimension: {dim}. Must be 1, 2, or 3.')

        self.dim = dim
        self.downscale_factor = downscale_factor
        self.out_channels = out_channels

        intermediate_channels = in_channels * (downscale_factor ** dim)

        self.conv = ConvBlock(
            in_channels=intermediate_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dim=dim,
            norm=norm,
            activation=activation,
            dropout=dropout
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Defines the computation performed at every call.

        Args:
            input_tensor (torch.Tensor): The input data with shape (batch_size, in_channels, *spatial_dims).

        Returns:
            torch.Tensor: The downsampled output data with shape (batch_size, out_channels, *downsampled_spatial_dims).
        '''
        hidden = unpixel_shuffle(input_tensor, self.downscale_factor, self.dim)
        output = self.conv(hidden)
        return output

    @staticmethod
    def auto_build(
        input_shape: Tuple[int, ...],
        output_shape: Optional[Tuple[int, ...]] = None,
        downscale_factor: Optional[int] = None,
        norm: Optional[str] = None,
        activation: str = 'relu',
        dropout: float = 0.0,
        depth_level: int = 1
    ) -> nn.Module:
        '''
        Automatically builds an UnPixelShuffleDownSample module or a Sequential of modules based on shapes.

        If the desired output spatial dimension is not an exact divisor of the input spatial dimension
        by the downscale factor, an AdaptiveAvgPool layer is prepended to match the output shape exactly.

        Args:
            input_shape (Tuple[int, ...]): Shape of the input data (without batch size).
            output_shape (Optional[Tuple[int, ...]], optional): Shape of the desired output data. Defaults to None.
            downscale_factor (Optional[int], optional): Factor to decrease spatial resolution by. Defaults to None.
            norm (Optional[str], optional): Normalization type. Defaults to None.
            activation (str, optional): Activation function type. Defaults to 'relu'.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            depth_level (int, optional): Level of network depth multiplier. Defaults to 1.

        Returns:
            nn.Module: An initialized UnPixelShuffleDownSample module or an nn.Sequential.
        '''
        dim = len(input_shape) - 1
        in_channels = input_shape[0]

        if output_shape is not None:
            out_channels = output_shape[0]
            if downscale_factor is None:
                downscale_factor = max(1, math.ceil(input_shape[1] / output_shape[1]))
        else:
            out_channels = in_channels
            if downscale_factor is None:
                downscale_factor = 2

        layers = []

        if output_shape is not None:
            expected_spatial_size = input_shape[1] // downscale_factor
            target_spatial_size = output_shape[1]
            if expected_spatial_size != target_spatial_size:
                pool_layer = None
                target_spatial = output_shape[1:]
                if dim == 1:
                    pool_layer = nn.AdaptiveAvgPool1d(target_spatial)
                elif dim == 2:
                    pool_layer = nn.AdaptiveAvgPool2d(target_spatial)
                elif dim == 3:
                    pool_layer = nn.AdaptiveAvgPool3d(target_spatial)

                if pool_layer is not None:
                    layers.append(pool_layer)

        block = UnPixelShuffleDownSample(
            in_channels=in_channels,
            out_channels=out_channels,
            downscale_factor=downscale_factor,
            dim=dim,
            norm=norm,
            activation=activation,
            dropout=dropout
        )
        layers.append(block)

        for _ in range(max(0, depth_level - 1)):
            layers.append(ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dim=dim,
                norm=norm,
                activation=activation,
                dropout=dropout
            ))

        if len(layers) == 1:
            return layers[0]
        else:
            return nn.Sequential(*layers)
