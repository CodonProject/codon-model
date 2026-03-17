import torch
from typing import Tuple


def pixel_shuffle(
    input_tensor: torch.Tensor,
    upscale_factor: int,
    out_channels: int,
    dim: int
) -> torch.Tensor:
    '''
    Performs pixel shuffle operation (depth-to-space) for 1D, 2D, and 3D data.

    This operation reshapes channel information into spatial dimensions,
    effectively upsampling the input tensor.

    Args:
        input_tensor (torch.Tensor): Input tensor with shape (batch_size, channels, *spatial_dims).
            channels must be equal to out_channels * (upscale_factor ** dim).
        upscale_factor (int): Factor to increase spatial resolution by.
        out_channels (int): Number of output channels after pixel shuffle.
        dim (int): Dimensionality of the data (1, 2, or 3).

    Returns:
        torch.Tensor: Upsampled tensor with shape (batch_size, out_channels, *upsampled_spatial_dims).
    '''
    batch_size, _, *spatial_dims = input_tensor.shape
    r = upscale_factor
    c = out_channels

    if dim == 1:
        l = spatial_dims[0]
        hidden = input_tensor.view(batch_size, c, r, l)
        hidden = hidden.permute(0, 1, 3, 2).contiguous()
        output = hidden.view(batch_size, c, l * r)
    elif dim == 2:
        h, w = spatial_dims
        hidden = input_tensor.view(batch_size, c, r, r, h, w)
        hidden = hidden.permute(0, 1, 4, 2, 5, 3).contiguous()
        output = hidden.view(batch_size, c, h * r, w * r)
    elif dim == 3:
        d, h, w = spatial_dims
        hidden = input_tensor.view(batch_size, c, r, r, r, d, h, w)
        hidden = hidden.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        output = hidden.view(batch_size, c, d * r, h * r, w * r)
    else:
        raise ValueError(f'Unsupported dimension: {dim}. Must be 1, 2, or 3.')

    return output


def unpixel_shuffle(
    input_tensor: torch.Tensor,
    downscale_factor: int,
    dim: int
) -> torch.Tensor:
    '''
    Performs inverse pixel shuffle operation (space-to-depth) for 1D, 2D, and 3D data.

    This operation reshapes spatial information into channel dimensions,
    effectively downsampling the input tensor.

    Args:
        input_tensor (torch.Tensor): Input tensor with shape (batch_size, channels, *spatial_dims).
        downscale_factor (int): Factor to decrease spatial resolution by.
        dim (int): Dimensionality of the data (1, 2, or 3).

    Returns:
        torch.Tensor: Downsampled tensor with shape (batch_size, channels * (downscale_factor ** dim), *downsampled_spatial_dims).
    '''
    batch_size, c, *spatial_dims = input_tensor.shape
    r = downscale_factor

    if dim == 1:
        l = spatial_dims[0]
        hidden = input_tensor.view(batch_size, c, l // r, r)
        hidden = hidden.permute(0, 1, 3, 2).contiguous()
        output = hidden.view(batch_size, c * r, l // r)
    elif dim == 2:
        h, w = spatial_dims
        hidden = input_tensor.view(batch_size, c, h // r, r, w // r, r)
        hidden = hidden.permute(0, 1, 3, 5, 2, 4).contiguous()
        output = hidden.view(batch_size, c * r * r, h // r, w // r)
    elif dim == 3:
        d, h, w = spatial_dims
        hidden = input_tensor.view(batch_size, c, d // r, r, h // r, r, w // r, r)
        hidden = hidden.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        output = hidden.view(batch_size, c * r * r * r, d // r, h // r, w // r)
    else:
        raise ValueError(f'Unsupported dimension: {dim}. Must be 1, 2, or 3.')

    return output
