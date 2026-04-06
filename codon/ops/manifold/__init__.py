import torch
import torch.nn.functional as F

from .linear import JIT
if JIT:
    from .linear import ManifoldLinearFuseFunction
    from .conv import ManifoldConvFuseFunction

def riemannian_manifold_linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    kappa: torch.Tensor,
    lambda_rate: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    rule: str = 'near',
    op: str = 'triton'
) -> torch.Tensor:
    '''
    Applies the Riemannian manifold linear projection using a hybrid of 
    native cuBLAS (for matmul) and Triton (for element-wise fusion).

    Args:
        input_tensor (torch.Tensor): The input data with shape (batch_size, in_features).
        weight (torch.Tensor): The weights of the layer with shape (out_features, in_features).
        kappa (torch.Tensor): Concentration parameter for the von Mises-Fisher (vMF) distribution.
        lambda_rate (torch.Tensor): Gravitational attraction coefficient.
        scale (torch.Tensor): Vector amplifier for the hyperspherical network.
        bias (torch.Tensor): Manifold bias vector.
        rule (str): Attraction rule, either 'near' or 'far'. Default is 'near'.

    Returns:
        torch.Tensor: The output data with shape (batch_size, out_features).
    '''
    x_norm = F.normalize(input_tensor, p=2, dim=1)
    w_norm = F.normalize(weight, p=2, dim=1)
    
    cosine = F.linear(x_norm, w_norm)
    
    if JIT and input_tensor.is_cuda and op == 'triton':
        output = ManifoldLinearFuseFunction.apply(
            cosine, kappa, lambda_rate, scale, bias, rule
        )
    else:
        cosine_clamp = torch.clamp(cosine, -1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cosine_clamp)
        
        exp_val = torch.exp(kappa * (cosine_clamp - 1.0))
        if rule == 'far': 
            attraction = 1.0 - exp_val
        else:
            attraction = exp_val
        
        safe_lambda = torch.clamp(lambda_rate, 1e-6, 1.0 - 1e-4)
        effective_theta = theta * (1.0 - safe_lambda * attraction)
        
        output = scale * torch.cos(effective_theta) + bias

    return output


def riemannian_manifold_conv2d(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    weight_ones: torch.Tensor,
    kappa: torch.Tensor,
    lambda_rate: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    rule: str = 'near',
    use_norm: bool = False,
    op: str = 'triton'
) -> torch.Tensor:
    '''
    Applies the Riemannian manifold 2D convolution using a hybrid of 
    native PyTorch (for spatial conv) and Triton (for element-wise fusion).

    Args:
        input_tensor (torch.Tensor): Input data with shape (batch_size, in_channels, H, W).
        weight (torch.Tensor): Convolution weights with shape (out_channels, in_channels, kH, kW).
        weight_ones (torch.Tensor): Fixed all-ones kernel (1, in_channels, kH, kW) for norm calculation.
        kappa (torch.Tensor): vMF concentration parameter.
        lambda_rate (torch.Tensor): Gravitational attraction coefficient.
        scale (torch.Tensor): Vector amplifier.
        bias (torch.Tensor): Manifold bias vector.
        stride (int): Stride of the convolution.
        padding (int): Padding of the convolution.
        dilation (int): Dilation of the convolution.
        rule (str): 'near' or 'far'.
        use_norm (bool): Whether to scale output by patch norm.
        op (str): 'triton' or 'pytorch'.

    Returns:
        torch.Tensor: The output manifold projection tensor.
    '''
    w_flat = weight.view(weight.size(0), -1)
    w_norm_flat = F.normalize(w_flat, p=2, dim=1)
    w_norm = w_norm_flat.view_as(weight)
    
    x_sq = F.conv2d(input_tensor ** 2, weight_ones, stride=stride, padding=padding, dilation=dilation)
    x_norm_val = torch.sqrt(torch.clamp(x_sq, min=1e-6))
    
    conv_proj = F.conv2d(input_tensor, w_norm, stride=stride, padding=padding, dilation=dilation)
    cosine = conv_proj / (x_norm_val + 1e-6)

    if JIT and input_tensor.is_cuda and op == 'triton':
        output = ManifoldConvFuseFunction.apply(
            cosine, kappa, lambda_rate, scale, bias, rule
        )
    else:
        cosine_clamp = torch.clamp(cosine, -1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cosine_clamp)
        exp_val = torch.exp(kappa * (cosine_clamp - 1.0))
        
        if rule == 'far':
            attraction = 1.0 - exp_val
        else:
            attraction = exp_val
            
        safe_lambda = torch.clamp(lambda_rate, 1e-6, 1.0 - 1e-4)
        effective_theta = theta * (1.0 - safe_lambda * attraction)
        
        scale_view = scale.view(1, -1, 1, 1)
        bias_view = bias.view(1, -1, 1, 1)
        output = scale_view * torch.cos(effective_theta) + bias_view

    if use_norm:
        output *= x_norm_val
        
    return output
