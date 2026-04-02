import torch
import torch.nn.functional as F

from .manifold_triton import JIT
if JIT:
    from .manifold_triton import ManifoldLinearFuseFunction

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
    # 1. Native PyTorch implementation for heavy MatMul (Optimized by cuBLAS)
    x_norm = F.normalize(input_tensor, p=2, dim=1)
    w_norm = F.normalize(weight, p=2, dim=1)
    
    # cosine shape: [batch_size, out_features]
    cosine = F.linear(x_norm, w_norm)
    
    # 2. Triton Fusion Engine for element-wise operations and backprop
    if JIT and input_tensor.is_cuda and op == 'triton':
        # Calls the Triton custom autograd function
        output = ManifoldLinearFuseFunction.apply(
            cosine, kappa, lambda_rate, scale, bias, rule
        )
    else:
        # Fallback to pure PyTorch logic if Triton is not available or running on CPU
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