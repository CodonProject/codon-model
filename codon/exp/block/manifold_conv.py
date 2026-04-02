import torch.nn.functional as F

from codon.base import *

import math
from typing import Tuple, Union

from .manifold import MainfoldLoss
from codon.exp.ops.manifold import riemannian_manifold_conv2d, euclidean_manifold_conv2d


class BasicManifoldConv2d(BasicModel):
    '''
    Base class for manifold-based 2D convolutional layers.
    
    Attributes:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (Tuple[int, int]): Size of the convolving kernel.
        stride (Union[int, Tuple[int, int]]): Stride of the convolution.
        padding (Union[int, Tuple[int, int]]): Padding added to all four sides of the input.
        dilation (Union[int, Tuple[int, int]]): Spacing between kernel elements.
        k_neighbors (int): Number of nearest neighbors for the Laplacian graph.
        weight (nn.Parameter): The learnable weights (manifold anchors) of the layer.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        k_neighbors: int = 2,
    ) -> None:
        '''
        Initializes the BasicManifoldConv2d layer.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (Union[int, Tuple[int, int]]): Size of the convolving kernel.
            stride (Union[int, Tuple[int, int]]): Stride of the convolution. Default: 1.
            padding (Union[int, Tuple[int, int]]): Zero-padding added to both sides of the input. Default: 0.
            dilation (Union[int, Tuple[int, int]]): Spacing between kernel elements. Default: 1.
            k_neighbors (int): Number of nearest neighbors to consider for Laplacian loss. Default: 2.
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.k_neighbors = min(k_neighbors, out_channels - 1)

        # Convolutional kernel acts as the manifold anchor [out_channels, in_channels, kH, kW]
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        
    def _flatten_weight(self) -> torch.Tensor:
        '''
        Flattens the convolutional kernel into a 2D matrix for computing topological loss.
        
        Returns:
            torch.Tensor: The flattened weight tensor with shape [out_channels, d].
        '''
        return self.weight.view(self.out_channels, -1)
    
    @property
    def loss_cosine(self) -> torch.Tensor:
        '''
        Calculates the cosine similarity penalty loss among the weight vectors.
        
        Returns:
            torch.Tensor: The computed cosine penalty loss.
        '''
        w_flat = self._flatten_weight()
        w_norm = F.normalize(w_flat, p=2, dim=1)
        C = torch.matmul(w_norm, w_norm.T)
        I = torch.eye(self.out_channels, device=C.device)
        return torch.sum((C * (1 - I)) ** 2) / (self.out_channels * (self.out_channels - 1))
    
    @property
    def loss_laplacian(self) -> torch.Tensor:
        '''
        Calculates the Laplacian regularization loss based on k-nearest neighbors.
        
        Returns:
            torch.Tensor: The computed Laplacian regularization loss.
        '''
        w_flat = self._flatten_weight()
        w_norm = F.normalize(w_flat, p=2, dim=1)
        C = torch.matmul(w_norm, w_norm.T)
        I = torch.eye(self.out_channels, device=C.device)
        
        _, topk_idx = torch.topk(C, self.k_neighbors + 1, dim=1)
        A = torch.zeros_like(C)
        A.scatter_(1, topk_idx, 1.0)
        A = A - I
        A = torch.max(A, A.T)
        
        return torch.sum(A * (1.0 - C)) / torch.sum(A)
    
    def compute_loss(self) -> MainfoldLoss:
        '''
        Computes both the cosine and Laplacian losses and returns them in a MainfoldLoss object.
        
        Returns:
            MainfoldLoss: An object containing the computed cosine and Laplacian losses.
        '''
        w_flat = self._flatten_weight()
        w_norm = F.normalize(w_flat, p=2, dim=1)

        C = torch.matmul(w_norm, w_norm.T)
        I = torch.eye(self.out_channels, device=C.device)

        loss_cos = torch.sum((C * (1 - I)) ** 2) / (self.out_channels * (self.out_channels - 1))
        
        _, topk_idx = torch.topk(C, self.k_neighbors + 1, dim=1)
        
        A = torch.zeros_like(C)
        A.scatter_(1, topk_idx, 1.0)
        A = A - I
        A = torch.max(A, A.T)
        
        loss_lap = torch.sum(A * (1.0 - C)) / torch.sum(A)
        
        return MainfoldLoss(cosine=loss_cos, laplacian=loss_lap)


class RiemannianManifoldConv2d(BasicManifoldConv2d):
    '''
    A 2D convolutional layer projecting patches onto a Riemannian manifold (hypersphere).
    
    Attributes:
        kappa (nn.Parameter): Concentration parameter for the von Mises-Fisher (vMF) distribution.
        lambda_rate (nn.Parameter): Gravitational attraction coefficient.
        scale (nn.Parameter): Vector amplifier for the hyperspherical network.
        bias (nn.Parameter): Manifold bias vector.
        weight_ones (torch.Tensor): Fixed all-ones kernel for computing patch norm rapidly.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        kappa_init: float = 2.0,
        lambda_init: float = 0.1,
        scale_init: float = 15.0,
        k_neighbors: int = 2,
        rule: str = 'near'
    ) -> None:
        '''
        Initializes the RiemannianManifoldConv2d layer.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the convolving kernel.
            stride (int): Stride of the convolution. Default: 1.
            padding (int): Zero-padding added to both sides of the input. Default: 0.
            dilation (int): Spacing between kernel elements. Default: 1.
            kappa_init (float): Initial value for the vMF concentration parameter.
            lambda_init (float): Initial value for the gravitational attraction coefficient.
            scale_init (float): Initial value for the vector amplifier scale.
            k_neighbors (int): Number of nearest neighbors for the Laplacian graph.
            rule (str): Attraction rule, either 'near' or 'far'.
        '''
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, k_neighbors)

        self.rule = rule.lower()
        self.kappa = nn.Parameter(torch.tensor(float(kappa_init)))
        self.lambda_rate = nn.Parameter(torch.tensor(float(lambda_init)))
        self.scale = nn.Parameter(torch.ones(out_channels) * scale_init)
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # All-ones kernel for ultra-fast calculation of patch norm
        weight_ones = torch.ones(1, in_channels, *self.kernel_size)
        self.register_buffer('weight_ones', weight_ones)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''
        Resets the parameters of the layer using Kaiming normal initialization.
        '''
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Defines the computation performed at every call.

        Args:

        Returns:
            torch.Tensor: The output manifold projection tensor.
        '''
        # 1. Weight normalization
        w_flat = self.weight.view(self.out_channels, -1)
        w_norm_flat = F.normalize(w_flat, p=2, dim=1)
        w_norm = w_norm_flat.view_as(self.weight)
        
        # 2. Ultra-fast calculation of the norm for each sliding patch of the input image
        # x_sq: [batch, 1, H_out, W_out]
        x_sq = F.conv2d(input_tensor ** 2, self.weight_ones, stride=self.stride, padding=self.padding, dilation=self.dilation)
        x_norm_val = torch.sqrt(torch.clamp(x_sq, min=1e-6))
        
        # 3. Calculate Cosine Feature Map
        # cosine: [batch, out_channels, H_out, W_out]
        conv_proj = F.conv2d(input_tensor, w_norm, stride=self.stride, padding=self.padding, dilation=self.dilation)
        cosine = conv_proj / (x_norm_val + 1e-6)
        cosine = torch.clamp(cosine, -1.0 + 1e-6, 1.0 - 1e-6)
        
        # 4. vMF gravitational field calculation (applied pixel-wise)
        theta = torch.acos(cosine)
        exp_val = torch.exp(self.kappa * (cosine - 1.0))
        attraction = exp_val if self.rule == 'near' else 1.0 - exp_val
        
        # 5. Riemannian geodesic pullback
        safe_lambda = torch.clamp(self.lambda_rate, 1e-6, 1.0 - 1e-4)
        effective_theta = theta * (1.0 - safe_lambda * attraction)
        
        # 6. Reconstruct the output (note the shape broadcasting)
        scale_view = self.scale.view(1, -1, 1, 1)
        bias_view = self.bias.view(1, -1, 1, 1)
        
        output = scale_view * torch.cos(effective_theta) + bias_view
        
        return output


class EuclideanManifoldConv2d(BasicManifoldConv2d):
    '''
    A 2D convolutional layer simulating a manifold structure in Euclidean space.
    
    Attributes:
        tau (nn.Parameter): Temperature or radius parameter for the basin of attraction.
        lambda_rate (nn.Parameter): Gravitational strength parameter.
        bias (nn.Parameter): Translation bias vector.
        weight_ones (torch.Tensor): Fixed all-ones kernel for computing patch norm rapidly.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        tau_init: float = 5.0,
        lambda_init: float = 0.5,
        k_neighbors: int = 2,
        rule: str = 'near'
    ) -> None:
        '''
        Initializes the EuclideanManifoldConv2d layer.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the convolving kernel.
            stride (int): Stride of the convolution. Default: 1.
            padding (int): Zero-padding added to both sides of the input. Default: 0.
            dilation (int): Spacing between kernel elements. Default: 1.
            tau_init (float): Initial value for the basin temperature/radius.
            lambda_init (float): Initial value for the gravitational strength.
            k_neighbors (int): Number of nearest neighbors for the Laplacian graph.
            rule (str): Attraction rule, either 'near' or 'far'.
        '''
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, k_neighbors)
        
        self.rule = rule.lower()
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        self.lambda_rate = nn.Parameter(torch.tensor(float(lambda_init)))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        weight_ones = torch.ones(1, in_channels, *self.kernel_size)
        self.register_buffer('weight_ones', weight_ones)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''
        Resets the parameters of the layer using Kaiming uniform initialization.
        '''
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Defines the computation performed at every call.

        Args:
            input_tensor (torch.Tensor): The input data tensor.

        Returns:
            torch.Tensor: The output manifold projection tensor.
        '''
        # 1. Base physical projection
        # base_proj: [batch, out_channels, H_out, W_out]
        base_proj = F.conv2d(input_tensor, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation)
        
        # 2. Ultra-fast algebraic expansion of the squared L2 distance for local patches
        # ||patch - W||^2 = ||patch||^2 + ||W||^2 - 2<patch, W>
        x_sq = F.conv2d(input_tensor ** 2, self.weight_ones, stride=self.stride, padding=self.padding, dilation=self.dilation)
        w_sq = torch.sum(self.weight ** 2, dim=(1,2,3)).view(1, -1, 1, 1)
        
        dist_sq = x_sq + w_sq - 2 * base_proj
        dist_sq = torch.clamp(dist_sq, min=1e-6)
        
        # 3. Compute the attraction index
        exp_val = torch.exp(-dist_sq / (self.tau ** 2 + 1e-8))
        attraction = exp_val if self.rule == 'near' else 1.0 - exp_val
        
        # 4. Gravitational correction
        safe_lambda = torch.clamp(self.lambda_rate, 1e-6, 1.0 - 1e-4)
        correction = safe_lambda * attraction * (w_sq - base_proj)
        
        # 5. Combine outputs
        output = base_proj + correction + self.bias.view(1, -1, 1, 1)
        
        return output
