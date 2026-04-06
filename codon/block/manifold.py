from codon.base import *

import torch.nn.functional as F

from typing import Tuple
from dataclasses import dataclass

from codon.ops.manifold import riemannian_manifold_linear, riemannian_manifold_conv2d


@dataclass
class MainfoldLoss:
    '''
    Dataclass for storing manifold-related loss components.

    Attributes:
        cosine (torch.Tensor): The cosine similarity loss.
        laplacian (torch.Tensor): The Laplacian regularization loss.
    '''
    cosine: torch.Tensor
    laplacian: torch.Tensor

    def factor_loss(self, factor_cos: float = 0.013, factor_lap: float = 0.012) -> torch.Tensor:
        '''
        Calculates the weighted sum of cosine and Laplacian losses.

        Args:
            factor_cos (float): The weight factor for the cosine loss.
            factor_lap (float): The weight factor for the Laplacian loss.

        Returns:
            torch.Tensor: The calculated total loss value.
        '''
        return self.cosine * factor_cos + self.laplacian * factor_lap


class BasicManifoldLinear(BasicModel):
    '''
    Base class for manifold-based neural network layers.

    Attributes:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        k_neighbors (int): Number of nearest neighbors to consider for Laplacian loss.
        weight (nn.Parameter): The learnable weights of the layer.
    '''

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k_neighbors: int = 2,
    ) -> None:
        '''
        Initializes the BasicManifoldLinear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            k_neighbors (int): Number of nearest neighbors for the Laplacian graph.
        '''
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.k_neighbors = min(k_neighbors, out_features - 1)

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
    
    @property
    def loss_cosine(self) -> torch.Tensor:
        '''
        Calculates the cosine similarity penalty loss among the weight vectors.
        
        Returns:
            torch.Tensor: The computed cosine penalty loss.
        '''
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        C = torch.matmul(w_norm, w_norm.T)
        I = torch.eye(self.out_features, device=C.device)

        return torch.sum((C * (1 - I)) ** 2) / (self.out_features * (self.out_features - 1))
    
    @property
    def loss_laplacian(self) -> torch.Tensor:
        '''
        Calculates the Laplacian regularization loss based on k-nearest neighbors.
        
        Returns:
            torch.Tensor: The computed Laplacian regularization loss.
        '''
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        C = torch.matmul(w_norm, w_norm.T)
        I = torch.eye(self.out_features, device=C.device)
        
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
        w_norm = F.normalize(self.weight, p=2, dim=1)

        C = torch.matmul(w_norm, w_norm.T)
        I = torch.eye(self.out_features, device=C.device)

        loss_cos = torch.sum((C * (1 - I)) ** 2) / (self.out_features * (self.out_features - 1))
        
        _, topk_idx = torch.topk(C, self.k_neighbors + 1, dim=1)
        
        A = torch.zeros_like(C)
        A.scatter_(1, topk_idx, 1.0)
        A = A - I
        A = torch.max(A, A.T)
        
        loss_lap = torch.sum(A * (1.0 - C)) / torch.sum(A)
        
        return MainfoldLoss(cosine=loss_cos, laplacian=loss_lap)


class RiemannianManifoldLinear(BasicManifoldLinear):
    '''
    A linear layer projecting data onto a Riemannian manifold (hypersphere).
    
    Attributes:
        kappa (nn.Parameter): Concentration parameter for the von Mises-Fisher (vMF) distribution.
        lambda_rate (nn.Parameter): Gravitational attraction coefficient.
        scale (nn.Parameter): Vector amplifier for the hyperspherical network.
        bias (nn.Parameter): Manifold bias vector.
    '''

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kappa_init: float = 2.0,
        lambda_init: float = 0.1,
        scale_init: float = 15.0,
        k_neighbors: int = 2,
        rule: str = 'near'
    ) -> None:
        '''
        Initializes the RiemannianManifoldLinear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            kappa_init (float): Initial value for the vMF concentration parameter.
            lambda_init (float): Initial value for the gravitational attraction coefficient.
            scale_init (float): Initial value for the vector amplifier scale.
            k_neighbors (int): Number of nearest neighbors for the Laplacian graph.
            rule (str): Attraction rule, either 'near' or 'far'.
        '''
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            k_neighbors=k_neighbors
        )

        self.kappa_init = kappa_init
        self.lambda_init = lambda_init
        self.scale_init = scale_init
        self.rule = rule.lower()

        if not self.rule in ['far', 'near']:
            raise ValueError(f"Invalid rule: {self.rule}, must be 'far' or 'near'")
        
        # Concentration parameter for the vMF distribution
        self.kappa = nn.Parameter(torch.tensor(float(kappa_init)))
        
        # Gravitational attraction coefficient
        self.lambda_rate = nn.Parameter(torch.tensor(float(lambda_init)))
        
        # Vector amplifier for the hyperspherical network
        self.scale = nn.Parameter(torch.ones(out_features) * scale_init)

        # Manifold bias vector
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''
        Resets the parameters of the layer.
        '''
        nn.init.normal_(self.weight, 0, 0.01)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Defines the computation performed at every call.

        Args:
            input_tensor (torch.Tensor): The input data with shape (batch_size, in_features).

        Returns:
            torch.Tensor: The output data with shape (batch_size, out_features).
        '''
        return riemannian_manifold_linear(
            input_tensor=input_tensor,
            weight=self.weight,
            kappa=self.kappa,
            lambda_rate=self.lambda_rate,
            scale=self.scale,
            bias=self.bias,
            rule=self.rule
        )


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
        use_norm (bool): Whether to scale the output by the input patch norm.
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
        rule: str = 'near',
        use_norm_gate: bool = False,
        use_norm: bool = False
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
            use_norm (bool): Whether to scale the output by the input patch norm. Default: True.
        '''
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, k_neighbors)

        self.rule = rule.lower()
        self.kappa = nn.Parameter(torch.tensor(float(kappa_init)))
        self.lambda_rate = nn.Parameter(torch.tensor(float(lambda_init)))
        self.scale = nn.Parameter(torch.ones(out_channels) * scale_init)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.use_norm_gate = use_norm_gate
        self.use_norm = use_norm

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
        return riemannian_manifold_conv2d(
            input_tensor=input_tensor,
            weight=self.weight,
            weight_ones=self.weight_ones,
            kappa=self.kappa,
            lambda_rate=self.lambda_rate,
            scale=self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            rule=self.rule,
            use_norm=self.use_norm
        )