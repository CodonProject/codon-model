from codon import *


class RMSNorm(BasicModel):
    '''
    Root Mean Square Normalization (RMSNorm).
    Formula: y = (x / RMS(x)) * gamma
    Where gamma is initialized to 1.
    '''
    def __init__(self, d_model: int, eps: float = 1e-6, channel_first: bool = False):
        '''
        Initializes RMSNorm.
        Args:
            d_model (int): Dimension of the features to be normalized.
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-6.
            channel_first (bool, optional): If True, features are in the 1st dimension [B, C, ...].
                If False, features are in the last dimension [B, ..., C]. Defaults to False.
        '''
        super().__init__()
        self.eps = eps
        self.channel_first = channel_first
        
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.
        '''
        dim = 1 if self.channel_first else -1
        
        orig_dtype = x.dtype 
        
        x_f32 = x.float()
        
        rms = torch.rsqrt(x_f32.pow(2).mean(dim=dim, keepdim=True) + self.eps)
        
        x_normed = (x_f32 * rms).to(orig_dtype)
        
        gamma = self.gamma
        if self.channel_first:
            for _ in range(x.ndim - gamma.ndim - 1):
                gamma = gamma.unsqueeze(-1)
            gamma = gamma.unsqueeze(0)
        
        return x_normed * gamma.to(orig_dtype)


class ZCRMSNorm(BasicModel):
    '''
    Zero-Centered Root Mean Square Normalization (ZCRMSNorm).

    A variant of RMSNorm where the scale parameter gamma is zero-centered (initialized to 0).
    Formula: y = (x / RMS(x)) * (1 + gamma)
    This preserves identity mapping at initialization and stabilizes gradients in deep networks.
    '''
    def __init__(self, d_model: int, eps: float = 1e-6, channel_first: bool = False):
        '''
        Initializes ZCRMSNorm.

        Args:
            d_model (int): Dimension of the features to be normalized.
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-6.
            channel_first (bool, optional): If True, features are in the 1st dimension [B, C, ...].
                If False, features are in the last dimension [B, ..., C]. Defaults to False.
        '''
        super().__init__()
        self.eps = eps
        self.channel_first = channel_first
        
        self.gamma = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.
        '''
        dim = 1 if self.channel_first else -1
        
        orig_dtype = x.dtype 
        
        x_f32 = x.float()
        
        rms = torch.rsqrt(x_f32.pow(2).mean(dim=dim, keepdim=True) + self.eps)
        
        x_normed = (x_f32 * rms).to(orig_dtype)
        
        gamma = self.gamma
        if self.channel_first:
            for _ in range(x.ndim - gamma.ndim - 1):
                gamma = gamma.unsqueeze(-1)
            gamma = gamma.unsqueeze(0)
        
        return x_normed * (1.0 + gamma.to(orig_dtype))