from codon import *
from codon.block.mlp import MLP


class AdaLayerNorm(BasicModel):
    '''
    Adaptive Layer Normalization (AdaLayerNorm) module.

    This module normalizes the input tensor and then applies scale and shift
    parameters computed from a conditional embedding using an MLP.

    Attributes:
        features_dim (int): Dimension of the input features.
        embedding_dim (int): Dimension of the embedding features.
        mlp (MLP): MLP module to predict scale and shift parameters.
    '''
    def __init__(
        self,
        features_dim: int,
        embedding_dim: int,
        hidden_features: int = None,
        mlp: MLP = None
    ) -> None:
        '''
        Initialize the AdaLayerNorm module.

        Args:
            features_dim (int): Dimension of the input features.
            embedding_dim (int): Dimension of the embedding features.
            hidden_features (int, optional): Dimension of hidden features in MLP. Defaults to None.
        '''
        super().__init__()
        self.features_dim = features_dim
        self.embedding_dim = embedding_dim

        if hidden_features is None: hidden_features = features_dim

        self.mlp = MLP(
            in_features=embedding_dim,
            hidden_features=hidden_features,
            out_features=features_dim*2
        ) if mlp is None else mlp
    
    def forward(self, input_tensor: torch.Tensor, embedding_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            input_tensor (torch.Tensor): Input tensor to be normalized.
            embedding_tensor (torch.Tensor): Condition embedding tensor.

        Returns:
            torch.Tensor: Normalized and modulated output tensor.
        '''
        normalized_tensor = F.layer_norm(
            input_tensor, 
            (self.features_dim,), 
            eps=1e-5
        )

        scale_shift: torch.Tensor = self.mlp(embedding_tensor)

        scale, shift = scale_shift.chunk(2, dim=-1)

        output = normalized_tensor * scale + shift
        return output


class MixedLayerNorm(BasicModel):
    '''
    Mixed Adaptive Layer Normalization (MixedLayerNorm) module.

    This module normalizes the input tensor and then applies scale (gamma) and 
    shift (beta) parameters computed from a conditional embedding. During training, 
    it applies Mixup regularization on the predicted scale and shift parameters to 
    enhance generalization.

    Attributes:
        features_dim (int): Dimension of the input features.
        condition_dim (int): Dimension of the condition embedding features.
        eps (float): A value added to the denominator for numerical stability.
        beta_distribution (torch.distributions.Beta): Beta distribution for Mixup sampling.
        affine (nn.Linear): Linear layer to predict scale and shift parameters.
    '''
    def __init__(
        self,
        features_dim: int,
        condition_dim: int,
        beta_concentration: float = 0.2,
        eps: float = 1e-5,
        bias: bool = True
    ) -> None:
        '''
        Initialize the MixedLayerNorm module.

        Args:
            features_dim (int): Dimension of the input features.
            condition_dim (int): Dimension of the condition embedding features.
            beta_concentration (float, optional): Concentration parameter for Beta distribution. Defaults to 0.2.
            eps (float, optional): Small value for numerical stability. Defaults to 1e-5.
            bias (bool, optional): If True, adds a learnable bias to the affine layer. Defaults to True.
        '''
        super().__init__()
        self.features_dim = features_dim
        self.condition_dim = condition_dim
        self.eps = eps
        # Define Beta distribution for Mixup interpolation
        self.beta_distribution = torch.distributions.Beta(
            beta_concentration,
            beta_concentration
        )
        # Map condition embedding to scale (gamma) and shift (beta) parameters
        self.affine = nn.Linear(condition_dim, features_dim * 2, bias=bias)
        
        # Initialize weights and biases for stable training start
        nn.init.xavier_uniform_(self.affine.weight)
        if self.affine.bias is not None:
            with torch.no_grad():
                self.affine.bias[:features_dim] = 0.0  # Initialize betas (shift) to 0
                self.affine.bias[features_dim:] = 1.0  # Initialize gammas (scale) to 1
        
    def forward(
        self, 
        input_tensor: torch.Tensor, 
        condition_tensor: torch.Tensor
    ) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            input_tensor (torch.Tensor): Input tensor to be normalized.
            condition_tensor (torch.Tensor): Condition embedding tensor.
        
        Returns:
            torch.Tensor: Normalized and modulated output tensor.
        '''
        # 1. Standard Layer Normalization (without learnable weight/bias)
        normalized_tensor = F.layer_norm(
            input_tensor,
            (self.features_dim,),
            eps=self.eps
        )
        # 2. Predict affine parameters from condition
        affine_params = self.affine(condition_tensor)
        if affine_params.ndim == 2:
            affine_params = affine_params.unsqueeze(1)
            
        betas, gammas = torch.split(affine_params, self.features_dim, dim=-1)
        # 3. In evaluation mode or when batch size is 1, skip Mixup
        if not self.training or input_tensor.size(0) == 1:
            output = gammas * normalized_tensor + betas
            return output
        # 4. Apply Mixup regularization during training
        batch_size = input_tensor.size(0)
        device = input_tensor.device
        # Shuffle parameters within the batch
        shuffle_indices = torch.randperm(batch_size, device=device)
        shuffled_betas = betas[shuffle_indices]
        shuffled_gammas = gammas[shuffle_indices]
        # Sample mixup ratio lambda from Beta distribution
        beta_samples = self.beta_distribution.sample((batch_size, 1, 1)).to(device)
        # Interpolate between original and shuffled parameters
        mixed_betas = beta_samples * betas + (1.0 - beta_samples) * shuffled_betas
        mixed_gammas = beta_samples * gammas + (1.0 - beta_samples) * shuffled_gammas
        # Apply the mixed scale and shift
        output = mixed_gammas * normalized_tensor + mixed_betas
        return output