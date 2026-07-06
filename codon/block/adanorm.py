from codon import *
from codon.block import MLP


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
        hidden_features: int = None
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
        )
    
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