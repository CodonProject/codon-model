from codon.base import *

from typing import Optional, List


class LowRankFusion(BasicModel):
    '''
    Low-rank Multimodal Fusion (LMF) module.

    Approximates the multimodal outer product using low-rank decomposition, efficiently
    implementing fusion by decomposing the weight tensor into a combination of
    modality-specific factors.
    '''

    def __init__(self, in_features: List[int], out_features: int, rank: int, dropout: float = 0.0, channel_first: bool = False):
        '''
        Initializes LowRankFusion.

        Args:
            in_features (List[int]): List of feature dimensions for each input modality.
            out_features (int): Output feature dimension after fusion.
            rank (int): Rank of the low-rank decomposition.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            channel_first (bool, optional): Whether feature dimension is at index 1 (e.g., CNN [B, C, H, W]).
                If False, features are assumed to be at the last dimension (e.g., Transformer [B, L, C]).
                Defaults to False.
        '''
        super().__init__()
        
        if not in_features:
            raise ValueError('in_features cannot be empty')

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.channel_first = channel_first
        
        self.modality_factors = nn.ModuleList([
            nn.Linear(dim, rank, bias=True) for dim in in_features
        ])
        
        self.fusion_weights = nn.Linear(rank, out_features, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights(self)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            inputs (List[torch.Tensor]): List of input tensors. All tensors should match in dimensions
                except for the last dimension (feature dimension).

        Returns:
            torch.Tensor: The fused output tensor.
        '''
        if len(inputs) != len(self.in_features):
            raise ValueError(f'Expected {len(self.in_features)} inputs, got {len(inputs)}')
        
        if self.channel_first:
            inputs = [x.movedim(1, -1) for x in inputs]
            
        fusion_tensor: Optional[torch.Tensor] = None
        
        for i, x in enumerate(inputs):
            projected = self.modality_factors[i](x)
            
            if fusion_tensor is None:
                fusion_tensor = projected
            else:
                fusion_tensor = fusion_tensor * projected
        
        if fusion_tensor is None:
             raise ValueError('No inputs processed')

        output = self.fusion_weights(self.dropout(fusion_tensor))
        
        if self.channel_first:
            output = output.movedim(-1, 1)
        
        return output


class GatedMultimodalUnit(BasicModel):
    '''
    Gated Multimodal Unit (GMU) module.

    Controls the contribution of each modality to the final fused representation
    via a learned gating mechanism.
    '''

    def __init__(self, in_features: List[int], out_features: int, channel_first: bool = False):
        '''
        Initializes GatedMultimodalUnit.

        Args:
            in_features (List[int]): List of feature dimensions for each input modality.
            out_features (int): Hidden layer feature dimension.
            channel_first (bool, optional): Whether feature dimension is at index 1. Defaults to False.
        '''
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.channel_first = channel_first
        
        self.feature_transforms = nn.ModuleList([
            nn.Linear(dim, out_features) for dim in in_features
        ])
        
        total_in_features = sum(in_features)
        self.gate_net = nn.Linear(total_in_features, len(in_features))
        
        self._init_weights(self)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            inputs (List[torch.Tensor]): List of input tensors.

        Returns:
            torch.Tensor: The fused output tensor.
        '''
        if len(inputs) != len(self.in_features):
            raise ValueError(f'Expected {len(self.in_features)} inputs, got {len(inputs)}')
        
        processed_inputs = []
        if self.channel_first:
            processed_inputs = [x.movedim(1, -1) for x in inputs]
        else:
            processed_inputs = inputs
            
        hidden_features = []
        for i, x in enumerate(processed_inputs):
            h = torch.tanh(self.feature_transforms[i](x))
            hidden_features.append(h)
            
        concatenated_input = torch.cat(processed_inputs, dim=-1)
        gate_logits = self.gate_net(concatenated_input) # (B, ..., num_modalities)
        gates = torch.softmax(gate_logits, dim=-1)
        
        output = torch.zeros_like(hidden_features[0])
        
        for i, h in enumerate(hidden_features):
            g = gates[..., i:i+1] # (B, ..., 1)
            output += g * h
            
        if self.channel_first:
            output = output.movedim(-1, 1)
            
        return output


class DiffusionMapsFusion(BasicModel):
    '''
    Diffusion Maps Fusion module.

    Based on the idea of diffusion maps in manifold learning.
    Constructs a graph Laplacian (or normalized affinity matrix) across feature channels
    to perform feature alignment and cross-diffusion in the manifold space.

    Currently only supports fusion of exactly two modalities.
    '''

    def __init__(self, in_features: List[int], out_features: int, sigma: float = 1.0, channel_first: bool = False):
        '''
        Initializes DiffusionMapsFusion.

        Args:
            in_features (List[int]): List of feature dimensions for two input modalities.
            out_features (int): Output feature dimension.
            sigma (float, optional): Bandwidth parameter for the Gaussian kernel. Defaults to 1.0.
            channel_first (bool, optional): Whether feature dimension is at index 1. Defaults to False.
        '''
        super().__init__()
        
        if len(in_features) != 2:
            raise ValueError('DiffusionMapsFusion currently only supports exactly 2 modalities.')
            
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma
        self.channel_first = channel_first
        
        self.projections = nn.ModuleList([
            nn.Linear(dim, out_features) for dim in in_features
        ])
        
        self.output_proj = nn.Linear(out_features * 2, out_features)
        
        self._init_weights(self)
        
    def _compute_affinity(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Computes the normalized affinity matrix (Diffusion Operator) between feature channels.

        Args:
            x (torch.Tensor): Input features (C, N), where C is the number of feature channels
                and N is the number of samples.

        Returns:
            torch.Tensor: Normalized affinity matrix (C, C).
        '''
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 <x_i, x_j>
        sq_norm = (x ** 2).sum(1, keepdim=True)
        dist_sq = sq_norm + sq_norm.t() - 2 * torch.mm(x, x.t())
        
        W = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        
        D_inv = 1.0 / (W.sum(1, keepdim=True) + 1e-8)
        P = D_inv * W
        
        return P

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            inputs (List[torch.Tensor]): List of two input tensors.

        Returns:
            torch.Tensor: The fused output tensor.
        '''
        if len(inputs) != 2:
            raise ValueError('Expected exactly 2 inputs')
            
        processed_inputs = []
        if self.channel_first:
            processed_inputs = [x.movedim(1, -1) for x in inputs]
        else:
            processed_inputs = inputs
            
        proj_feats = [self.projections[i](x) for i, x in enumerate(processed_inputs)]
        xA, xB = proj_feats[0], proj_feats[1]
        
        flat_xA = xA.reshape(-1, xA.shape[-1])
        flat_xB = xB.reshape(-1, xB.shape[-1])
        
        xA_T = flat_xA.t()
        xB_T = flat_xB.t()
        
        P_A = self._compute_affinity(xA_T) # (C, C)
        P_B = self._compute_affinity(xB_T) # (C, C)
        
        diffused_A_T = torch.mm(P_B, xA_T) # (C, N_samples)
        diffused_B_T = torch.mm(P_A, xB_T) # (C, N_samples)
        
        diffused_A = diffused_A_T.t().view(xA.shape)
        diffused_B = diffused_B_T.t().view(xB.shape)
        
        combined = torch.cat([diffused_A, diffused_B], dim=-1)
        output = self.output_proj(combined)
        
        if self.channel_first:
            output = output.movedim(-1, 1)
            
        return output


class CompactMultimodalPooling(BasicModel):
    '''
    Compact Multimodal Pooling (MCB/CBP) module.

    Approximates the outer product of multimodal features using Count Sketch and FFT.
    Supports fusion of two or more modalities.
    '''

    def __init__(self, in_features: List[int], out_features: int, channel_first: bool = False):
        '''
        Initializes CompactMultimodalPooling.

        Args:
            in_features (List[int]): List of feature dimensions for each input modality.
            out_features (int): Output feature dimension. Should typically be higher than
                input dimensions to preserve information.
            channel_first (bool, optional): Whether feature dimension is at index 1 (e.g., CNN [B, C, H, W]).
                If False, features are assumed to be at the last dimension (e.g., Transformer [B, L, C]).
                Defaults to False.
        '''
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.channel_first = channel_first
        
        for i, dim in enumerate(in_features):
            self.register_buffer(f'h_{i}', torch.randint(0, out_features, (dim,)))
            self.register_buffer(f's_{i}', torch.randint(0, 2, (dim,)) * 2 - 1) # Map {0, 1} to {-1, 1}

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            inputs (List[torch.Tensor]): List of input tensors.

        Returns:
            torch.Tensor: The fused output tensor.
        '''
        if len(inputs) != len(self.in_features):
            raise ValueError(f'Expected {len(self.in_features)} inputs, got {len(inputs)}')
        
        if self.channel_first:
            inputs = [x.movedim(1, -1) for x in inputs]
            
        batch_size = inputs[0].size(0)
        fft_product: Optional[torch.Tensor] = None
        
        for i, x in enumerate(inputs):
            h = getattr(self, f'h_{i}') # (dim,)
            s = getattr(self, f's_{i}') # (dim,)
            
            output_shape = list(x.shape)
            output_shape[-1] = self.out_features
            sketch = torch.zeros(output_shape, device=x.device, dtype=x.dtype)
            
            weighted_x = x * s # (..., dim)
            
            flat_x = weighted_x.reshape(-1, weighted_x.shape[-1]) # (N, dim)
            flat_sketch = sketch.view(-1, self.out_features) # (N, out)
            
            h_expanded = h.expand(flat_x.shape[0], -1)
            
            flat_sketch.scatter_add_(1, h_expanded, flat_x)
            
            sketch = flat_sketch.view(output_shape)
            
            fft_x = torch.fft.rfft(sketch, dim=-1)
            
            if fft_product is None:
                fft_product = fft_x
            else:
                fft_product = fft_product * fft_x
        
        if fft_product is None:
            raise ValueError('No inputs processed')
            
        output = torch.fft.irfft(fft_product, n=self.out_features, dim=-1)
        
        if self.channel_first:
            output = output.movedim(-1, 1)
        
        return output
