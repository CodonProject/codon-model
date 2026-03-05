from codon.base import *
from codon.utils.theta import validate_rope_config

import math
import warnings


class BaseEmbedding(BasicModel):
    '''
    Base class for Positional Embeddings.
    '''

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None, start_pos: int = 0) -> torch.Tensor:
        '''
        Forward pass for positional embedding.

        Args:
            x (torch.Tensor): Input tensor.
            positions (torch.Tensor, optional): Position indices. Defaults to None.
            start_pos (int, optional): Starting position. Defaults to 0.

        Returns:
            torch.Tensor: Output tensor with positional information.
        '''
        raise NotImplementedError


class SinusoidalEmbedding(BaseEmbedding):
    '''
    Sinusoidal absolute positional encoding.

    Implements the standard sinusoidal positional encoding proposed in "Attention Is All You Need".
    Uses sine and cosine functions of different frequencies:
        PE(pos, 2i) = sin(pos / base^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / base^(2i/d_model))

    Attributes:
        model_dim (int): The dimension of the model.
        max_len (int): Maximum sequence length.
        base (int): Base for computing frequencies.
        pe (torch.Tensor): Buffer containing the positional encodings. Shape: [1, max_len, model_dim].
    '''

    def __init__(self, model_dim: int, max_len: int = 131072, base: int = 500000):
        '''
        Initializes the absolute positional encoding module.

        Args:
            model_dim (int): The dimension of the model.
            max_len (int, optional): Maximum sequence length. Defaults to 131072.
            base (int, optional): Base for computing frequencies. Defaults to 500000.
        '''
        super().__init__()

        self.model_dim = model_dim
        self.max_len = max_len
        self.base = base

        config = validate_rope_config(self.max_len, self.base)
        if not config.is_passed:
            warnings.warn(f'Sinusoidal validation failed: {config.info}. Suggested base: {config.suggested_base}')
        
        pe = torch.zeros(max_len, model_dim)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(base) / model_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None, start_pos: int = 0) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor. Shape: [Batch_Size, Seq_Len, model_dim].
            positions (torch.Tensor, optional): Explicit position indices. Shape: [Batch_Size, Seq_Len].
                                                If provided, retrieves embeddings for these indices.
            start_pos (int, optional): Starting position index. Used if positions is None. Defaults to 0.

        Returns:
            torch.Tensor: Tensor with positional encoding added.
        '''
        if positions is not None:
            # pe: [1, max_len, dim] -> [max_len, dim] -> [Batch, Seq_Len, Dim]
            pe = self.pe.squeeze(0)[positions]
        else:
            seq_len = x.size(1)
            pe = self.pe[:, start_pos : start_pos + seq_len, :]

        return x + pe


class BaseRotaryEmbedding(BaseEmbedding):
    '''
    Base class for Rotary Positional Embeddings.

    Attributes:
        model_dim (int): The dimension of the model.
        max_len (int): Maximum sequence length.
        base (int): Base for computing frequencies.
        cos_cached (torch.Tensor): Cached cosine values.
        sin_cached (torch.Tensor): Cached sine values.
    '''

    def __init__(self, model_dim: int, max_len: int = 131072, base: int = 500000) -> None:
        '''
        Initializes the BaseRotaryEmbedding module.

        Args:
            model_dim (int): The dimension of the model.
            max_len (int, optional): Maximum sequence length. Defaults to 131072.
            base (int, optional): Base for computing frequencies. Defaults to 500000.
        '''
        super().__init__()
        self.model_dim = model_dim
        self.max_len = max_len
        self.base = base

        config = validate_rope_config(self.max_len, self.base)
        if not config.is_passed:
            warnings.warn(f'RoPE validation failed: {config.info}. Suggested base: {config.suggested_base}')

        inv_freq = 1.0 / (base ** (torch.arange(0, model_dim, 2).float() / model_dim))

        t = torch.arange(max_len, dtype=torch.float)

        freqs = torch.outer(t, inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Split the vector into two halves and rotate them: [-x2, x1].
        The split operation is performed on the last dimension (model_dim),
        regardless of whether the input is 3D or 4D.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Rotated tensor.
        '''
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(BaseRotaryEmbedding):
    '''
    Rotary Positional Embedding (RoPE).

    Attributes:
        model_dim (int): The dimension of the model.
        max_len (int): Maximum sequence length.
        base (int): Base for computing frequencies.
        cos_cached (torch.Tensor): Cached cosine values.
        sin_cached (torch.Tensor): Cached sine values.
    '''

    def __init__(self, model_dim: int, max_len: int = 131072, base: int = 500000) -> None:
        '''
        Initialize the RoPE module.

        Args:
            model_dim (int): The dimension of the model (or head_dim). Must be even.
            max_len (int, optional): Maximum sequence length for pre-computing position encodings. 
                                     Defaults to 131072.
            base (int, optional): Base for computing frequencies. Defaults to 500000.
        '''
        super().__init__(model_dim, max_len, base)

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None, start_pos: int = 0) -> torch.Tensor:
        '''
        Apply rotary positional encoding.
        
        Automatically adapts to two types of inputs:
        1. [Batch, Seq_Len, Dim]
        2. [Batch, Head, Seq_Len, Head_Dim]

        Args:
            x (torch.Tensor): Input tensor.
            positions (torch.Tensor, optional): Explicit position indices. Shape: [Batch, Seq_Len].
                If provided, uses these indices to retrieve positional embeddings.
            start_pos (int, optional): Starting position index for KV Cache inference.
                                       Used if positions is None. Defaults to 0.

        Returns:
            torch.Tensor: Tensor with positional information added.
        '''
        ndim = x.ndim
        seq_len = x.shape[-2]

        if positions is not None:
            # positions: [Batch, Seq_Len] -> cos/sin: [Batch, Seq_Len, Dim]
            cos = self.cos_cached[positions]
            sin = self.sin_cached[positions]
            
            if ndim == 4:
                # [Batch, Seq_Len, Dim] -> [Batch, 1, Seq_Len, Dim]
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
        else:
            cos = self.cos_cached[start_pos : start_pos + seq_len, :]
            sin = self.sin_cached[start_pos : start_pos + seq_len, :]
            
            shape = [1] * (ndim - 2) + [seq_len, -1]
            cos = cos.view(*shape)
            sin = sin.view(*shape)

        return (x * cos) + (self._rotate_half(x) * sin)


class InterleavedRotaryEmbedding(BaseRotaryEmbedding):
    '''
    Interleaved Multimodal Rotary Positional Embedding (MRoPE-Interleave).
    
    Supports multi-dimensional positions (e.g., 3D for video: time, height, width), 
    with frequency channels assigned in a rotating interleaved manner across dimensions.

    Attributes:
        model_dim (int): The dimension of the model.
        max_len (int): Maximum sequence length.
        base (int): Base for computing frequencies.
        cos_cached (torch.Tensor): Cached cosine values.
        sin_cached (torch.Tensor): Cached sine values.
        num_axes (int): Number of positional axes.
        axis_mask (torch.Tensor): Mask for assigning frequency channels to axes.
        interleave_idx (torch.Tensor): Indices for interleaving.
    '''

    def __init__(self, model_dim: int, max_len: int = 131072, base: int = 500000, num_axes: int = 3) -> None:
        '''
        Initializes the MRoPEInterleaved module.

        Args:
            model_dim (int): The dimension of the model. Must be even and divisible by num_axes.
            max_len (int, optional): Maximum sequence length for pre-computing position encodings.
                                     Defaults to 131072.
            base (int, optional): Base for computing frequencies. Defaults to 500000.
            num_axes (int, optional): Number of positional axes (e.g., 3 for time, height, width).
                                      Defaults to 3.
        '''
        assert model_dim % 2 == 0, 'model_dim must be even'
        assert model_dim % num_axes == 0, f'model_dim {model_dim} not divisible by num_axes {num_axes}'
        
        super().__init__(model_dim, max_len, base)
        
        self.num_axes = num_axes
        
        self.register_buffer(
            'axis_mask', 
            torch.arange(model_dim) % num_axes, 
            persistent=False
        )
        
        k = model_dim // num_axes
        idx = []
        for p in range(model_dim):
            j = p % num_axes
            i = p // num_axes
            pos_in_old = j * k + i
            idx.append(pos_in_old)
            
        self.register_buffer('interleave_idx', torch.tensor(idx, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None, start_pos: int = 0) -> torch.Tensor:
        '''
        Apply multimodal rotary positional encoding.

        Args:
            x (torch.Tensor): Input tensor. Shape: [Batch, Seq_Len, Dim] or [Batch, Head, Seq_Len, Head_Dim].
            positions (torch.Tensor, optional): Position index tensor. Shape: [Batch, Seq_Len] or
                [Batch, Seq_Len, num_axes].
                If 2D tensor, it will be automatically expanded to [Batch, Seq_Len, num_axes].
                If None and num_axes=1, linear position indices will be automatically created.
            start_pos (int, optional): Starting position index. Defaults to 0.

        Returns:
            torch.Tensor: Tensor with positional information added.
        
        Raises:
            ValueError: If positions is None and num_axes > 1.
        '''
        ndim = x.ndim
        seq_len = x.shape[-2]
        batch_size = x.shape[0]
        
        if positions is None:
            if self.num_axes == 1:
                positions = torch.arange(0, seq_len, device=x.device, dtype=torch.long)
            else:
                raise ValueError('positions must be provided when num_axes > 1 (e.g. for vision/multimodal inputs)')
        
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.num_axes)
            
        if positions.ndim == 2:
            positions = positions.unsqueeze(-1).expand(-1, -1, self.num_axes)
            
        if positions.ndim == 3 and positions.shape[-1] == 1:
            positions = positions.expand(-1, -1, self.num_axes)
            
        batch_size = positions.shape[0]
        
        cos_list, sin_list = [], []
        
        for ax in range(self.num_axes):
            pos_ax = positions[..., ax]
            pos_ax = torch.clamp(pos_ax + start_pos, 0, self.max_len - 1).long()
            
            cos_full = self.cos_cached[pos_ax]
            sin_full = self.sin_cached[pos_ax]
            
            mask = (self.axis_mask == ax)
            cos_ax = cos_full[..., mask]
            sin_ax = sin_full[..., mask]
            
            cos_list.append(cos_ax)
            sin_list.append(sin_ax)
        
        cos_all = torch.cat(cos_list, dim=-1)
        sin_all = torch.cat(sin_list, dim=-1)
        
        cos_all = cos_all[..., self.interleave_idx]
        sin_all = sin_all[..., self.interleave_idx]
        
        if ndim == 4:
            shape = [batch_size, 1, seq_len, -1]
            cos_all = cos_all.view(*shape)
            sin_all = sin_all.view(*shape)
        
        return (x * cos_all) + (self._rotate_half(x) * sin_all)
