from codon import *
import json


@dataclass
class AutoVisionEncoderOutput:
    '''
    Output of autoencoder vision model encoder.

    Attributes:
        z_q (torch.Tensor): Quantized latent tensor.
        loss (torch.Tensor): Quantization loss.
        indices (torch.Tensor): Quantized indices.
        grid_shape (tuple): Grid shape as (num_patches_h, num_patches_w).
        entropy (torch.Tensor): Average bit-wise entropy from codebook.
        perplexity (torch.Tensor): Perplexity calculated as 2^entropy.
        hidden_states (torch.Tensor): Hidden states before quantization.
    '''
    z_q: torch.Tensor
    loss: torch.Tensor = None
    indices: torch.Tensor = None
    grid_shape: tuple = None
    entropy: torch.Tensor = None
    perplexity: torch.Tensor = None
    hidden_states: torch.Tensor = None


@dataclass
class AutoVisionDecoderOutput:
    '''
    Output of autoencoder vision model decoder.

    Attributes:
        reconstructed (torch.Tensor): Reconstructed output tensor.
        grid_shape (tuple): Grid shape as (num_patches_h, num_patches_w).
        hidden_states (torch.Tensor): Hidden states after attention.
    '''
    reconstructed: torch.Tensor
    grid_shape: tuple = None
    hidden_states: torch.Tensor = None


class KVCache:
    '''
    Key-Value Cache container to manage dynamic state during autoregressive generation.
    '''
    def __init__(self) -> None:
        # states list: each element is a tuple of (past_key, past_value) for a layer
        self.states: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def update(self, next_states: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        '''Update the cache states with the newly computed KV projections.'''
        self.states = next_states

    @property
    def current_len(self) -> int:
        '''Return the current sequence length of stored keys.'''
        if not self.states or self.states[0] is None:
            return 0
        # Shape of key is expected to be [batch, heads, seq_len, head_dim]
        return self.states[0][0].shape[-2]

    def clear(self) -> None:
        '''Flush the cache.'''
        self.states = []



class AutoencoderVisionModel(BasicModel):
    '''
    Base class for autoencoder vision models with encoding/decoding capabilities.

    Attributes:
        gradient_checkpointing (bool): Whether gradient checkpointing is enabled.
    '''
    def __init__(self):
        super().__init__()
        self.codebook_size: int = 0

    @staticmethod
    def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, max_value: float = 1.0) -> torch.Tensor:
        '''
        Compute Peak Signal-to-Noise Ratio between two images.

        Args:
            img1 (torch.Tensor): Reference image tensor.
            img2 (torch.Tensor): Comparison image tensor.
            max_value (float): Maximum possible pixel value. Defaults to 1.0.

        Returns:
            torch.Tensor: PSNR value in dB (higher is better).
        '''
        mse = torch.mean((img1 - img2) ** 2)
        psnr = 10 * torch.log10(max_value ** 2 / mse)
        return psnr

    def encode(self, x: torch.Tensor) -> AutoVisionEncoderOutput:
        '''
        Encode an image to latent representation.

        Args:
            x (torch.Tensor): Input image tensor with shape [batch, channels, height, width].

        Returns:
            AutoVisionEncoderOutput: Output containing latent representation and grid_shape.
        '''
        return self._encode(x)

    def decode(self, encoder_output: AutoVisionEncoderOutput) -> AutoVisionDecoderOutput:
        '''
        Decode a latent representation to an image.

        Args:
            encoder_output (AutoVisionEncoderOutput): Output from encode method containing
                                                      latent representation and grid_shape.

        Returns:
            AutoVisionDecoderOutput: Output containing reconstructed image and grid_shape.
        '''
        return self._decode(encoder_output)

    def _encode(self, x: torch.Tensor) -> AutoVisionEncoderOutput:
        raise NotImplementedError('Subclasses must implement _encode method')

    def _decode(self, encoder_output: AutoVisionEncoderOutput) -> AutoVisionDecoderOutput:
        raise NotImplementedError('Subclasses must implement _decode method')


class VisionEmbedding(BasicModel):
    def __init__(
        self, 
        hidden_dim: int, 
        dead_codes: Union[List[int], str],
        codebook_dim: int = 15,
        vision_model: Optional[AutoencoderVisionModel] = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.codebook_dim = codebook_dim
        self.original_vocab_size = 2 ** codebook_dim
        
        self.vision_model = vision_model
        
        if isinstance(dead_codes, str):
            with open(dead_codes, 'r', encoding='utf-8') as f:
                data: dict = json.load(f)
                dead_codes_list = data.get('dead_codes', [])
        else:
            dead_codes_list = dead_codes
            
        dead_codes_set = set(dead_codes_list)
        
        self.effective_vocab_size = self.original_vocab_size - len(dead_codes_set) + 1
        self.embedding = nn.Embedding(self.effective_vocab_size, hidden_dim)
        
        mapping = torch.full((self.original_vocab_size,), self.effective_vocab_size - 1, dtype=torch.long)
        
        active_codes = [i for i in range(self.original_vocab_size) if i not in dead_codes_set]
        for new_idx, old_idx in enumerate(active_codes):
            mapping[old_idx] = new_idx
            
        self.register_buffer('index_mapping', mapping)
    
    def forward(self, original_indices: torch.Tensor) -> torch.Tensor:
        mapped_indices = self.index_mapping[original_indices]
        return self.embedding(mapped_indices)
    
    @torch.no_grad()
    @torch.compiler.disable
    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.vision_model is None:
            raise ValueError
        
        self.vision_model.eval()
        
        enc_out = self.vision_model.encode(image)
        
        batch_size = image.size(0)
        indices = enc_out.indices.view(batch_size, -1)
        
        return self.forward(indices)