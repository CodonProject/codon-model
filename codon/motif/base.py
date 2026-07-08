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


@dataclass
class CausalLanguageModelOutput:
    '''
    Output of causal language model.

    Attributes:
        logits (torch.Tensor): Prediction logits.
        past_key_values (list, optional): List of past key value states.
        aux_loss (torch.Tensor, optional): Auxiliary loss.
        attentions (list, optional): List of attention weights.
        hidden_states (tuple, optional): Tuple of hidden states.
    '''
    logits: torch.Tensor
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    aux_loss: Optional[torch.Tensor] = None
    attentions: Optional[List[torch.Tensor]] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None


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


class Sampler:
    def __init__(
        self,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.15
    ) -> None:
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    @torch.no_grad()
    def __call__(self, logits: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        Args:
            logits (torch.Tensor): [batch_size, vocab_size]
            input_ids (torch.Tensor, optional): 历史已生成的 token ids [batch_size, seq_len]
        '''
        # 0. Repetition Penalty
        if self.repetition_penalty != 1.0 and input_ids is not None:
            for i in range(logits.shape[0]):
                unique_tokens = torch.unique(input_ids[i])
                for token_id in unique_tokens:
                    val = logits[i, token_id]
                    if val > 0:
                        logits[i, token_id] = val / self.repetition_penalty
                    else:
                        logits[i, token_id] = val * self.repetition_penalty

        # 1. Temperature
        if self.temperature != 1.0:
            temp = max(self.temperature, 1e-5)
            logits = logits / temp

        # 2. Top-K
        if self.top_k is not None and self.top_k > 0:
            top_k = min(self.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # 3. Top-P
        if self.top_p is not None and 0.0 < self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token


class CausalLanguageModel(BasicModel):
    '''
    Base class for causal language models with optimized autoregressive generation capabilities.
    '''

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        sampler: Optional[Sampler] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True
    ) -> torch.Tensor:
        '''
        Generate text tokens autoregressively using a prefill-decode pipeline.

        Args:
            input_ids (torch.Tensor): Input prompt token IDs with shape [batch, seq_len].
            max_new_tokens (int): Maximum number of new tokens to generate.
            sampler (Sampler, optional): Instance of Sampler. If None, default Sampler(0.7) is used.
            eos_token_id (int, optional): End-of-sequence token ID.
            use_cache (bool): Whether to leverage KV caching for decode steps.

        Returns:
            torch.Tensor: Generated token IDs with shape [batch, seq_len + num_generated].
        '''
        self.eval()
        if sampler is None:
            sampler = Sampler(temperature=0.7)

        generated = input_ids.clone()
        
        kv_cache = KVCache() if use_cache else None

        with torch.no_grad():
            # 1. Prefill 
            outputs = self.forward(
                input_ids=input_ids,
                start_pos=0,
                past_key_values=None,
                use_cache=use_cache
            )
            
            logits = outputs.logits[:, -1, :]
            next_token = sampler(logits)
            generated = torch.cat([generated, next_token], dim=-1)

            if use_cache and outputs.past_key_values is not None:
                kv_cache.update(outputs.past_key_values)

            # 2. Decode
            for _ in range(max_new_tokens - 1):
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

                if use_cache and kv_cache is not None:
                    current_pos = kv_cache.current_len
                    outputs = self.forward(
                        input_ids=next_token,
                        start_pos=current_pos,
                        past_key_values=kv_cache.states,
                        use_cache=True
                    )
                    kv_cache.update(outputs.past_key_values)
                else:
                    outputs = self.forward(
                        input_ids=generated,
                        start_pos=0,
                        past_key_values=None,
                        use_cache=False
                    )

                logits = outputs.logits[:, -1, :]
                next_token = sampler(logits, input_ids=generated)
                generated = torch.cat([generated, next_token], dim=-1)

            return generated

    def compute_perplexity(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        '''
        Compute perplexity from logits and target tokens.

        Args:
            logits (torch.Tensor): Model output logits with shape [batch, seq_len, vocab_size].
            targets (torch.Tensor): Target token IDs with shape [batch, seq_len].

        Returns:
            torch.Tensor: Perplexity value (lower is better).
        '''
        batch_size, seq_len, vocab_size = logits.shape

        logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
        targets_flat = targets.reshape(batch_size * seq_len)

        loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')
        perplexity = torch.exp(loss)

        return perplexity


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