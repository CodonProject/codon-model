import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Any, Iterator, Union, Optional, List, Tuple
from dataclasses import dataclass

from codon.base import BasicModel


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


class CausalLanguageModel(BasicModel):
    '''
    Base class for causal language models with text generation capabilities.

    Attributes:
        gradient_checkpointing (bool): Whether gradient checkpointing is enabled.
    '''

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        eos_token_id: int = None
    ) -> torch.Tensor:
        '''
        Generate text tokens autoregressively.

        Args:
            input_ids (torch.Tensor): Input token IDs with shape [batch, seq_len].
            max_new_tokens (int): Maximum number of new tokens to generate. Defaults to 100.
            temperature (float): Sampling temperature. Higher values increase randomness.
                                 Defaults to 1.0.
            top_k (int, optional): If set, sample only from top k tokens. Defaults to None.
            eos_token_id (int, optional): End-of-sequence token ID. If None, generation
                                          stops after max_new_tokens. Defaults to None.

        Returns:
            torch.Tensor: Generated token IDs with shape [batch, seq_len + num_generated].
        '''
        self.eval()
        with torch.no_grad():
            batch_size, seq_len = input_ids.shape
            generated = input_ids.clone()

            past_key_values = None
            for _ in range(max_new_tokens):
                if seq_len > 1:
                    outputs = self.forward(
                        input_ids=generated,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    past_key_values = outputs.past_key_values
                    logits = outputs.logits[:, -1, :]
                else:
                    outputs = self.forward(input_ids=generated)
                    logits = outputs.logits[:, -1, :]

                logits = logits / temperature

                if top_k is not None:
                    top_k_vals = torch.topk(logits, top_k).values[:, -1]
                    logits = torch.where(logits < top_k_vals.unsqueeze(1), torch.full_like(logits, float('-inf')), logits)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)

                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

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
        '''
        Internal encoding method to be implemented by subclasses.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            AutoVisionEncoderOutput: Output containing latent representation and grid_shape.
        '''
        raise NotImplementedError('Subclasses must implement _encode method')

    def _decode(self, encoder_output: AutoVisionEncoderOutput) -> AutoVisionDecoderOutput:
        '''
        Internal decoding method to be implemented by subclasses.

        Args:
            encoder_output (AutoVisionEncoderOutput): Output from encode method.

        Returns:
            AutoVisionDecoderOutput: Output containing reconstructed image.
        '''
        raise NotImplementedError('Subclasses must implement _decode method')
