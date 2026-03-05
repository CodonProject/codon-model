from codon.base import *


from dataclasses import dataclass
from typing import Tuple


@dataclass
class LookupFreeQuantizationOutput:
    z_q: torch.Tensor
    loss: torch.Tensor
    indices: torch.Tensor
    entropy: torch.Tensor
    perplexity: torch.Tensor


class LookupFreeQuantization(BasicModel):
    '''
    Lookup-Free Quantization (LFQ) module.
    
    Based on the MagViT-2 paper. Directly projects the latent into a low-dimensional space for binarization (Sign),
    and combines binary bits into integer indices.
    '''

    def __init__(
        self,
        latent_dim: int = 256,
        codebook_dim: int = 18,
        entropy_weight: float = 0.1,
        commitment_weight: float = 0.25,
        diversity_gamma: float = 1.0,
    ):
        '''
        Args:
            latent_dim (int): Dimension of input/output features (Encoder output dimension).
            codebook_dim (int): Dimension of quantization space (number of bits). Vocabulary size is 2^codebook_dim.
            entropy_weight (float): Weight for entropy loss, encouraging Codebook utilization.
            commitment_weight (float): Weight for commitment loss, pulling Encoder output closer to quantized values.
            diversity_gamma (float): Scaling factor for entropy penalty.
        '''
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_dim = codebook_dim
        self.entropy_weight = entropy_weight
        self.commitment_weight = commitment_weight
        self.diversity_gamma = diversity_gamma
        
        self.project_in = nn.Linear(latent_dim, codebook_dim) if latent_dim != codebook_dim else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, latent_dim) if latent_dim != codebook_dim else nn.Identity()
        
        self.register_buffer("basis", 2 ** torch.arange(codebook_dim))

    def entropy_loss(self, affine_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Calculates bit-based entropy loss.
        
        Args:
            affine_logits (torch.Tensor): Projected logits [B*H*W, codebook_dim].
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                loss: Entropy loss scalar (maximize entropy -> minimize loss).
                avg_entropy: Average entropy (for monitoring).
        '''
        probs = torch.sigmoid(affine_logits)
        
        # [B*H*W, D] -> [D]
        avg_probs = torch.mean(probs, dim=0)
        
        entropy = - (avg_probs * torch.log(avg_probs + 1e-5) + 
                    (1 - avg_probs) * torch.log(1 - avg_probs + 1e-5))
        
        loss = - torch.mean(entropy) * self.diversity_gamma
        
        return loss, torch.mean(entropy)

    def forward(self, z: torch.Tensor) -> LookupFreeQuantizationOutput:
        '''
        Args:
            z (torch.Tensor): Input tensor with shape [B, C, H, W].
            
        Returns:
            LookupFreeQuantizationOutput: The output containing quantized latent, loss, indices, etc.
        '''
        B, C, H, W = z.shape
        
        z_permuted = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z_permuted.view(-1, C)
        
        z_e = self.project_in(z_flattened)
        
        z_q = torch.sign(z_e)
        
        z_q = z_e + (z_q - z_e).detach()
        
        commitment_loss = torch.mean((z_q.detach() - z_e) ** 2)
        
        entropy_loss, avg_entropy = self.entropy_loss(z_e)
        
        total_loss = self.commitment_weight * commitment_loss + self.entropy_weight * entropy_loss
        
        # [N, codebook_dim] * [codebook_dim] -> sum -> [N]
        is_positive = (z_q > 0).long()
        indices = (is_positive * self.basis).sum(dim=1)
        indices = indices.view(B, H, W)
        
        z_out = self.project_out(z_q)
        z_out = z_out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous() # [B, C, H, W]
        
        perplexity = 2 ** avg_entropy
        
        return LookupFreeQuantizationOutput(
            z_q=z_out,
            loss=total_loss,
            indices=indices,
            entropy=avg_entropy,
            perplexity=perplexity
        )
