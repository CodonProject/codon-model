from codon import *
from codon.block import (
    MLP, AdaLayerNorm, MultiHeadAttention
)

from typing import Optional, Tuple


class BidirectionalBlock(BasicModel):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        time_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.time_dim  = time_dim
        self.dropout_p = dropout
        
        self.attn_norm = AdaLayerNorm(features_dim=model_dim, embedding_dim=time_dim)
        
        self.attn = MultiHeadAttention(
            hidden_size=model_dim,
            num_heads=num_heads,
            is_causal=False,
            dropout=dropout
        )
        
        self.ffn_norm = AdaLayerNorm(features_dim=model_dim, embedding_dim=time_dim)
        
        self.ffn = MLP.SwiGLU(
            in_features=model_dim,
            dropout=dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        norm_x = self.attn_norm(x, t_emb)
        attn_out = self.attn(norm_x, attention_mask=mask)
        x = x + attn_out.output
        norm_x = self.ffn_norm(x, t_emb)
        x = x + self.ffn(norm_x)
        return x


class DiscreteDiffusionScheduler:
    def __init__(self, num_steps: int = 100, schedule_type: str = 'cosine'):
        self.num_steps = num_steps
        self.schedule_type = schedule_type
        
        if schedule_type == 'linear':
            alphas = torch.linspace(0.0, 0.99, num_steps + 1)
        elif schedule_type == 'cosine':
            steps = torch.arange(num_steps + 1, dtype=torch.float32)
            alphas = 1.0 - torch.cos((steps / num_steps) * (math.pi / 2))
            alphas = alphas * 0.99
        else:
            raise ValueError(f'Unknown schedule type: {schedule_type}')
            
        self.alphas = alphas
    
    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        return self.alphas[t].to(t.device)
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        mask_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = x_0.shape
        alpha_t = self.get_alpha(t).view(B, 1) # [B, 1]
        
        random_probs = torch.rand(x_0.shape, device=x_0.device)
        is_masked = random_probs < alpha_t
        
        x_t = torch.where(is_masked, torch.tensor(mask_id, device=x_0.device), x_0)
        return x_t, is_masked
    
    def step(
        self,
        x_t: torch.Tensor,
        logits_x0: torch.Tensor,
        t: int,
        mask_id: int
    ) -> torch.Tensor:
        alpha_t = self.alphas[t]
        alpha_t_minus = self.alphas[t - 1]
        
        probs_x0 = F.softmax(logits_x0, dim=-1) # [B, L, vocab_size]
        dist = torch.distributions.Categorical(probs=probs_x0)
        pred_x0 = dist.sample() # [B, L]
        
        p_reveal = (alpha_t - alpha_t_minus) / (alpha_t + 1e-8)
        
        reveal_mask = torch.rand(x_t.shape, device=x_t.device) < p_reveal
        should_reveal = (x_t == mask_id) & reveal_mask
        
        x_t_minus = torch.where(should_reveal, pred_x0, x_t)
        return x_t_minus