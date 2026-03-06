from codon.base import *

from codon.block.transformer import TransformerMoEDecoder
from codon.block.embedding   import RotaryEmbedding

from typing import Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class MotifA1Output:
    logits: torch.Tensor
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    aux_loss: Optional[torch.Tensor] = None
    attentions: Optional[List[torch.Tensor]] = None


class MotifA1(BasicModel):
    def __init__(
        self,
        vocab_size: int = 32000,
        model_dim: int = 768,
        num_layers: int = 8,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        dropout: float = 0.1,
        tie_weights: bool = False
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, model_dim)
        self.position_emb = RotaryEmbedding(model_dim)
        self.dropout = nn.Dropout(dropout)

        self.decoder = nn.ModuleList([
            TransformerMoEDecoder(
                model_dim=model_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                top_k=1,
                num_experts=8,
                num_shared_experts=1,
                use_aux_loss=False,
                use_expert_gate=True,
                use_qk_norm=True,
                use_attn_gate=True,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.RMSNorm(model_dim)
        self.proj_out = nn.Linear(model_dim, vocab_size, bias=False)

        if tie_weights:
            self.proj_out.weight = self.token_emb.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor = None,
        start_pos: int = 0,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> MotifA1Output:
        x = self.token_emb(input_ids)
        x = self.dropout(x)

        new_kv_cache = [] if use_cache else None
        all_attentions = [] if output_attentions else None
        aux_loss = None

        for i, layer in enumerate(self.decoder):
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            out = layer(
                hidden_states=x,
                attention_mask=mask,
                output_attentions=output_attentions,
                position_emb=self.position_emb,
                embedding_start=start_pos,
                past_key_value=layer_past,
                use_cache=use_cache
            )
            
            x = out.output
            
            if use_cache:
                new_kv_cache.append(out.past_key_value)
            
            if output_attentions:
                all_attentions.append(out.attention_weights)
            
            if out.aux_loss is not None:
                if aux_loss is None:
                    aux_loss = out.aux_loss
                else:
                    aux_loss += out.aux_loss

        x = self.norm(x)
        logits = self.proj_out(x)

        return MotifA1Output(
            logits=logits,
            past_key_values=new_kv_cache,
            aux_loss=aux_loss,
            attentions=all_attentions
        )
