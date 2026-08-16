from codon import *
from codon.block.transformer import TransformerMoEDecoder
from codon.block.embedding   import RotaryEmbedding
from codon.utils.tokens      import PackedTokenizer
from codon.model.cache       import ModelCache, build_cache
from codon.model.types.language import CausalLanguageModel, CausalLanguageModelOutput


class MotifA2(CausalLanguageModel):

    def __init__(
        self,
        vocab_size: int = 2**14,
        model_dim: int = 768,
        num_layers: int = 16,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        dropout: float = 0.1,
        tie_weights: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.token_emb = nn.Embedding(vocab_size, model_dim)
        self.position_emb = RotaryEmbedding(model_dim // num_heads)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.ModuleList([
            TransformerMoEDecoder(
                model_dim=model_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                top_k=1,
                use_qk_norm=True,
                num_shared_experts=1,
                num_experts=2,
                use_aux_loss=True,
                idx=str(idx)
            )
            for idx in range(num_layers)
        ])

        self.norm = nn.RMSNorm(model_dim)
        self.proj_out = nn.Linear(model_dim, vocab_size, bias=False)

        if tie_weights:
            self.proj_out.weight = self.token_emb.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                torch.nn.init.zeros_(module.weight[module.padding_idx])
    
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor = None,
        start_pos: Union[int, torch.Tensor] = 0,
        past_key_values: Optional[ModelCache] = None,
        output_attentions: bool = False
    ) -> CausalLanguageModelOutput:
        x = self.token_emb(input_ids)
        x = self.dropout(x)

        all_attentions = [] if output_attentions else None
        aux_loss = None

        for i, layer in enumerate(self.decoder):
            layer_past = None
            if isinstance(past_key_values, ModelCache):
                if past_key_values[i] is None:
                    past_key_values[i] = build_cache(layer.attn)
                layer_past = past_key_values[i]
            
            out = layer(
                hidden_states=x,
                attention_mask=mask,
                output_attentions=output_attentions,
                position_emb=self.position_emb,
                embedding_start=start_pos,
                past_key_value=layer_past,
            )
            
            x = out.output
            
            if output_attentions:
                all_attentions.append(out.attention_weights)
            
            if out.aux_loss is not None:
                if aux_loss is None:
                    aux_loss = out.aux_loss
                else:
                    aux_loss += out.aux_loss

        x = self.norm(x)
        logits = self.proj_out(x)

        return CausalLanguageModelOutput(
            logits=logits,
            past_key_values=past_key_values,
            aux_loss=aux_loss,
            attentions=all_attentions
        )