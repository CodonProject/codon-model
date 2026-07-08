from codon import *
from codon.block.transformer import TransformerDenseDecoder
from codon.block.embedding   import RotaryEmbedding
from codon.utils.tokens      import PackedTokenizer
from codon.utils.onnx        import patch_rms_norm

from .base import CausalLanguageModel, CausalLanguageModelOutput


class MotifA1Tokenizer(PackedTokenizer):
    
    __remote_resource__ = {
        'repo': 'CodonProject/MotifA1-SFT',
        'files': ['motif.vocab'],
        'repo_type': 'model'
    }


class MotifA1(CausalLanguageModel):

    __remote_resource__ = {
        'repo': 'CodonProject/MotifA1-SFT',
        'files': ['MotifA1_SFT.safetensors'],
        'repo_type': 'model'
    }

    def __init__(
        self,
        vocab_size: int = 2**13,
        model_dim: int = 768,
        num_layers: int = 16,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        dropout: float = 0.1,
        tie_weights: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.token_emb = nn.Embedding(vocab_size, model_dim)
        self.position_emb = RotaryEmbedding(model_dim // num_heads)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.ModuleList([
            TransformerDenseDecoder(
                model_dim=model_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                use_qk_norm=True,
                use_attn_gate=False,
                use_swiglu=True,
                dropout=dropout,
                attn_bias=False,
                attn_type='mha',
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
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> CausalLanguageModelOutput:
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

        return CausalLanguageModelOutput(
            logits=logits,
            past_key_values=new_kv_cache,
            aux_loss=aux_loss,
            attentions=all_attentions
        )
