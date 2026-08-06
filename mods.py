from codon import *
from codon.block import (
    MLP, MultiHeadAttention, AttentionOutput,
    BasicEmbedding, TransformerDecoderOutput,
    RotaryEmbedding
)
from codon.model.types.language import CausalLanguageModelOutput, CausalLanguageModel
from codon.exp.block.attn_film import MultiHeadAttentionFiLM


class Decoder(BasicModel):
    def __init__(
        self,
        attn_type: str = 'mha'
    ):
        super().__init__()

        self.model_dim = 768
        self.attn_type = attn_type
        self.idx = self.safecode()

        self.attn_norm = nn.RMSNorm(self.model_dim)

        self.attn = MultiHeadAttention(
            self.model_dim,
            num_heads=12
        ) if attn_type == 'mha' else MultiHeadAttentionFiLM(
            self.model_dim,
            num_heads=12
        )

        self.ffn = MLP.SwiGLU(self.model_dim)
        self.fn_norm = nn.RMSNorm(self.model_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        output_attentions: bool = False,
        position_emb: BasicEmbedding = None,
        embedding_start: Union[int, torch.Tensor] = 0,
        embedding_pos: torch.Tensor = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] = None,
        use_cache: bool = False
    ) -> TransformerDecoderOutput:
        x = self.attn_norm(hidden_states)

        attention_output: AttentionOutput = self.attn(
            hidden_states=x,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            position_emb=position_emb,
            embedding_start=embedding_start,
            embedding_pos=embedding_pos,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        hidden_states = hidden_states + attention_output.output

        x = self.fn_norm(hidden_states)

        hidden_states = hidden_states + self.ffn(x)

        return TransformerDecoderOutput(
            idx=self.idx,
            output=hidden_states,
            attention_weights=attention_output.attention_weights,
            attention_mask=attention_mask,
            aux_loss=None,
            past_key_value=attention_output.past_key_value,
            use_emb=position_emb,
            emb_start=embedding_start,
            emb_pos=embedding_pos
        )


class Model(CausalLanguageModel):
    def __init__(self, attn_type: str = 'mha'):
        super().__init__()

        self.attn_type = attn_type
        self.token_emb = nn.Embedding(2**13, 768)
        self.position_emb = RotaryEmbedding(768 // 12)
        self.dropout = nn.Dropout(0.1)

        self.decoders = nn.ModuleList([
            Decoder(self.attn_type) for _ in range(4)
        ])

        self.norm = nn.RMSNorm(768)
        self.proj_out = nn.Linear(768, 2**13, bias=False)

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
        past_key_values = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> CausalLanguageModelOutput:
        x = self.token_emb(input_ids)
        x = self.dropout(x)

        new_kv_cache = [] if use_cache else None
        all_attentions = [] if output_attentions else None
        aux_loss = None

        for i, layer in enumerate(self.decoders):
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            out: TransformerDecoderOutput = layer(
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

