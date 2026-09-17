from codon import *

from codon.ops import AttentionOutput
from codon.block.attention import GatedDeltaAttention, MultiHeadAttention
from codon.block.mlp import MLP
from codon.block.embedding import InterleavedRotaryEmbedding
from codon.block.norm import RMSNorm

from codon.model.cache import (
    BasicLayerCache,
    GatedDeltaAttentionLayerCache,
    KVLayerCache
)
from .config import MotifChordConfig


@dataclass
class DecoderOutput:
    output: torch.Tensor
    past_key_value: Optional[BasicLayerCache] = None
    attention_weights: Optional[torch.Tensor] = None
    payload: Optional[dict] = None


class GDNDecoder(BasicModel):
    def __init__(
        self,
        config: MotifChordConfig
    ):
        super().__init__()

        self.model_dim = config.model_dim

        self.attn_norm = RMSNorm(config.model_dim)
        self.attn = GatedDeltaAttention(
            hidden_size=config.model_dim,
            num_heads=config.num_heads,
            head_k_dim=config.gdn_head_k_dim,
            bias=False,
            dropout=config.dropout
        )

        self.ffn_norm = RMSNorm(config.model_dim)
        self.ffn = MLP.SwiGLU(
            in_features=config.model_dim,
            dropout=config.dropout
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        output_attentions: bool = False,
        position_emb: InterleavedRotaryEmbedding = None,
        embedding_start: Union[int, torch.Tensor] = 0,
        embedding_pos: torch.Tensor = None,
        past_key_value: Optional[GatedDeltaAttentionLayerCache] = None
    ) -> DecoderOutput:
        residual = hidden_states
        attn_out: AttentionOutput = self.attn(
            hidden_states=self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            position_emb=position_emb,
            embedding_start=embedding_start,
            embedding_pos=embedding_pos,
            past_key_value=past_key_value
        )
        hidden_states = residual + self.dropout(attn_out.output)

        residual = hidden_states
        hidden_states = residual + self.dropout(self.ffn(self.ffn_norm(hidden_states)))

        return DecoderOutput(
            output=hidden_states,
            past_key_value=attn_out.past_key_value,
            attention_weights=attn_out.attention_weights,
            payload=attn_out.payload
        )


class GQADecoder(BasicModel):
    def __init__(
        self,
        config: MotifChordConfig,
        rope_emb: InterleavedRotaryEmbedding = None
    ):
        super().__init__()

        self.model_dim = config.model_dim
        self.rope_emb = rope_emb

        self.attn_norm = RMSNorm(config.model_dim)
        self.attn = MultiHeadAttention(
            hidden_size=config.model_dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            use_qk_norm=True,
            use_gate=True,
            bias=False,
            dropout=config.dropout,
            is_causal=True
        )

        self.ffn_norm = RMSNorm(config.model_dim)
        self.ffn = MLP.SwiGLU(
            in_features=config.model_dim,
            dropout=config.dropout
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        output_attentions: bool = False,
        position_emb: InterleavedRotaryEmbedding = None,
        embedding_start: Union[int, torch.Tensor] = 0,
        embedding_pos: torch.Tensor = None,
        past_key_value: Optional[KVLayerCache] = None
    ) -> DecoderOutput:
        current_rope = position_emb if position_emb is not None else self.rope_emb

        residual = hidden_states
        attn_out: AttentionOutput = self.attn(
            hidden_states=self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            position_emb=current_rope,
            embedding_start=embedding_start,
            embedding_pos=embedding_pos,
            past_key_value=past_key_value
        )
        hidden_states = residual + self.dropout(attn_out.output)

        residual = hidden_states
        hidden_states = residual + self.dropout(self.ffn(self.ffn_norm(hidden_states)))

        return DecoderOutput(
            output=hidden_states,
            past_key_value=attn_out.past_key_value,
            attention_weights=attn_out.attention_weights,
            payload=attn_out.payload
        )


class HybridDecoder(BasicModel):
    def __init__(
        self,
        config: MotifChordConfig,
        layer_type: str,
        rope_emb: InterleavedRotaryEmbedding = None,
        idx: Union[int, str] = None,
    ):
        super().__init__()

        if layer_type not in ('gdn', 'gqa'):
            raise ValueError(f"layer_type must be 'gdn' or 'gqa', got {layer_type!r}")

        self.layer_type = layer_type
        self.idx = str(idx) if idx is not None else safecode()

        if layer_type == 'gdn':
            self.inner = GDNDecoder(config)
        else:
            self.inner = GQADecoder(config, rope_emb=rope_emb)

        self.attn = self.inner.attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        output_attentions: bool = False,
        position_emb: InterleavedRotaryEmbedding = None,
        embedding_start: Union[int, torch.Tensor] = 0,
        embedding_pos: torch.Tensor = None,
        past_key_value: Optional[BasicLayerCache] = None,
    ) -> DecoderOutput:
        return self.inner(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            position_emb=position_emb,
            embedding_start=embedding_start,
            embedding_pos=embedding_pos,
            past_key_value=past_key_value,
        )