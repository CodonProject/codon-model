from codon import *

from codon.model.cache import ModelCache, build_cache
from codon.model.types.language import CausalLanguageModel, CausalLanguageModelOutput
from codon.block.embedding import InterleavedRotaryEmbedding

from .config import MotifChordConfig
from .vision import MotifChordVisionProjector
from .audio import MotifChordAudioProjector
from .decoder import HybridDecoder


class MotifChord(CausalLanguageModel):

    def __init__(self, config: MotifChordConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.model_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.position_emb = InterleavedRotaryEmbedding(
            model_dim=config.model_dim // config.num_heads,
            num_axes=1,
        )

        self.decoder = nn.ModuleList([
            HybridDecoder(
                config=config,
                layer_type=config.layer_pattern[idx],
                rope_emb=self.position_emb,
                idx=idx,
            )
            for idx in range(config.num_layers)
        ])

        self.norm = nn.RMSNorm(config.model_dim)
        self.proj_out = nn.Linear(config.model_dim, config.vocab_size, bias=False)

        if config.tie_weights:
            self.proj_out.weight = self.token_emb.weight

        self.vision = MotifChordVisionProjector(config)
        self.audio = MotifChordAudioProjector(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor = None,
        mel: torch.Tensor = None,
        mask: torch.Tensor = None,
        start_pos: Union[int, torch.Tensor] = 0,
        past_key_values: Optional[ModelCache] = None,
        output_attentions: bool = False,
    ) -> CausalLanguageModelOutput:
        text_emb = self.dropout(self.token_emb(input_ids))

        is_prefill = past_key_values is None or len(past_key_values) == 0

        modal_len = 0
        modal_embs = []
        if is_prefill:
            if pixel_values is not None:
                vis_emb = self.vision(pixel_values).to(text_emb.dtype)
                modal_embs.append(vis_emb)
                modal_len += vis_emb.shape[1]
            if mel is not None:
                aud_emb = self.audio(mel).to(text_emb.dtype)
                modal_embs.append(aud_emb)
                modal_len += aud_emb.shape[1]

        modal_embs.append(text_emb)
        x = torch.cat(modal_embs, dim=1) if len(modal_embs) > 1 else text_emb

        if isinstance(start_pos, torch.Tensor):
            effective_start = start_pos + modal_len
        else:
            effective_start = start_pos + modal_len

        all_attentions = [] if output_attentions else None

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
                embedding_start=effective_start,
                past_key_value=layer_past,
            )
            x = out.output

            if output_attentions:
                all_attentions.append(out.attention_weights)

        x = self.norm(x)
        logits = self.proj_out(x)

        return CausalLanguageModelOutput(
            logits=logits,
            past_key_values=past_key_values,
            aux_loss=None,
            attentions=all_attentions,
        )