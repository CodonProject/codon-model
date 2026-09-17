from codon import *

from codon.impl.whisper_tiny import WhisperTinyAudioEncoder
from codon.block.mlp import MLP

from .config import MotifChordConfig


class MotifChordAudioProjector(BasicModel):

    def __init__(self, config: MotifChordConfig):
        super().__init__()

        self.encoder = WhisperTinyAudioEncoder(
            pool_stride=config.audio_pool_stride,
        ).from_remote().freeze()

        self.proj = MLP(
            in_features=config.audio_dim,
            hidden_features=config.model_dim,
            out_features=config.model_dim,
            bias=False,
            dropout=config.dropout,
        )

    def forward(
        self,
        mel: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            feats, _ = self.encoder(mel, attention_mask=attention_mask)
        return self.proj(feats)