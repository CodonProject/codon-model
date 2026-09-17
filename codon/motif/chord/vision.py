from codon import *

from codon.impl.dinov3_vit import DINOv3ViT_Base
from codon.block.mlp import MLP

from .config import MotifChordConfig


class MotifChordVisionProjector(BasicModel):
    def __init__(self, config: MotifChordConfig):
        super().__init__()
        self.config = config

        self.encoder = DINOv3ViT_Base.from_remote().freeze()
        self._encoder_dtype = next(self.encoder.parameters()).dtype

        self.proj = MLP(
            in_features=config.vision_dim,
            hidden_features=config.model_dim,
            out_features=config.model_dim,
            bias=False,
            dropout=config.dropout,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self._encoder_dtype)

        with torch.no_grad():
            feats = self.encoder.forward_features(pixel_values)
        patch_tokens = feats['x_norm_patchtokens'].to(next(self.proj.parameters()).dtype)
        return self.proj(patch_tokens)