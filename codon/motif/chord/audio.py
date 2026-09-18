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
        self._encoder_dtype = next(self.encoder.parameters()).dtype

        self.proj = MLP(
            in_features=config.audio_dim,
            hidden_features=config.model_dim,
            out_features=config.model_dim,
            bias=False,
            dropout=config.dropout,
        )

    @torch.compiler.disable
    def forward(
        self,
        mel: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 冻结塔（+ 投影）整段 eager 执行：torch.compile 时把重塔留在图外，
        # 与 codon/motif/base.py 的 VisionEmbedding.embed_image 同一套做法。
        # 冻结塔在 no_grad 下运行、不受 autocast 保护：输入必须先对齐塔的 dtype，
        # 否则半精度训练会报 Input type (float) and weight type (bf16) 之类的错。
        with torch.no_grad():
            feats, _ = self.encoder(
                mel.to(self._encoder_dtype), attention_mask=attention_mask
            )
        return self.proj(feats.to(next(self.proj.parameters()).dtype))