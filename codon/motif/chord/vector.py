from codon import *

from codon.block.mlp import MLP

from .config import MotifChordConfig


class MotifChordVectorProjector(BasicModel):
    '''
    连续向量模态（动作 / 本体感觉 / 触觉）→ model_dim 的逐行投影塔。

    与视觉 / 音频塔的差别：这里没有可冻结的编码器，调用方给的张量本身就是特征
    （`[T, in_features]`，或 `[in_features]` 表示单步）。逐行线性投影之后写到
    codon.j2 的占位符位置，因此**输出行数与输入行数一一对应** —— Session 侧
    `_expand_vector_blocks` 就是按同一个行数展开 `<|action_pad|>` 的。

    形状约定：`MotifChord._inject_modality` 会对每个输入做 `item.unsqueeze(0)`，
    再 `squeeze(0)` 取回 `[N, model_dim]`，所以本塔必须返回 `[1, N, model_dim]`
    （与视觉塔返回 `[1, num_patches, dim]` 一致）。
    '''

    def __init__(self, config: MotifChordConfig, in_features: int):
        super().__init__()
        self.config = config
        self.in_features = int(in_features)

        self.proj = MLP(
            in_features=self.in_features,
            hidden_features=int(getattr(config, 'vector_hidden_dim', 0) or config.model_dim),
            out_features=config.model_dim,
            bias=False,
            dropout=config.dropout,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.float()
        if values.dim() == 1:                      # [D] -> [1, D]
            values = values.unsqueeze(0)
        if values.dim() == 2:                      # [T, D] -> [1, T, D]
            values = values.unsqueeze(0)
        return self.proj(values)
