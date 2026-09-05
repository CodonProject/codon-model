# 注意力机制包：从单体 codon/block/attention.py 机制级拆分而来。
# 本轮只拆出纯 MHA（mha.py）；MLA/HCA/CSA/KEV 暂留在 _legacy.py，
# 保持 from codon.block.attention import X 的对外兼容。
from codon.block.attention.mha import MultiHeadAttention
from codon.block.attention.base import BasicAttention, BasicLinearAttention
from codon.block.attention._legacy import (
    MultiHeadAttentionLegacy,
    MultiHeadAttentionKEV
)
from codon.ops import (
    AttentionOutput,
    apply_attention
)

__all__ = [
    'BasicAttention',
    'BasicLinearAttention',
    'MultiHeadAttention',
    'MultiHeadAttentionLegacy',
    'MultiHeadAttentionKEV',
    'AttentionOutput',
    'apply_attention'
]
