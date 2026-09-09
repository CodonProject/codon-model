import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import unittest
import torch

from codon.block.embedding import (
    InterleavedFourierRotaryEmbedding,
    InterleavedRotaryEmbedding,
)


class TestInterleavedFourierRotaryEmbedding(unittest.TestCase):
    '''
    回归测试：频率表只有 half = model_dim // 2 列，而父类 interleave_idx 是按
    model_dim 生成的，直接复用会 IndexError（index out of bounds）。
    '''

    def test_constructs_for_various_dims_and_axes(self):
        for dim, axes in ((32, 2), (64, 2), (48, 3), (32, 1), (128, 4)):
            with self.subTest(dim=dim, axes=axes):
                emb = InterleavedFourierRotaryEmbedding(
                    model_dim=dim, max_len=256, num_axes=axes
                )
                self.assertEqual(tuple(emb.cos_cached.shape), (256, dim))
                self.assertEqual(tuple(emb.sin_cached.shape), (256, dim))

    def test_forward_shape_and_multi_axis_positions(self):
        dim, axes, batch, heads, seq = 32, 2, 2, 4, 5
        emb = InterleavedFourierRotaryEmbedding(model_dim=dim, max_len=256, num_axes=axes)
        x = torch.randn(batch, heads, seq, dim)
        pos = torch.arange(seq).view(1, seq, 1).expand(batch, seq, axes)
        out = emb(x, positions=pos)
        self.assertEqual(tuple(out.shape), tuple(x.shape))

    def test_positions_are_required_when_multiple_axes(self):
        emb = InterleavedFourierRotaryEmbedding(model_dim=32, max_len=256, num_axes=2)
        with self.assertRaises(ValueError):
            emb(torch.randn(1, 1, 4, 32))

    def test_single_axis_matches_parent_geometry(self):
        '''num_axes=1 时，半宽重排索引应退化为恒等，与普通交错 RoPE 的相位一致。'''
        dim, seq = 32, 7
        fourier = InterleavedFourierRotaryEmbedding(
            model_dim=dim, max_len=256, num_axes=1, sigma=0.0
        )
        plain = InterleavedRotaryEmbedding(model_dim=dim, max_len=256, num_axes=1)
        # sigma=0 时对角系数为 1、非对角为 0，等价于纯 RoPE 表
        self.assertLess(
            (fourier.cos_cached - plain.cos_cached).abs().max().item(), 1e-6
        )
        self.assertLess(
            (fourier.sin_cached - plain.sin_cached).abs().max().item(), 1e-6
        )

    def test_different_positions_give_different_phase(self):
        torch.manual_seed(0)
        emb = InterleavedFourierRotaryEmbedding(model_dim=32, max_len=256, num_axes=2)
        x = torch.randn(1, 1, 1, 32)      # 非零输入：相位差异才会体现在输出上
        outs = [
            emb(x, positions=torch.tensor([[[t, t]]]))
            for t in range(4)
        ]
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertGreater((outs[i] - outs[j]).abs().max().item(), 0.0)

    def test_start_pos_offsets_positions(self):
        emb = InterleavedFourierRotaryEmbedding(model_dim=32, max_len=256, num_axes=2)
        x = torch.randn(1, 1, 3, 32)
        base = torch.tensor([[[0, 0], [1, 1], [2, 2]]])
        shifted = torch.tensor([[[3, 3], [4, 4], [5, 5]]])
        a = emb(x, positions=base, start_pos=3)
        b = emb(x, positions=shifted)
        self.assertLess((a - b).abs().max().item(), 1e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
