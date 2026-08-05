import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import unittest
import torch
from codon.block.attention import MultiHeadAttention
from codon.model.cache import (
    build_cache,
    KVLayerCache,
    TensorLayerCache,
    HCALayerCache,
    CSALayerCache
)


class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.seq_len = 64
        self.hidden_size = 512
        self.num_heads = 8
        self.num_kv_heads = 2  # GQA
        self.head_dim = self.hidden_size // self.num_heads
        
        self.x = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)

    def test_mutex_hca_csa_error(self):
        """测试 HCA 和 CSA 的互斥抛错"""
        with self.assertRaises(AssertionError):
            MultiHeadAttention(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                use_hca=True,
                use_csa=True
            )

    def test_standard_gqa(self):
        """测试标准 GQA 前向传播与 Cache 更新"""
        model = MultiHeadAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads
        ).to(self.device)

        cache = build_cache(model)
        self.assertIsInstance(cache, KVLayerCache)

        # 1. Prefill 阶段
        res1 = model(self.x, past_key_value=cache)
        self.assertEqual(res1.output.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(cache.seq_length, self.seq_len)

        # 2. Decode 单 Step 阶段
        x_next = torch.randn(self.batch_size, 1, self.hidden_size, device=self.device)
        res2 = model(x_next, past_key_value=cache)
        self.assertEqual(res2.output.shape, (self.batch_size, 1, self.hidden_size))
        self.assertEqual(cache.seq_length, self.seq_len + 1)

    def test_mla_attention(self):
        """测试 MLA 机制与 TensorLayerCache"""
        model = MultiHeadAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            q_lora_rank=64,
            kv_lora_rank=32,
            rope_dim=16
        ).to(self.device)

        cache = build_cache(model)
        self.assertIsInstance(cache, TensorLayerCache)

        # Prefill
        res = model(self.x, past_key_value=cache)
        self.assertEqual(res.output.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(cache.seq_length, self.seq_len)

    def test_hca_attention(self):
        """测试 HCA (FP4 / FP16) 机制"""
        for fp4_option in [True, False]:
            model = MultiHeadAttention(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                use_hca=True,
                hca_latent_dim=64,
                hca_block_size=16,
                hca_fp4_storage=fp4_option
            ).to(self.device)

            cache = build_cache(model)
            self.assertIsInstance(cache, HCALayerCache)

            res = model(self.x, past_key_value=cache)
            self.assertEqual(res.output.shape, (self.batch_size, self.seq_len, self.hidden_size))
            
            # Block 数量验证 (64 / 16 = 4 blocks)
            expected_blocks = self.seq_len // 16
            self.assertEqual(cache.seq_length, expected_blocks)

    def test_csa_attention(self):
        """测试 CSA 机制与 Top-K 选择"""
        model = MultiHeadAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            use_csa=True,
            csa_compressed_dim=64,
            csa_block_size=4,
            csa_top_k=8
        ).to(self.device)

        cache = build_cache(model)
        self.assertIsInstance(cache, CSALayerCache)

        res = model(self.x, past_key_value=cache)
        self.assertEqual(res.output.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(cache.seq_length, self.seq_len // 4)

    def test_backward_pass(self):
        """测试全模式下的训练梯度反向传播"""
        configs = [
            {"name": "Standard GQA", "kwargs": {"num_kv_heads": 2}},
            {"name": "MLA", "kwargs": {"q_lora_rank": 64, "kv_lora_rank": 32, "rope_dim": 16}},
            {"name": "HCA", "kwargs": {"use_hca": True, "hca_latent_dim": 64, "hca_block_size": 16}},
            {"name": "CSA", "kwargs": {"use_csa": True, "csa_compressed_dim": 64, "csa_block_size": 4, "csa_top_k": 8}}
        ]

        for config in configs:
            with self.subTest(mode=config["name"]):
                model = MultiHeadAttention(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    **config["kwargs"]
                ).to(self.device)
                
                x = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device, requires_grad=True)
                out = model(x).output
                loss = out.sum()
                loss.backward()

                self.assertIsNotNone(x.grad, f"Gradient lost in {config['name']}")
                self.assertFalse(torch.isnan(x.grad).any(), f"NaN gradient detected in {config['name']}")


if __name__ == "__main__":
    unittest.main()