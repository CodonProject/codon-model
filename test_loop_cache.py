import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
import torch

from exp_loop import Looped, LoopedConfig, LoopedLM, LoopedVLM
from codon.model.cache import ModelCache, KVLayerCache, GatedDeltaAttentionLayerCache

TOL = 1e-5


def small_cfg(**kw):
    base = dict(model_dim=128, num_heads=4, recurrence=1,
                prelude_attn_types=['mha'], body_attn_types=['gdn', 'mha'],
                coda_attn_types=['mha'])
    base.update(kw)
    return LoopedConfig(**base)


class TestLoopedCache(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.L = 16
        self.x = torch.randn(2, self.L, 128)

    def _causally_fair_decode_diff(self, model, pre_len):
        """Prefill on prefix, then decode one token at a time; each step is compared
        against a causally-fair one-shot forward on the tokens visible so far."""
        worst = 0.0
        with torch.no_grad():
            cache = model.create_cache()
            y_pre = model(self.x[:, :pre_len], past_key_values=cache)
            self.assertLess((y_pre - model(self.x[:, :pre_len])).abs().max().item(), TOL)

            for t in range(pre_len, self.L):
                step = model(self.x[:, t:t + 1], past_key_values=cache, embedding_start=t)
                ref = model(self.x[:, :t + 1])[:, -1:]
                worst = max(worst, (step - ref).abs().max().item())
        return worst

    def test_cache_slots_and_types(self):
        """槽位按 prelude → body → coda 排布，且缓存类型由注意力机制决定。"""
        model = Looped(small_cfg()).eval()
        cache = model.create_cache()
        self.assertIsInstance(cache, ModelCache)
        self.assertEqual(len(cache), 0)

        with torch.no_grad():
            model(self.x, past_key_values=cache)

        self.assertEqual(len(cache), 4)                       # 1 prelude + 2 body + 1 coda
        self.assertEqual(sorted(cache.layer_caches), [0, 1, 2, 3])
        self.assertIsInstance(cache[0], KVLayerCache)         # prelude mha
        self.assertIsInstance(cache[1], GatedDeltaAttentionLayerCache)  # body gdn
        self.assertIsInstance(cache[2], KVLayerCache)         # body mha
        self.assertIsInstance(cache[3], KVLayerCache)         # coda mha
        self.assertEqual(cache.seq_length, self.L)

    def test_prefill_matches_no_cache(self):
        """带缓存的一次性前向必须与不带缓存完全一致。"""
        model = Looped(small_cfg()).eval()
        with torch.no_grad():
            y_plain = model(self.x)
            y_cached = model(self.x, past_key_values=model.create_cache())
        self.assertLess((y_plain - y_cached).abs().max().item(), TOL)

    def test_incremental_decode_matches_full_forward(self):
        """prefill + 逐 token decode 必须复现一次性前向（recurrence=1）。"""
        model = Looped(small_cfg()).eval()
        diff = self._causally_fair_decode_diff(model, self.L // 2)
        self.assertLess(diff, TOL)

    def test_seq_length_grows(self):
        model = Looped(small_cfg()).eval()
        cache = model.create_cache()
        with torch.no_grad():
            model(self.x, past_key_values=cache)
            self.assertEqual(cache.seq_length, self.L)
            model(self.x[:, :1], past_key_values=cache, embedding_start=self.L)
            self.assertEqual(cache.seq_length, self.L + 1)

    def test_reset(self):
        model = Looped(small_cfg()).eval()
        cache = model.create_cache()
        with torch.no_grad():
            model(self.x, past_key_values=cache)
        cache.reset()
        self.assertEqual(cache.seq_length, 0)

    def test_gdn_only_and_mha_only(self):
        """纯 GDN 与纯 MHA 两种极端配置都要能缓存解码。"""
        for attn in ('gdn', 'mha'):
            with self.subTest(attn=attn):
                model = Looped(small_cfg(
                    prelude_attn_types=[attn],
                    body_attn_types=[attn, attn],
                    coda_attn_types=[attn],
                )).eval()
                self.assertLess(self._causally_fair_decode_diff(model, self.L // 2), TOL)

    def test_loop_outputs_with_cache(self):
        model = Looped(small_cfg()).eval()
        with torch.no_grad():
            y, loops = model(self.x, past_key_values=model.create_cache(),
                             return_all_loop_outputs=True)
        self.assertEqual(len(loops), 1)          # recurrence=1
        self.assertEqual(tuple(loops[0].shape), (2, self.L, 128))
        self.assertEqual(tuple(y.shape), (2, self.L, 128))

    def test_recurrence_gt_one_rejected(self):
        """recurrence > 1 时缓存不可能与一次性前向等价，必须显式报错。"""
        model = Looped(small_cfg(recurrence=2)).eval()
        with self.assertRaises(ValueError):
            model.create_cache()
        with self.assertRaises(ValueError):
            with torch.no_grad():
                model(self.x, past_key_values=ModelCache())
        # 不带缓存时 recurrence > 1 仍正常工作
        with torch.no_grad():
            self.assertEqual(tuple(model(self.x).shape), (2, self.L, 128))


class TestLoopedLM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        cfg = LoopedConfig(model_dim=128, num_heads=4, recurrence=1,
                           prelude_attn_types=['mha'],
                           body_attn_types=['mha', 'gdn'],
                           coda_attn_types=['mha'])
        self.model = LoopedLM(cfg, vocab_size=256).eval()
        self.ids = torch.randint(0, 256, (1, 8))

    def test_position_embedding_is_interleaved_fourier_2d(self):
        """默认位置编码必须是 InterleavedFourierRotaryEmbedding(num_axes=2)。"""
        from codon.block.embedding import InterleavedFourierRotaryEmbedding
        emb = self.model.position_emb
        self.assertIsInstance(emb, InterleavedFourierRotaryEmbedding)
        self.assertEqual(emb.num_axes, 2)
        self.assertEqual(emb.model_dim, 128 // 4)          # head_dim
        # 频率表宽度应为 head_dim，且为「半宽重排后复制成两份」
        self.assertEqual(tuple(emb.cos_cached.shape), (self.model.config.max_len, 32))
        half = emb.cos_cached.shape[1] // 2
        self.assertTrue(torch.equal(emb.cos_cached[:, :half], emb.cos_cached[:, half:]))

    def test_rope_fallback_still_available(self):
        from codon.block.embedding import RotaryEmbedding
        cfg = LoopedConfig(model_dim=128, num_heads=4, recurrence=1,
                           prelude_attn_types=['mha'], body_attn_types=['mha'],
                           coda_attn_types=['mha'], pos_emb_type='rope')
        m = LoopedLM(cfg, vocab_size=256).eval()
        self.assertIsInstance(m.position_emb, RotaryEmbedding)
        out = m(input_ids=self.ids)
        self.assertEqual(tuple(out.logits.shape), (1, 8, 256))

    def test_positions_differ_per_token(self):
        """不同 token 位置必须拿到不同的旋转相位（二维位置 [t, t] 生效）。"""
        emb = self.model.position_emb
        pos = self.model._build_positions(torch.zeros(1, 4, 32), 0, 4)
        self.assertEqual(tuple(pos.shape), (1, 4, 2))
        self.assertTrue(torch.equal(pos[..., 0], pos[..., 1]))
        x = torch.randn(1, 1, 4, 32)   # 非零输入，相位差异才可见
        outs = [emb(x, positions=pos[:, t:t + 1]) for t in range(4)]
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertGreater((outs[i] - outs[j]).abs().max().item(), 0.0)

    def test_forward_shapes_and_cache(self):
        cache = self.model.create_cache()
        out = self.model(input_ids=self.ids, start_pos=0, past_key_values=cache)
        self.assertEqual(tuple(out.logits.shape), (1, 8, 256))
        self.assertIs(out.past_key_values, cache)
        self.assertEqual(cache.seq_length, 8)

    def test_loop_hidden_states_exposed(self):
        out = self.model(input_ids=self.ids, return_all_loop_outputs=True)
        self.assertIsNotNone(out.hidden_states)
        self.assertEqual(len(out.hidden_states), 1)
        self.assertEqual(tuple(out.hidden_states[0].shape), (1, 8, 128))

    def test_logits_match_incremental(self):
        with torch.no_grad():
            full = self.model(input_ids=self.ids).logits
            cache = self.model.create_cache()
            self.model(input_ids=self.ids[:, :4], start_pos=0, past_key_values=cache)
            step = self.model(input_ids=self.ids[:, 4:5], start_pos=4, past_key_values=cache)
        self.assertLess((step.logits[:, -1] - full[:, 4]).abs().max().item(), TOL)

    def test_generate(self):
        gen = self.model.generate(self.ids, max_new_tokens=4, temperature=1.0)
        self.assertEqual(tuple(gen.shape), (1, 8 + 4))

    def test_recurrence_gt_one_rejected(self):
        cfg = LoopedConfig(model_dim=128, num_heads=4, recurrence=2,
                           prelude_attn_types=['mha'], body_attn_types=['mha'],
                           coda_attn_types=['mha'])
        model = LoopedLM(cfg, vocab_size=256).eval()
        with self.assertRaises(ValueError):
            model.generate(self.ids, max_new_tokens=2)


class TestLoopedVLM(unittest.TestCase):
    '''LoopedLM + MobileNetV3 的多模态结合。'''

    PATCH = 49          # 224 / 32 -> 7x7
    GRID = 7

    def setUp(self):
        torch.manual_seed(0)
        cfg = LoopedConfig(model_dim=128, num_heads=4, recurrence=1,
                           prelude_attn_types=['mha'],
                           body_attn_types=['mha', 'gdn'],
                           coda_attn_types=['mha'])
        self.vlm = LoopedVLM(cfg, vocab_size=256, backend='small').eval()
        self.seq_len = 3 + self.PATCH + 5
        self.input_ids = torch.randint(4, 256, (1, self.seq_len))
        self.input_ids[0, 3:3 + self.PATCH] = 0
        self.patch_idx = torch.arange(3, 3 + self.PATCH).unsqueeze(0)
        self.image = torch.randn(3, 224, 224)

    def _run(self, images=None, indices=None, **kw):
        with torch.no_grad():
            return self.vlm(
                input_ids=self.input_ids,
                images=images,
                image_patch_indices=indices,
                **kw,
            )

    def test_vision_feature_shape(self):
        with torch.no_grad():
            feats = self.vlm.encode_image(self.image.unsqueeze(0))
        self.assertEqual(tuple(feats.shape), (1, self.PATCH, 128))

    def test_declares_image_capability(self):
        """LoopedVLM 应通过 meta 声明支持图片输入。"""
        self.assertTrue(self.vlm.supports_image)
        self.assertFalse(self.vlm.supports_thinking)
        self.assertFalse(self.vlm.supports_tool)

    def test_forward_with_image(self):
        out = self._run(images=[self.image], indices=self.patch_idx)
        self.assertEqual(tuple(out.logits.shape), (1, self.seq_len, 256))

    def test_patch_positions_are_2d_grid(self):
        """图像 patch 用 (row, col) 二维位置（叠加 start_pos 基准），文本用 (t, t)。"""
        base = 3
        pos = self.vlm._build_vlm_positions(
            1, self.seq_len, self.patch_idx, [(self.GRID, self.GRID)], [self.PATCH],
            base, torch.device('cpu'),
        )
        patch_pos = pos[0, 3:3 + self.PATCH]
        expected = torch.tensor(
            [[base + r, base + c] for r in range(self.GRID) for c in range(self.GRID)]
        )
        self.assertTrue(torch.equal(patch_pos, expected))
        # 行/列都必须落在网格范围内，不能被总 patch 数污染
        self.assertLess(int(patch_pos[:, 0].max()), base + self.GRID)
        self.assertLess(int(patch_pos[:, 1].max()), base + self.GRID)
        # 文本位置仍是 (t, t)，且同样叠加 start_pos 基准
        self.assertTrue(torch.equal(pos[0, 0], torch.tensor([base, base])))
        self.assertTrue(torch.equal(pos[0, self.seq_len - 1],
                                    torch.tensor([base + self.seq_len - 1] * 2)))

    def test_vision_features_reach_the_sequence(self):
        """占位位置的嵌入必须等于投影后的视觉特征。"""
        with torch.no_grad():
            feats = self.vlm.encode_image(self.image.unsqueeze(0)).squeeze(0)
            pos = self.vlm._build_vlm_positions(
                1, self.seq_len, self.patch_idx, [(self.GRID, self.GRID)], [self.PATCH],
                0, torch.device('cpu'),
            )
            x = self.vlm.language.embed(self.input_ids, positions=pos)
            x[0, self.patch_idx[0]] = feats
        self.assertTrue(torch.allclose(x[0, 3:3 + self.PATCH], feats))

    def test_image_affects_logits(self):
        a = self._run(images=[self.image], indices=self.patch_idx)
        b = self._run(images=[torch.randn(3, 224, 224)], indices=self.patch_idx)
        self.assertFalse(torch.allclose(a.logits, b.logits))

    def test_same_image_reproducible(self):
        a = self._run(images=[self.image], indices=self.patch_idx)
        b = self._run(images=[self.image], indices=self.patch_idx)
        self.assertTrue(torch.allclose(a.logits, b.logits, atol=1e-6))

    def test_placeholder_count_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self._run(images=[self.image],
                      indices=torch.arange(3, 3 + self.PATCH - 1).unsqueeze(0))

    def test_out_of_range_indices_rejected(self):
        with self.assertRaises(ValueError):
            with torch.no_grad():
                self.vlm(input_ids=self.input_ids[:, :20], images=[self.image],
                         image_patch_indices=self.patch_idx)

    def test_text_only_still_works(self):
        with torch.no_grad():
            out = self.vlm(input_ids=torch.randint(4, 256, (1, 8)))
        self.assertEqual(tuple(out.logits.shape), (1, 8, 256))

    def test_prefill_decode_matches_one_shot(self):
        """含图的 prefill + decode 必须复现一次性前向。"""
        pre = 3 + self.PATCH
        with torch.no_grad():
            full = self.vlm(input_ids=self.input_ids, images=[self.image],
                            image_patch_indices=self.patch_idx).logits
            cache = self.vlm.create_cache()
            self.vlm(input_ids=self.input_ids[:, :pre], images=[self.image],
                     image_patch_indices=self.patch_idx, past_key_values=cache)
            step = self.vlm(input_ids=self.input_ids[:, pre:pre + 1],
                            start_pos=pre, past_key_values=cache).logits
        self.assertLess((step[:, -1] - full[:, pre]).abs().max().item(), TOL)

    def test_two_images_batched(self):
        """每 batch 一张图，占位符数量各自匹配。"""
        batch_ids = torch.randint(4, 256, (2, self.seq_len))
        batch_ids[:, 3:3 + self.PATCH] = 0
        idx = self.patch_idx.expand(2, -1).contiguous()
        with torch.no_grad():
            out = self.vlm(input_ids=batch_ids,
                           images=[[torch.randn(3, 224, 224)], [torch.randn(3, 224, 224)]],
                           image_patch_indices=idx)
        self.assertEqual(tuple(out.logits.shape), (2, self.seq_len, 256))


if __name__ == '__main__':
    unittest.main(verbosity=2)
