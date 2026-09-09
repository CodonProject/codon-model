'''
LoopedVLM 接入 pretrain pipeline 的回归测试（CPU，小配置）。

覆盖：
  1. 默认 batch_builder：纯文本 (stage, inputs, labels) 仍走 {'input_ids': ...}
  2. 自定义 batch_builder：多模态 kwargs 能传到模型
  3. 多模态数据流 + 训练循环能跑通并更新参数
  4. image_patch_id 能从 input_ids 自扫出占位符位置
'''
import os, sys, shutil

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import unittest
import torch

from exp_loop import LoopedConfig, LoopedVLM
from codon.pipeline.pretrain import PretrainConfig, PretrainPipeline
from codon.utils.plan import TrainingPlan, Stage
from demo_vlm_pretrain import ToyVLMStream, VLMPretrainPipeline
from demo_image_embed import build_tokenizer

PATCH_ID = 1
PATCHES = 49          # 224 / 32 -> 7x7
VOCAB = 256


def tiny_vlm(seq_ok=True):
    cfg = LoopedConfig(model_dim=64, num_heads=2, recurrence=1,
                       prelude_attn_types=['mha'], body_attn_types=['mha'],
                       coda_attn_types=['mha'],
                       pos_emb_type='fourier_interleaved', pos_num_axes=2)
    return LoopedVLM(cfg, vocab_size=VOCAB, backend='small')


class TestPretrainBatchBuilder(unittest.TestCase):
    def test_default_builder_text_only(self):
        out, labels = PretrainPipeline._default_batch_builder(
            (Stage('F', 8, 0, 2, 0, 1), torch.zeros(2, 8, dtype=torch.long),
             torch.ones(2, 8, dtype=torch.long))
        )
        self.assertEqual(list(out), ['input_ids'])
        self.assertEqual(tuple(out['input_ids'].shape), (2, 8))
        self.assertEqual(tuple(labels.shape), (2, 8))

    def test_to_device_keeps_nesting(self):
        nested = [[torch.zeros(1)] for _ in range(3)]
        moved = PretrainPipeline._to_device(nested, 'cpu')
        self.assertIsInstance(moved, list)
        self.assertIsInstance(moved[0], list)
        self.assertEqual(len(moved), 3)
        flat = PretrainPipeline._to_device([torch.zeros(1)], 'cpu')
        self.assertIsInstance(flat[0], torch.Tensor)

    def test_config_accepts_batch_builder(self):
        cfg = PretrainConfig(batch_builder=lambda b: ({'input_ids': b[1]}, b[2]))
        self.assertTrue(callable(cfg.batch_builder))
        self.assertIsNone(PretrainConfig().batch_builder)


class TestVLMPretrain(unittest.TestCase):
    CKPT = './_tmp_test_vlm_ckpt'

    def setUp(self):
        torch.manual_seed(0)
        # pretrain 会自动续训 last.pt；测试间共享目录会导致 epoch 已满而跳过训练
        shutil.rmtree(self.CKPT, ignore_errors=True)
        self.model = tiny_vlm()
        self.seq_len = 16 + PATCHES + 8
        plan = TrainingPlan(total_tokens=0, total_steps=2, step_mode='min', stages=[
            Stage('F', self.seq_len, 0, 1, 0, 2)])
        self.plan = plan
        self.stream = ToyVLMStream(plan, vocab_size=VOCAB, patch_id=PATCH_ID,
                                   patches_per_image=PATCHES)
        self.stream.nested_images = True

    def _pipeline(self, batch_builder):
        cfg = PretrainConfig(
            compiled=False, base_context=self.seq_len, target_context=self.seq_len,
            global_batch_tokens=self.seq_len, step_mode='min',
            use_progress=False, save_every_steps=0, sanity_every_steps=0,
            ckpt_dir=self.CKPT, batch_builder=batch_builder,
        )
        return VLMPretrainPipeline(self.model, build_tokenizer(), cfg,
                                   self.stream, device=torch.device('cpu'))

    def test_training_step_updates_language_params(self):
        before = [p.detach().clone() for p in self.model.language.parameters()]
        pipe = self._pipeline(lambda b: (
            {'input_ids': b[1], 'images': b[3], 'image_patch_id': PATCH_ID}, b[2]))
        metrics = pipe.train(self.stream, num_epochs=1, steps_per_epoch=2,
                             batch_fn=lambda s: iter(s))
        self.assertIn('loss/train', metrics)
        self.assertTrue(metrics['loss/train'] > 0)
        # warmup 首步 lr 极小，参数变化量在 1e-8 量级，用阈值而非 torch.equal
        max_delta = max((a - b).abs().max().item() for a, b in
                        zip(before, self.model.language.parameters()))
        self.assertGreater(max_delta, 0.0, '语言塔参数应被更新')

    def test_vision_tower_frozen_by_default(self):
        self.assertFalse(any(p.requires_grad for p in self.model.vision.parameters()))
        self.assertTrue(any(p.requires_grad for p in self.model.vision_proj.parameters()))

    def test_image_patch_id_locates_placeholders(self):
        """给 image_patch_id 时无需显式传 image_patch_indices。"""
        ids = torch.randint(4, VOCAB, (1, self.seq_len))
        ids[0, 3:3 + PATCHES] = PATCH_ID
        images = [torch.randn(3, 224, 224)]
        with torch.no_grad():
            a = self.model(input_ids=ids, images=images, image_patch_id=PATCH_ID)
            b = self.model(input_ids=ids, images=images,
                           image_patch_indices=torch.arange(3, 3 + PATCHES).unsqueeze(0))
        self.assertTrue(torch.allclose(a.logits, b.logits, atol=1e-6))

    def test_vision_tower_frozen_stays_unchanged_after_step(self):
        before = [p.detach().clone() for p in self.model.vision.parameters()]
        pipe = self._pipeline(lambda b: (
            {'input_ids': b[1], 'images': b[3], 'image_patch_id': PATCH_ID}, b[2]))
        pipe.train(self.stream, num_epochs=1, steps_per_epoch=1, batch_fn=lambda s: iter(s))
        for a, b in zip(before, self.model.vision.parameters()):
            self.assertTrue(torch.equal(a, b), '冻结的视觉塔不应被更新')


if __name__ == '__main__':
    unittest.main(verbosity=2)
