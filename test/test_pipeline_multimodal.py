'''
MotifChord 与 codon.pipeline.pretrain / codon.pipeline.sft 的适配测试。

复用 test_chord_forward.py 里的假塔 MotifChord（真实模型 __init__ 要拉远程权重）：

    pretrain: `PretrainConfig.batch_builder` -> (model_kwargs, labels)，
              训练循环 `self.model(**model_kwargs)`，模态字段能透传；
    sft:      `SFTPipeline.train_step` 按 forward 签名挑 batch 字段
              （images / *_patch_indices / attention_mask -> mask），纯文本模型不受影响。
'''

import math
import os
import shutil
import sys
import unittest
import uuid
from unittest import mock

import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'test'))

from test_chord_forward import (          # noqa: E402  (复用测试夹具)
    VISION_TOKENS,
    build_model,
    build_tokenizer,
)

from codon.model.types.language import CausalLanguageModel, CausalLanguageModelOutput
from codon.pipeline.base import move_to_device, select_model_kwargs
from codon.pipeline.pretrain import PretrainConfig, PretrainPipeline
from codon.pipeline.sft import SFTConfig, SFTPipeline, SFTStage, build_sft_stages


def _tmpdir(testcase, prefix: str) -> str:
    '''
    工作区内的临时目录 + 自动清理。

    两个坑：`tempfile.mkdtemp` 建出来的目录在受限沙箱里写入会被拒（ACL），
    而系统临时目录根本不在允许写入的范围内，所以这里用 `os.makedirs` + 唯一后缀。
    '''
    path = os.path.join(project_root, f'{prefix}{uuid.uuid4().hex[:8]}')
    os.makedirs(path, exist_ok=True)
    testcase.addCleanup(shutil.rmtree, path, True)
    return path


def _setup_quiet(pipe):
    '''
    `pipeline.setup()` 会注册「进程退出兜底保存 checkpoint」的 atexit 回调。

    测试里屏蔽掉：既不会在退出时往磁盘写 `last.pt`，也不会被文件沙箱卡住。
    '''
    from codon.utils.lifecycle import exit_manager

    with mock.patch.object(
        exit_manager, 'register', lambda *args, **kwargs: args[0] if args else None
    ):
        pipe.setup()
    return pipe


class _TextOnlyLM(CausalLanguageModel):
    '''未声明模态能力、forward 也不吃模态字段的纯文本模型。'''

    def __init__(self, vocab_size: int, model_dim: int = 16):
        super().__init__()
        self.token_emb = torch.nn.Embedding(vocab_size, model_dim)
        self.proj_out = torch.nn.Linear(model_dim, vocab_size, bias=False)

    def forward(self, input_ids, start_pos=0, past_key_values=None, output_attentions=False):
        return CausalLanguageModelOutput(logits=self.proj_out(self.token_emb(input_ids)))


def _counting_vision(model):
    '''给假视觉塔套一层计数器，用来断言「模态真的被编码了」。'''
    vision = model.vision
    counter = {'n': 0}

    class _Counting(torch.nn.Module):
        def forward(self, x):
            counter['n'] += 1
            return vision(x)

    model.vision = _Counting()
    return counter


def _make_batch(vocab_size: int, batch_size: int = 2, seq_len: int = 12):
    '''构造一条「文本 + 图像占位符」的 batch（labels 在占位符处 -100）。

    images 用嵌套写法（每 batch 行各一张图）；扁平 list 的语义是「每行都带这些图」。
    '''
    input_ids = torch.randint(8, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    for b in range(batch_size):
        input_ids[b, 2:2 + VISION_TOKENS] = 0
        labels[b, 2:2 + VISION_TOKENS] = -100
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.long),
        'images': [[torch.randn(3, 32, 32)] for _ in range(batch_size)],
        'image_patch_indices': torch.stack([
            torch.arange(2, 2 + VISION_TOKENS) for _ in range(batch_size)
        ]),
    }


class TestModelKwargsSelection(unittest.TestCase):
    '''pipeline.base：按 forward 签名挑字段'''

    def test_text_model_gets_only_supported_keys(self):
        model = _TextOnlyLM(32)
        picked = select_model_kwargs(model, _make_batch(32, batch_size=1))

        self.assertEqual(set(picked), {'input_ids'})    # 不吃 mask/images，不会 TypeError

    def test_multimodal_model_gets_aliased_mask(self):
        model = build_model()
        batch = _make_batch(model.config.vocab_size, batch_size=2)

        picked = select_model_kwargs(model, batch)

        self.assertEqual(
            set(picked), {'input_ids', 'mask', 'images', 'image_patch_indices'}
        )
        self.assertIs(picked['mask'], batch['attention_mask'])   # attention_mask -> mask

    def test_move_to_device_keeps_nesting(self):
        nested = [[torch.zeros(2), torch.ones(2)], [torch.full((2,), 2.0)]]
        moved = move_to_device(nested, torch.device('cpu'))

        self.assertIsInstance(moved, list)
        self.assertIsInstance(moved[0], list)
        self.assertEqual(len(moved[0]), 2)
        self.assertEqual(len(moved[1]), 1)


class TestPretrainMultimodal(unittest.TestCase):
    '''pipeline.pretrain：batch_builder 透传模态'''

    def setUp(self):
        self.tokenizer = build_tokenizer()
        self.model = build_model(vocab_size=self.tokenizer.vocab_size)
        self.counter = _counting_vision(self.model)
        self.tmpdir = _tmpdir(self, 'chord_pretrain_')

    def _pipeline(self, builder):
        config = PretrainConfig(
            compiled=False,
            use_progress=False,
            save_every_steps=0,
            sanity_every_steps=0,
            ckpt_dir=self.tmpdir,
            base_context=64,
            target_context=64,
            global_batch_tokens=64 * 2,
            step_mode='min',
            batch_builder=builder,
        )
        pipe = PretrainPipeline(self.model, tokenizer=self.tokenizer, config=config)
        return _setup_quiet(pipe)

    def test_pretrain_step_drives_modality(self):
        data = _make_batch(self.tokenizer.vocab_size, batch_size=2)
        images, indices = data['images'], data['image_patch_indices']
        batch = (torch.randn(0), data['input_ids'], data['labels'])   # (stage, inputs, labels)

        def builder(b):
            return (
                {'input_ids': b[1], 'images': images, 'image_patch_indices': indices},
                b[2],
            )

        metrics = self._pipeline(builder).train_step(batch)

        # 每 batch 行一张图 -> 视觉塔 2 次（模态真的被编码并注入；修复 SFT 前这里是 0）
        self.assertEqual(self.counter['n'], 2)
        self.assertTrue(math.isfinite(metrics['loss/train']))
        self.assertGreaterEqual(metrics['loss/train'], 0.0)
        self.assertIsNotNone(self.model.token_emb.weight.grad)

    def test_pretrain_supports_image_patch_id(self):
        '''只给 image_patch_id 时 forward 自己扫占位符（训练管线常这么喂）。'''
        data = _make_batch(self.tokenizer.vocab_size, batch_size=2)
        images = data['images']
        batch = (torch.randn(0), data['input_ids'], data['labels'])

        def builder(b):
            return (
                {'input_ids': b[1], 'images': images, 'image_patch_id': 0},
                b[2],
            )

        metrics = self._pipeline(builder).train_step(batch)

        self.assertEqual(self.counter['n'], 2)          # 两行各一张图
        self.assertEqual(metrics['loss/train'], metrics['loss/train'])   # 非 NaN

    def test_pretrain_text_only_still_works(self):
        '''默认 batch_builder（纯文本 (stage, inputs, labels)）不受影响。'''
        input_ids = torch.randint(8, self.tokenizer.vocab_size, (2, 12))

        pipe = self._pipeline(None)
        metrics = pipe.train_step((torch.randn(0), input_ids, input_ids.clone()))

        self.assertEqual(self.counter['n'], 0)
        self.assertTrue(math.isfinite(metrics['loss/train']))


class _StubStageDataset:
    '''最小 SFT stage：__getitem__ 直接返回预批 batch dict。'''

    def __init__(self, batch):
        self.batch = batch

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.batch


class TestSFTMultimodal(unittest.TestCase):
    '''pipeline.sft：模态字段透传 + 数据集占位符参数'''

    def setUp(self):
        self.tokenizer = build_tokenizer()
        self.model = build_model(vocab_size=self.tokenizer.vocab_size)
        self.counter = _counting_vision(self.model)
        self.tmpdir = _tmpdir(self, 'chord_sft_')

    def _pipeline(self, batch):
        config = SFTConfig(
            compiled=False, use_progress=False, save_every_steps=0,
            probe_every_steps=0, ckpt_dir=self.tmpdir,
        )
        stage = SFTStage(name='stage0', dataset=_StubStageDataset(batch), epochs=1)
        pipe = SFTPipeline(self.model, tokenizer=self.tokenizer, stages=[stage], config=config)
        return _setup_quiet(pipe)

    def test_sft_step_drives_modality(self):
        batch = _make_batch(self.tokenizer.vocab_size, batch_size=2)

        metrics = self._pipeline(batch).train_step(batch)

        self.assertEqual(self.counter['n'], 2)          # 两行各一张图（修复前这里是 0）
        self.assertTrue(math.isfinite(metrics['loss/train']))
        self.assertGreaterEqual(metrics['loss/train'], 0.0)
        self.assertIsNotNone(self.model.token_emb.weight.grad)

    def test_sft_text_only_batch_unchanged(self):
        input_ids = torch.randint(8, self.tokenizer.vocab_size, (2, 12))
        batch = {
            'input_ids': input_ids,
            'labels': input_ids.clone(),
            'attention_mask': torch.ones(2, 12, dtype=torch.long),
        }

        metrics = self._pipeline(batch).train_step(batch)

        self.assertEqual(self.counter['n'], 0)
        self.assertTrue(math.isfinite(metrics['loss/train']))

    def test_pad_positions_do_not_affect_earlier_logits(self):
        '''右填充 + labels=-100：改 padding 区的 token 不影响前面位置的 logits。

        （GDN 状态只向后传播、GQA 的 padding key 被 attention_mask 屏蔽，
        所以右填充对 batch 内真实 token 的训练信号是安全的。）
        '''
        data = _make_batch(self.tokenizer.vocab_size, batch_size=1, seq_len=16)
        data['input_ids'][0, 10:] = 7
        data['attention_mask'][0, 10:] = 0
        data['labels'][0, 10:] = -100

        perturbed = {k: (v.clone() if torch.is_tensor(v) else list(v)) for k, v in data.items()}
        perturbed['input_ids'][0, 11:] = 9

        with torch.no_grad():
            first = self.model(**select_model_kwargs(self.model, data)).logits
            second = self.model(**select_model_kwargs(self.model, perturbed)).logits

        self.assertTrue(torch.allclose(first[:, :10], second[:, :10], atol=1e-6))
        self.assertEqual(self.counter['n'], 2)          # 两次都注入了模态

    def test_build_sft_stages_injects_model_patch_size(self):
        '''build_sft_stages 把 patch_size 落到数据集 -> Session。'''
        folder = _tmpdir(self, 'chord_sft_data_')
        with open(os.path.join(folder, 'rows.jsonl'), 'w', encoding='utf-8') as f:
            f.write('{"input": "hello", "content": "world"}\n')

        captured = {}

        import codon.data.sft as sft_data
        real_session = sft_data.Session

        def spy_session(*args, **kwargs):
            captured.update(kwargs)
            return real_session(*args, **kwargs)

        sft_data.Session = spy_session
        try:
            stages = build_sft_stages(
                [{'name': 's0', 'folder': folder}],
                self.tokenizer, pad_length=32, batch_size=1,
                patch_size=16, audio_pool_stride=8,
            )
            dataset = stages[0].dataset
            dataset._build_sample(dataset.groups[0])
        finally:
            sft_data.Session = real_session

        self.assertEqual(dataset.patch_size, 16)
        self.assertEqual(captured.get('patch_size'), 16)
        self.assertEqual(captured.get('audio_pool_stride'), 8)

    def test_custom_dataset_without_patch_size_still_builds(self):
        '''自定义数据集没有这两个形参时不会被注入，不报 TypeError。'''

        class _Tiny:
            def __init__(self, folder, tokenizer, pad_length, batch_size):
                self.args = (folder, pad_length, batch_size)

            def __len__(self):
                return 1

        folder = _tmpdir(self, 'chord_sft_custom_')
        stages = build_sft_stages(
            [{'name': 's0', 'folder': folder}],
            self.tokenizer, pad_length=32, batch_size=1,
            dataset_cls=_Tiny, patch_size=16, audio_pool_stride=8,
        )

        self.assertEqual(stages[0].dataset.args, (folder, 32, 1))


if __name__ == '__main__':
    unittest.main(verbosity=2)
