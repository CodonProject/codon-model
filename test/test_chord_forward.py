'''
MotifChord.forward 对 codon/res/codon.j2 占位符约定的适配测试。

覆盖链路：
    Session(codon.j2) -> <|modality_image_start|><pad>...<end|> / audio 同构
    Session.to_tensors() -> image_patch_indices / audio_patch_indices
    MotifChord.forward -> x[b, indices] = 模态特征（序列长度不变、位置不移位）

视觉/音频塔在 __init__ 里会 from_remote() 拉远程权重，测试里用假塔替换：
假塔返回「某个已知 token 的嵌入」，于是「占位符注入」应当与「把占位符换成该 token」
的前向逐元素相等 —— 这样一次断言就能同时校验注入位置、序列长度与 RoPE 位置。
'''

import os
import sys
import unittest

import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from unittest import mock

from codon.res import LM
from codon.model.cache import ModelCache
from codon.model.sampler import Sampler
from codon.utils.session import Session
from codon.utils.tokens import PackedTokenizer

import codon.motif.chord.model as chord_model
from codon.motif.chord.config import MotifChordConfig
from codon.motif.chord.model import MotifChord


# ---------------------------------------------------------------- 假模态塔

VISION_TOKENS = 6          # 假视觉塔每张图输出 6 行特征
AUDIO_TOKENS = 4           # 假音频塔每段 mel 输出 4 行特征
VISION_TARGET = 5          # 假视觉特征 == token_emb.weight[5]
AUDIO_TARGET = 7           # 假音频特征 == token_emb.weight[7]

_FAKE_WEIGHTS = {}


def _make_fake_projector(key: str, num_tokens: int, target_id: int):
    '''构造一个假模态塔：无论输入是什么，都返回 num_tokens 行 token_emb[target_id]。'''

    class _FakeProjector(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.num_tokens = num_tokens
            self.target_id = target_id

        def forward(self, x):
            weight = _FAKE_WEIGHTS[key]
            emb = weight[self.target_id]
            batch = x.shape[0]
            return emb.view(1, 1, -1).expand(batch, self.num_tokens, -1)

    return _FakeProjector


def build_model(vocab_size: int = 64, dropout: float = 0.0) -> MotifChord:
    config = MotifChordConfig(
        vocab_size=vocab_size,
        model_dim=32,
        num_layers=4,                 # make_layer_pattern 要求 4 的倍数
        num_heads=4,
        num_kv_heads=2,
        dropout=dropout,
        gdn_head_k_dim=16,
        vision_dim=8,
        audio_dim=8,
        audio_pool_stride=8,
    )
    vision_cls = _make_fake_projector('vision', VISION_TOKENS, VISION_TARGET)
    audio_cls = _make_fake_projector('audio', AUDIO_TOKENS, AUDIO_TARGET)

    with mock.patch.object(chord_model, 'MotifChordVisionProjector', vision_cls), \
         mock.patch.object(chord_model, 'MotifChordAudioProjector', audio_cls):
        model = MotifChord(config)

    _FAKE_WEIGHTS['vision'] = model.token_emb.weight
    _FAKE_WEIGHTS['audio'] = model.token_emb.weight
    model.eval()
    return model


# ---------------------------------------------------------------- 最小 tokenizer

SPECIALS = [
    '<|unk|>', '<|im_start|>', '<|im_end|>', '<|system|>', '<|user|>', '<|model|>',
    '<|thought_start|>', '<|thought_end|>',
    '<|effort_low|>', '<|effort_high|>', '<|effort_max|>',
    '<|modality_image_start|>', '<|modality_image_pad|>', '<|modality_image_end|>',
    '<|modality_audio_start|>', '<|modality_audio_pad|>', '<|modality_audio_end|>',
    '<|pad|>', '[unused_42]',
]
WORDS = ['hello', 'world']


def build_tokenizer() -> PackedTokenizer:
    '''A2 / chord 风格词表 + 真实的 codon.j2 模板。'''
    vocab = {tok: i for i, tok in enumerate(SPECIALS + WORDS)}
    raw = Tokenizer(WordLevel(vocab, unk_token='<|unk|>'))
    raw.add_special_tokens(SPECIALS)

    tok = PackedTokenizer(raw)
    with open(LM['jinja'], encoding='utf-8') as f:
        tok.set_chat_template(f.read())
    tok.config['unk_token'] = '<|unk|>'
    tok.config['pad_token'] = '<|pad|>'
    tok.config['eos_token'] = '<|im_end|>'
    return tok


class TestChordPlaceholderForward(unittest.TestCase):
    '''model.py：占位符就地注入'''

    PAD = 0

    def setUp(self):
        self.model = build_model()

    def _ids(self, pads=None, length=14):
        ids = torch.randint(8, 64, (1, length))
        if pads:
            for start, count in pads:
                ids[0, start:start + count] = self.PAD
        return ids

    def test_injection_equals_token_replacement(self):
        '''占位符注入 == 把占位符替换成对应 token（位置不移位、长度不变）。'''
        ids = self._ids(pads=[(3, VISION_TOKENS)])
        idx = torch.arange(3, 3 + VISION_TOKENS).unsqueeze(0)
        image = torch.randn(3, 32, 32)

        with torch.no_grad():
            injected = self.model(
                input_ids=ids, images=[image], image_patch_indices=idx
            ).logits

            replaced_ids = ids.clone()
            replaced_ids[0, 3:3 + VISION_TOKENS] = VISION_TARGET
            replaced = self.model(input_ids=replaced_ids).logits

            without = self.model(input_ids=ids).logits

        self.assertEqual(injected.shape, (1, ids.shape[1], self.model.config.vocab_size))
        self.assertTrue(torch.allclose(injected, replaced, atol=1e-6))
        self.assertFalse(torch.allclose(injected, without))

    def test_image_patch_id_scan_matches_explicit_indices(self):
        '''只给 image_patch_id 时，forward 自己扫出来的位置必须与显式 indices 等价。'''
        ids = self._ids(pads=[(2, VISION_TOKENS)])
        idx = torch.arange(2, 2 + VISION_TOKENS).unsqueeze(0)
        image = torch.randn(3, 32, 32)

        with torch.no_grad():
            explicit = self.model(
                input_ids=ids, images=[image], image_patch_indices=idx
            ).logits
            scanned = self.model(
                input_ids=ids, images=[image], image_patch_id=self.PAD
            ).logits

        self.assertTrue(torch.allclose(explicit, scanned))

    def test_batched_nested_images(self):
        '''每 batch 一行各自的图，占位符位置不同也要各写各的。'''
        ids = torch.full((2, 16), 9)
        starts = (2, 5)
        rows = []
        for b, start in enumerate(starts):
            ids[b, start:start + VISION_TOKENS] = self.PAD
            rows.append(list(range(start, start + VISION_TOKENS)))
        indices = torch.tensor(rows)
        images = [[torch.randn(3, 32, 32)] for _ in range(2)]

        with torch.no_grad():
            injected = self.model(
                input_ids=ids, images=images, image_patch_indices=indices
            ).logits

            replaced_ids = ids.clone()
            replaced_ids[ids == self.PAD] = VISION_TARGET
            replaced = self.model(input_ids=replaced_ids).logits

        self.assertTrue(torch.allclose(injected, replaced, atol=1e-6))

    def test_flat_image_list_broadcasts_over_batch(self):
        '''扁平 list（一张图）按 batch 广播，等价于每行各给同一张图。'''
        ids = torch.full((2, 16), 9)
        rows = []
        for b in range(2):
            ids[b, 3:3 + VISION_TOKENS] = self.PAD
            rows.append(list(range(3, 3 + VISION_TOKENS)))
        indices = torch.tensor(rows)
        image = torch.randn(3, 32, 32)

        with torch.no_grad():
            flat = self.model(
                input_ids=ids, images=[image], image_patch_indices=indices
            ).logits
            nested = self.model(
                input_ids=ids, images=[[image], [image]], image_patch_indices=indices
            ).logits

        self.assertTrue(torch.allclose(flat, nested))

    def test_audio_injection_matches_token_replacement(self):
        ids = self._ids(pads=[(4, AUDIO_TOKENS)])
        idx = torch.arange(4, 4 + AUDIO_TOKENS).unsqueeze(0)
        mel = torch.randn(80, 64)

        with torch.no_grad():
            injected = self.model(
                input_ids=ids, audios=[mel], audio_patch_indices=idx
            ).logits

            replaced_ids = ids.clone()
            replaced_ids[0, 4:4 + AUDIO_TOKENS] = AUDIO_TARGET
            replaced = self.model(input_ids=replaced_ids).logits

        self.assertTrue(torch.allclose(injected, replaced, atol=1e-6))

    def test_scan_patch_indices_pads_uneven_rows(self):
        ids = torch.full((2, 10), 9)
        ids[0, 2:5] = self.PAD
        ids[1, 4] = self.PAD

        scanned = MotifChord._scan_patch_indices(ids, self.PAD)

        self.assertEqual(scanned.tolist(), [[2, 3, 4], [4, -1, -1]])

    def test_patch_count_mismatch_raises(self):
        '''占位符数与特征行数不等时直接报错，不静默截断。'''
        ids = self._ids(pads=[(3, VISION_TOKENS - 1)])
        idx = torch.arange(3, 3 + VISION_TOKENS - 1).unsqueeze(0)

        with self.assertRaises(ValueError):
            self.model(
                input_ids=ids, images=[torch.randn(3, 32, 32)], image_patch_indices=idx
            )

    def test_indices_without_modality_raises(self):
        ids = self._ids(pads=[(3, VISION_TOKENS)])
        idx = torch.arange(3, 3 + VISION_TOKENS).unsqueeze(0)

        with self.assertRaises(ValueError):
            self.model(input_ids=ids, image_patch_indices=idx)

    def test_image_without_placeholder_raises(self):
        ids = self._ids()          # 没有占位符
        with self.assertRaises(ValueError):
            self.model(
                input_ids=ids,
                images=[torch.randn(3, 32, 32)],
                image_patch_indices=torch.full((1, 0), -1),
            )

    def test_cache_reuse_matches_full_forward(self):
        '''prefill 注入模态 + decode 复用 cache，末位 logits 要与整段前向一致。'''
        ids = self._ids(pads=[(3, VISION_TOKENS)])
        idx = torch.arange(3, 3 + VISION_TOKENS).unsqueeze(0)
        image = torch.randn(3, 32, 32)
        next_token = torch.tensor([[11]])

        cache = ModelCache()
        with torch.no_grad():
            prefill = self.model(
                input_ids=ids, images=[image], image_patch_indices=idx,
                past_key_values=cache,
            )
            prefill_len = cache.seq_length
            step = self.model(
                input_ids=next_token, start_pos=cache.seq_length, past_key_values=cache,
            )
            full = self.model(
                input_ids=torch.cat([ids, next_token], dim=1),
                images=[image], image_patch_indices=idx,
            )

        self.assertEqual(prefill_len, ids.shape[1])
        self.assertEqual(cache.seq_length, ids.shape[1] + 1)
        self.assertTrue(torch.allclose(
            step.logits[:, -1, :], full.logits[:, -1, :], atol=1e-3, rtol=1e-3
        ))
        self.assertEqual(prefill.logits.shape[:2], (1, ids.shape[1]))

    def test_cache_layer_lengths_agree_across_hybrid_layers(self):
        '''GDN（steps 计数）与 GQA（KV 长度）两层 cache 的 seq_length 必须一致。'''
        ids = self._ids(pads=[(3, VISION_TOKENS)])
        idx = torch.arange(3, 3 + VISION_TOKENS).unsqueeze(0)
        cache = ModelCache()
        layers = len(self.model.decoder)

        with torch.no_grad():
            self.model(
                input_ids=ids, images=[torch.randn(3, 32, 32)],
                image_patch_indices=idx, past_key_values=cache,
            )
            lengths = [cache[i].seq_length for i in range(layers)]

            self.model(
                input_ids=torch.tensor([[11]]),
                start_pos=cache.seq_length, past_key_values=cache,
            )
            lengths_after = [cache[i].seq_length for i in range(layers)]

        self.assertEqual(lengths, [ids.shape[1]] * layers)
        self.assertEqual(lengths_after, [ids.shape[1] + 1] * layers)
        # 混合架构：前 3 层 GDN、第 4 层 GQA（make_layer_pattern）
        self.assertEqual(type(cache[0]).__name__, 'GatedDeltaAttentionLayerCache')
        self.assertEqual(type(cache[layers - 1]).__name__, 'KVLayerCache')

    def test_cache_with_attention_mask_matches_full_forward(self):
        '''chat() 的路径：prefill 带 mask 且注入模态，decode 不传 mask。'''
        ids = self._ids(pads=[(3, VISION_TOKENS)])
        idx = torch.arange(3, 3 + VISION_TOKENS).unsqueeze(0)
        image = torch.randn(3, 32, 32)
        next_token = torch.tensor([[11]])

        cache = ModelCache()
        with torch.no_grad():
            self.model(
                input_ids=ids, images=[image], image_patch_indices=idx,
                mask=torch.ones(1, ids.shape[1], dtype=torch.long),
                past_key_values=cache,
            )
            step = self.model(
                input_ids=next_token, start_pos=cache.seq_length, past_key_values=cache,
            )
            full = self.model(
                input_ids=torch.cat([ids, next_token], dim=1),
                images=[image], image_patch_indices=idx,
                mask=torch.ones(1, ids.shape[1] + 1, dtype=torch.long),
            )

        self.assertTrue(torch.allclose(
            step.logits[:, -1, :], full.logits[:, -1, :], atol=1e-3, rtol=1e-3
        ))

    def test_batched_cache_matches_full_forward(self):
        '''B=2 各自带图 + cache 逐 token 解码，末位 logits 仍要与整段前向一致。'''
        ids = torch.full((2, 16), 9)
        rows = []
        for b, start in enumerate((2, 5)):
            ids[b, start:start + VISION_TOKENS] = self.PAD
            rows.append(list(range(start, start + VISION_TOKENS)))
        indices = torch.tensor(rows)
        images = [[torch.randn(3, 32, 32)] for _ in range(2)]
        next_tokens = torch.tensor([[11], [12]])

        cache = ModelCache()
        with torch.no_grad():
            self.model(
                input_ids=ids, images=images, image_patch_indices=indices,
                past_key_values=cache,
            )
            step = self.model(
                input_ids=next_tokens, start_pos=cache.seq_length, past_key_values=cache,
            )
            full = self.model(
                input_ids=torch.cat([ids, next_tokens], dim=1),
                images=images, image_patch_indices=indices,
            )

        self.assertTrue(torch.allclose(
            step.logits[:, -1, :], full.logits[:, -1, :], atol=1e-3, rtol=1e-3
        ))

    def test_incremental_prefill_with_new_image_matches_full_forward(self):
        '''两段增量 prefill（第二段带新图）与「一次整段前向」等价。'''
        image_a, image_b = torch.randn(3, 32, 32), torch.randn(3, 32, 32)

        first = torch.tensor([[9, 10, self.PAD, self.PAD, self.PAD, self.PAD, self.PAD, self.PAD, 11]])
        second = torch.tensor([[12, self.PAD, self.PAD, self.PAD, self.PAD, self.PAD, self.PAD, 13]])

        idx_first = torch.arange(2, 2 + VISION_TOKENS).unsqueeze(0)
        idx_second = torch.arange(1, 1 + VISION_TOKENS).unsqueeze(0)

        cache = ModelCache()
        with torch.no_grad():
            self.model(
                input_ids=first, images=[image_a], image_patch_indices=idx_first,
                past_key_values=cache,
            )
            first_len = cache.seq_length
            second_out = self.model(
                input_ids=second, images=[image_b], image_patch_indices=idx_second,
                start_pos=cache.seq_length, past_key_values=cache,
            )

            full_ids = torch.cat([first, second], dim=1)
            full_indices = torch.cat(
                [idx_first, idx_second + first.shape[1]], dim=1
            )
            full = self.model(
                input_ids=full_ids, images=[image_a, image_b],
                image_patch_indices=full_indices,
            )

        self.assertEqual(first_len, first.shape[1])
        self.assertEqual(cache.seq_length, first.shape[1] + second.shape[1])
        self.assertTrue(torch.allclose(
            second_out.logits[:, -1, :], full.logits[:, -1, :], atol=1e-3, rtol=1e-3
        ))

    def test_generate_resumes_from_non_empty_cache(self):
        '''generate 收到非空 cache 时从 cache 长度续上位置，而不是把位置重置为 0。'''
        first = torch.tensor([[9, 10, self.PAD, self.PAD, self.PAD, self.PAD, self.PAD, self.PAD, 11]])
        second = torch.tensor([[12, 13, 14]])
        idx_first = torch.arange(2, 2 + VISION_TOKENS).unsqueeze(0)
        image = torch.randn(3, 32, 32)

        cache = ModelCache()
        with torch.no_grad():
            self.model(
                input_ids=first, images=[image], image_patch_indices=idx_first,
                past_key_values=cache,
            )
            cached_len = cache.seq_length

        out = self.model.generate(
            input_ids=second,
            max_new_tokens=2,
            sampler=Sampler(temperature=0.7),
            past_key_values=cache,
        )

        self.assertEqual(out.shape, (1, second.shape[1] + 2))
        self.assertEqual(cache.seq_length, cached_len + second.shape[1] + 2 - 1)

    def test_history_modality_not_needed_when_cache_is_retained(self):
        '''
        保留 cache 续写时，历史图片**不必**再传：图像位置的 K/V 已经在 cache 里，
        第二轮连视觉塔都不会被调用。
        '''
        image = torch.randn(3, 32, 32)
        chunk1 = torch.tensor([[9, 10, self.PAD, self.PAD, self.PAD,
                                self.PAD, self.PAD, self.PAD, 11]])
        chunk2 = torch.tensor([[12, 13, 14]])
        idx1 = torch.arange(2, 2 + VISION_TOKENS).unsqueeze(0)

        # 统计视觉塔调用次数：第二轮不应该再编码历史图片
        vision = self.model.vision
        counter = {'n': 0}

        class _Counting(torch.nn.Module):
            def forward(self, x):
                counter['n'] += 1
                return vision(x)

        self.model.vision = _Counting()

        cache = ModelCache()
        with torch.no_grad():
            self.model(
                input_ids=chunk1, images=[image], image_patch_indices=idx1,
                past_key_values=cache,
            )
            self.assertEqual(counter['n'], 1)

            # 第二轮：只喂新 token，不带历史图片，也不带任何图片
            turn2 = self.model(
                input_ids=chunk2, start_pos=cache.seq_length, past_key_values=cache,
            )

            # 参考：整段重算（历史图片照常提供）
            full = self.model(
                input_ids=torch.cat([chunk1, chunk2], dim=1),
                images=[image], image_patch_indices=idx1,
            )

        self.assertEqual(counter['n'], 2)          # 只有一次是参考整段重算触发的
        self.assertTrue(torch.allclose(
            turn2.logits[:, -1, :], full.logits[:, -1, :], atol=1e-3, rtol=1e-3
        ))

        # 反过来：第二轮「再传一次历史图」会被当成「chunk2 自己带图」，而 chunk2 没有占位符
        # -> 必须显式报错，而不是静默重复注入或静默忽略。
        with self.assertRaises(ValueError):
            self.model(
                input_ids=chunk2,
                images=[image],
                image_patch_indices=torch.full((1, 0), -1, dtype=torch.long),
                start_pos=cache.seq_length,
                past_key_values=cache,
            )

    def test_generate_prefill_decode(self):
        ids = self._ids(pads=[(3, VISION_TOKENS)])
        idx = torch.arange(3, 3 + VISION_TOKENS).unsqueeze(0)
        image = torch.randn(3, 32, 32)

        out = self.model.generate(
            input_ids=ids,
            images=[image],
            image_patch_indices=idx,
            max_new_tokens=3,
            sampler=Sampler(temperature=0.7),
        )

        self.assertEqual(out.shape, (1, ids.shape[1] + 3))
        self.assertTrue(torch.equal(out[:, :ids.shape[1]], ids))

    def test_meta_declares_modalities(self):
        self.assertTrue(MotifChord.meta.supports_image)
        self.assertTrue(MotifChord.meta.supports_audio)
        self.assertTrue(MotifChord.meta.supports_thinking)
        self.assertFalse(MotifChord.meta.supports_tool)
        self.assertEqual(MotifChord.audio_subtypes, ('speech', 'music'))
        self.assertTrue(MotifChord.meta.supports_audio_subtype('音乐'))
        self.assertFalse(MotifChord.meta.supports_audio_subtype('general'))


class TestChatCacheIntegration(unittest.TestCase):
    '''utils/generate.chat + MotifChord + ModelCache：模态只在 prefill 注入，位置由 cache 续上。'''

    def test_chat_drives_modelcache_positions(self):
        from codon.utils.generate import chat

        tokenizer = build_tokenizer()
        model = build_model(vocab_size=tokenizer.vocab_size)

        # 32x48 -> (32//16) * (48//16) = 6 == VISION_TOKENS
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': torch.randn(3, 32, 48)},
            {'type': 'text', 'text': 'hello'},
        ]}]

        calls = []
        forward = model.forward

        def spy(*args, **kwargs):
            calls.append({
                'start_pos': kwargs.get('start_pos'),
                'seq_len': int(kwargs['input_ids'].shape[1]),
                'images': kwargs.get('images'),
                'mask': kwargs.get('mask'),
            })
            return forward(*args, **kwargs)

        model.forward = spy                     # 实例属性，不注册成子模块
        chunks = list(chat(
            model=model,
            tokenizer=tokenizer,
            device=torch.device('cpu'),
            messages=messages,
            max_new_tokens=3,
            enable_thinking=False,
        ))

        self.assertGreaterEqual(len(calls), 2)
        prefill = calls[0]
        self.assertEqual(prefill['start_pos'], 0)
        self.assertIsNotNone(prefill['images'])          # 模态只在 prefill
        self.assertIsNotNone(prefill['mask'])
        self.assertEqual(prefill['images'][0].shape[-2:], (32, 48))

        # decode 步：每次 1 个 token，位置严格按 ModelCache 的 seq_length 递增（不重头算）
        for offset, call in enumerate(calls[1:], start=1):
            self.assertEqual(call['seq_len'], 1)
            self.assertIsNone(call['images'])
            self.assertIsNone(call['mask'])
            self.assertEqual(call['start_pos'], prefill['seq_len'] + offset - 1)

        self.assertIn(chunks[-1].finish_reason, ('stop', 'length'))

    def test_chat_cache_logits_match_no_cache_reference(self):
        '''chat 走 ModelCache 的逐步 logits，与「每步整段重算（无 cache）」逐位一致。'''
        from codon.utils.generate import chat

        tokenizer = build_tokenizer()
        model = build_model(vocab_size=tokenizer.vocab_size)
        image = torch.randn(3, 32, 48)
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': 'hello'},
        ]}]

        recorded = []
        forward = model.forward

        def spy(*args, **kwargs):
            out = forward(*args, **kwargs)
            recorded.append({
                'input_ids': kwargs['input_ids'].clone(),
                'logits': out.logits[:, -1, :].clone(),
                'image_patch_indices': kwargs.get('image_patch_indices'),
            })
            return out

        model.forward = spy
        list(chat(
            model=model, tokenizer=tokenizer, device=torch.device('cpu'),
            messages=messages, max_new_tokens=4, enable_thinking=False,
        ))
        model.forward = forward

        self.assertGreaterEqual(len(recorded), 2)
        prefill = recorded[0]
        prompt_len = int(prefill['input_ids'].shape[1])

        # 由 cache 路径实际处理的序列拼出完整 token 序列
        sequence = prefill['input_ids']
        for call in recorded[1:]:
            sequence = torch.cat([sequence, call['input_ids']], dim=1)

        modal = {
            'images': [image],
            'image_patch_indices': prefill['image_patch_indices'],
        }
        with torch.no_grad():
            for step, call in enumerate(recorded):
                prefix = sequence[:, :prompt_len + step]
                reference = forward(
                    input_ids=prefix,
                    mask=torch.ones(1, prefix.shape[1], dtype=torch.long),
                    **modal,
                ).logits[:, -1, :]
                self.assertTrue(
                    torch.allclose(call['logits'], reference, atol=1e-3, rtol=1e-3),
                    f'step {step}: cache logits 与整段重算不一致',
                )


class TestSessionCodonTemplate(unittest.TestCase):
    '''session.py：codon.j2 三连 token 的占位符展开 + 模态能力开关'''

    def setUp(self):
        self.tok = build_tokenizer()

    def _ids_of(self):
        return {
            name: self.tok.token_to_id(f'<|modality_{name}|>')
            for name in ('image_start', 'image_pad', 'image_end',
                         'audio_start', 'audio_pad', 'audio_end')
        }

    def test_image_and_audio_placeholders_expand(self):
        ids_of = self._ids_of()
        session = Session(self.tok, patch_size=16, audio_pool_stride=8)

        image = torch.randn(3, 32, 48)      # (32//16) * (48//16) = 2 * 3 = 6
        mel = torch.randn(80, 64)           # ((64-1)//2+1)//8 = 32//8 = 4

        session.add_message({'role': 'user', 'content': [
            {'type': 'image', 'image': image},
            {'type': 'audio', 'audio': mel},
            {'type': 'text', 'text': 'hello'},
        ]})
        t = session.to_tensors(device='cpu', batch_dim=True)
        seq = t['input_ids'][0].tolist()

        img_start = seq.index(ids_of['image_start'])
        img_end = seq.index(ids_of['image_end'])
        between = seq[img_start + 1:img_end]
        self.assertEqual(len(between), 6)
        self.assertTrue(all(tok == ids_of['image_pad'] for tok in between))

        aud_start = seq.index(ids_of['audio_start'])
        aud_end = seq.index(ids_of['audio_end'])
        between = seq[aud_start + 1:aud_end]
        self.assertEqual(len(between), 4)
        self.assertTrue(all(tok == ids_of['audio_pad'] for tok in between))

        self.assertEqual(
            t['image_patch_indices'][0].tolist(),
            list(range(img_start + 1, img_end)),
        )
        self.assertEqual(
            t['audio_patch_indices'][0].tolist(),
            list(range(aud_start + 1, aud_end)),
        )

        # 占位符不进 language modeling loss
        for idx in t['image_patch_indices'][0].tolist() + t['audio_patch_indices'][0].tolist():
            self.assertEqual(int(t['labels'][0][idx]), -100)
            self.assertEqual(int(t['attention_mask'][0][idx]), 1)

        self.assertEqual(len(t['images']), 1)
        self.assertEqual(len(t['audios']), 1)
        self.assertTrue(torch.equal(t['images'][0], image))
        self.assertTrue(torch.equal(t['audios'][0], mel))

    def test_model_consumes_session_output(self):
        '''Session.to_tensors() -> MotifChord.forward 端到端跑通（占位符数与假塔输出一致）。'''
        session = Session(self.tok, patch_size=16, audio_pool_stride=8)
        session.add_message({'role': 'user', 'content': [
            {'type': 'image', 'image': torch.randn(3, 32, 48)},   # (32//16) * (48//16) = 6
            {'type': 'audio', 'audio': torch.randn(80, 64)},      # 4
            {'type': 'text', 'text': 'world'},
        ]})
        session.add_generation_prompt(enable_thinking=False)
        t = session.to_tensors(device='cpu', batch_dim=True)

        model = build_model(vocab_size=self.tok.vocab_size)
        with torch.no_grad():
            out = model(
                input_ids=t['input_ids'],
                images=t['images'],
                image_patch_indices=t['image_patch_indices'],
                audios=t['audios'],
                audio_patch_indices=t['audio_patch_indices'],
                mask=t['attention_mask'],
            )

        self.assertEqual(out.logits.shape, (1, t['input_ids'].shape[1], self.tok.vocab_size))

    def test_capability_switch_off_renders_fallback_text(self):
        '''image_capab=False 时模板渲染 "unsupported" fallback 文本，不产生占位符。'''
        session = Session(self.tok, patch_size=16, image_capab=False)
        session.add_message({'role': 'user', 'content': [
            {'type': 'image', 'image': torch.randn(3, 32, 32)},
        ]})

        seq = session.input_ids
        self.assertNotIn(self.tok.token_to_id('<|modality_image_start|>'), seq)
        self.assertNotIn(self.tok.token_to_id('<|modality_image_pad|>'), seq)
        self.assertEqual(session.to_tensors()['image_patch_indices'].numel(), 0)

        between = seq[
            seq.index(session.token_id('user')) + 1: seq.index(session.token_id('im_end'))
        ]
        self.assertEqual(len(between), 1)      # fallback 文本（玩具词表里是 unk）

    def test_audio_capability_auto_on(self):
        '''带了 mel 的消息自动按「支持音频」渲染出音频占位符。'''
        session = Session(self.tok, audio_pool_stride=8)
        session.add_message({'role': 'user', 'content': [
            {'type': 'audio', 'audio': torch.randn(80, 64)},
        ]})

        self.assertIn('<|modality_audio_start|>', session.decode())
        self.assertEqual(session.to_tensors()['audio_patch_indices'].numel(), 4)

    def test_text_only_message_has_no_placeholders(self):
        session = Session(self.tok, patch_size=16, audio_pool_stride=8)
        session.add_message({'role': 'user', 'content': 'hello world'})
        t = session.to_tensors()

        self.assertEqual(t['image_patch_indices'].numel(), 0)
        self.assertEqual(t['audio_patch_indices'].numel(), 0)
        self.assertEqual(t['images'], [])
        self.assertEqual(t['audios'], [])

    def test_legacy_two_token_image_form_still_expands(self):
        '''旧模板 <start><end> 两连形态仍然按原来的方式在中间插入 patch。'''
        template = (
            "{% for m in messages %}<|im_start|><|user|>"
            "{% if m['content'] is string %}{{ m['content'] }}{% else %}"
            "{% for it in m['content'] %}"
            "{% if it['type'] == 'image' %}<|modality_image_start|><|modality_image_end|>"
            "{% else %}{{ it['text'] }}{% endif %}{% endfor %}{% endif %}<|im_end|>"
            "{% endfor %}"
        )
        tok = build_tokenizer()
        tok.set_chat_template(template)
        session = Session(tok, patch_size=16)
        session.add_message({'role': 'user', 'content': [
            {'type': 'image', 'image': torch.randn(3, 32, 48)},
        ]})
        t = session.to_tensors()

        self.assertEqual(t['image_patch_indices'].numel(), 6)


class TestAudioTokenCount(unittest.TestCase):
    '''Session 的音频占位符数与真实 Whisper Tiny + pool 输出对齐。'''

    def test_matches_whisper_tiny_pool(self):
        from codon.impl.whisper_tiny import WhisperTinyAudioEncoder

        encoder = WhisperTinyAudioEncoder(pool_stride=8)
        encoder.eval()
        session = Session(build_tokenizer(), audio_pool_stride=8)

        for frames in (3000, 1600, 64):
            with torch.no_grad():
                feats, _ = encoder(torch.randn(1, 80, frames))
            self.assertEqual(
                feats.shape[1],
                session._audio_token_count(torch.randn(80, frames)),
                msg=f'frames={frames}',
            )

        with torch.no_grad():
            feats, _ = encoder(torch.randn(1, 80, 3000))
        self.assertEqual(feats.shape[1], 187)          # 30s -> 187，与文档一致

        mel = torch.randn(80, 64)
        self.assertEqual(Session(build_tokenizer(), audio_pool_stride=1)._audio_token_count(mel), 32)
        self.assertEqual(Session(build_tokenizer(), audio_pool_stride=8)._audio_token_count(mel), 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
