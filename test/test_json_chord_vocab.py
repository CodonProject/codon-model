'''
新词表（codon/res 风格，如仓库里的 chord.zip）下的强制 JSON 兼容性测试。

chord.zip 是 A2 词表：所有特殊 token 都是 `<|...|>` 风格（`<|thought_end|>` /
`<|tool_name_divider|>` / `<|im_end|>`…），**没有** A1 的 `[cot_end]` / `[im_end]`。

    python test/test_json_chord_vocab.py

覆盖：
1. Session / resolve_token_id 能按逻辑名解析两种风格的 token；
2. 强制 JSON 的词表里剔除了全部 codon/res 保留 token（含 <|tool_name_divider|>）；
3. chat() 在 chord 词表下能正确定位思考段结束 -> 只在答案段强制 JSON，
   并且不会把结构 token 泄漏成正文，最后用 <|im_end|> 收尾。
'''

import os
import sys
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import unittest

import torch

from codon.utils.tokens import PackedTokenizer
from codon.utils.session import Session, resolve_token_id, resolve_token_name, token_name_table
from codon.model.grammar import JSONConstraint, TokenVocab, build_json_constraint
from codon.model.sampler import Sampler
from codon.model.cache import ModelCache
from codon.model.types.language import CausalLanguageModelOutput
from codon.utils.generate import chat

CHORD_PATH = os.path.join(project_root, 'chord.zip')

SPEC_TOKENS = [
    '<|im_start|>', '<|im_end|>', '<|system|>', '<|user|>', '<|model|>', '<|tool_response|>',
    '<|thought_start|>', '<|effort_low|>', '<|effort_high|>', '<|effort_max|>', '<|thought_end|>',
    '<|tool_call_start|>', '<|tool_call_end|>', '<|tool_name_divider|>',
    '<|modality_image_start|>', '<|modality_image_pad|>', '<|modality_image_end|>',
    '<|modality_audio_start|>', '<|modality_audio_pad|>', '<|modality_audio_end|>',
    '<|modality_video_start|>', '<|modality_video_pad|>', '<|modality_video_end|>',
    '<|fim_prefix|>', '<|fim_suffix|>', '<|fim_middle|>', '<|safe_escape|>',
    '<|pad|>', '<|unk|>', '<|sep|>',
]


def load_tokenizer():
    return PackedTokenizer(CHORD_PATH)


class _ScriptedModel:
    '''按脚本给出 logits 的假模型：第 n 次 forward 抬高 script[n] 的 logits。'''

    def __init__(self, script, vocab_size):
        self.script = list(script)
        self.vocab_size = vocab_size
        self.step = 0

    def eval(self):
        return self

    def forward(self, input_ids, start_pos=0, past_key_values=None, **kwargs):
        logits = torch.full((1, input_ids.shape[-1], self.vocab_size), -20.0)
        target = self.script[min(self.step, len(self.script) - 1)]
        logits[0, -1, target] = 20.0
        self.step += 1
        return CausalLanguageModelOutput(logits=logits, past_key_values=None)


@unittest.skipUnless(os.path.exists(CHORD_PATH), 'chord.zip not found')
class TestChordTokenResolution(unittest.TestCase):
    def setUp(self):
        self.tokenizer = load_tokenizer()

    def test_session_resolves_angle_tokens(self):
        session = Session(self.tokenizer)
        self.assertEqual(session.token_name('cot_end'), '<|thought_end|>')
        self.assertEqual(session.token_id('cot_end'), self.tokenizer.token_to_id('<|thought_end|>'))
        self.assertEqual(session.token_id('cot_start'), self.tokenizer.token_to_id('<|thought_start|>'))
        self.assertEqual(session.token_id('im_end'), self.tokenizer.token_to_id('<|im_end|>'))
        self.assertEqual(session.token_id('tool_name_divider'),
                         self.tokenizer.token_to_id('<|tool_name_divider|>'))

    def test_both_spellings_and_aliases(self):
        session = Session(self.tokenizer)
        im_end = self.tokenizer.token_to_id('<|im_end|>')
        # 逻辑名 / A1 写法 / A2 写法 都能查到同一个 token
        for name in ['im_end', '[im_end]', '<|im_end|>']:
            self.assertEqual(session.token_id(name), im_end, name)
        thought_end = self.tokenizer.token_to_id('<|thought_end|>')
        for name in ['cot_end', '[cot_end]', '<|thought_end|>', 'thought_end']:
            self.assertEqual(session.token_id(name), thought_end, name)
        # 取不到的名字返回 None，而不是抛错
        self.assertIsNone(session.token_id('[unused_42]'))
        self.assertIsNone(session.token_id('nonexistent_token'))
        # 既不是逻辑名也不是别名的字符串，退化为「词表里恰好有这个名字的 token」
        self.assertEqual(session.token_id('the'), self.tokenizer.token_to_id('the'))

    def test_module_level_helpers(self):
        self.assertEqual(token_name_table(self.tokenizer)['cot_end'], '<|thought_end|>')
        self.assertEqual(resolve_token_name(self.tokenizer, '[im_end]'), '<|im_end|>')
        self.assertEqual(resolve_token_id(self.tokenizer, 'im_end'),
                         self.tokenizer.token_to_id('<|im_end|>'))
        self.assertEqual(resolve_token_id(self.tokenizer, '<|thought_end|>'),
                         self.tokenizer.token_to_id('<|thought_end|>'))

    def test_all_specials_resolved(self):
        session = Session(self.tokenizer)
        ids = session.special_ids()
        for name in SPEC_TOKENS:
            token_id = self.tokenizer.token_to_id(name)
            self.assertIsNotNone(token_id, name)
            self.assertIn(token_id, ids, name)

    def test_no_a1_tokens_in_chord(self):
        for name in ['[im_end]', '[cot_end]', '[pad]', '[unused_42]']:
            self.assertIsNone(self.tokenizer.token_to_id(name), name)


@unittest.skipUnless(os.path.exists(CHORD_PATH), 'chord.zip not found')
class TestChordForcedJSON(unittest.TestCase):
    def setUp(self):
        self.tokenizer = load_tokenizer()
        self.vocab = TokenVocab.from_tokenizer(self.tokenizer)

    def test_specials_banned_from_constrained_vocab(self):
        for name in SPEC_TOKENS:
            token_id = self.tokenizer.token_to_id(name)
            self.assertNotIn(token_id, self.vocab.texts, name)
        # 模型词表大小 16384，可用 token 应当是绝大多数
        self.assertEqual(self.vocab.vocab_size, self.tokenizer.vocab_size)
        self.assertGreater(len(self.vocab.texts), 15000)

    def test_constraint_uses_angle_eos(self):
        eos_id = self.tokenizer.token_to_id('<|im_end|>')
        constraint = build_json_constraint(self.tokenizer, mode='object', eos_token_id=eos_id)
        self.assertEqual(constraint.eos_token_id, eos_id)
        constraint.commit_text('{"a": 1}')
        self.assertTrue(constraint.is_complete)
        token = int(Sampler(temperature=1.0, constraint=constraint)(
            torch.randn(1, self.tokenizer.vocab_size)).item())
        self.assertEqual(token, eos_id)

    def test_forced_json_walk_on_chord(self):
        '''真实 chord 词表下随机 logits 也必须产出合法 JSON。'''
        eos_id = self.tokenizer.token_to_id('<|im_end|>')
        closers = [
            token_id for token_id, text in self.vocab.texts.items()
            if text in ('"', '}', ']')
        ]
        completed = 0
        for seed in range(3):
            torch.manual_seed(seed)
            constraint = build_json_constraint(self.tokenizer, mode='object', eos_token_id=eos_id)
            sampler = Sampler(temperature=1.0, constraint=constraint)
            pieces = []
            for _ in range(400):
                logits = torch.randn(1, self.tokenizer.vocab_size)
                logits[0, eos_id] = -1e9
                logits[0, closers] += 6.0
                token = int(sampler(logits).item())
                if token == eos_id:
                    break
                pieces.append(self.vocab.texts.get(token, ''))
            text = ''.join(pieces)
            if constraint.is_complete:
                completed += 1
                self.assertIsInstance(json.loads(text), dict)
        self.assertGreater(completed, 0)


@unittest.skipUnless(os.path.exists(CHORD_PATH), 'chord.zip not found')
class TestChordChatGating(unittest.TestCase):
    '''chat() 在 chord 词表下：思考段自由、<|thought_end|> 之后才强制 JSON。'''

    def setUp(self):
        self.tokenizer = load_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        # chord 是 byte-level BPE：挑一个词表里真实存在的词 token 来当「自由生成」的内容
        self.word_id = self.tokenizer.token_to_id('the')
        self.word = self.tokenizer.decode([self.word_id], skip_special_tokens=True)

    def token(self, text):
        token_id = self.tokenizer.token_to_id(text)
        self.assertIsNotNone(token_id, text)
        return token_id

    def run_chat(self, script, max_new_tokens=32, response_format=None):
        model = _ScriptedModel(script, self.vocab_size)
        chunks = list(chat(
            model=model,
            tokenizer=self.tokenizer,
            device='cpu',
            messages=[{'role': 'user', 'content': 'hello'}],
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            response_format=response_format,
        ))
        content = ''.join(
            chunk.content for chunk in chunks if not chunk.is_cot and chunk.finish_reason is None
        )
        cot = ''.join(
            chunk.content for chunk in chunks if chunk.is_cot and chunk.finish_reason is None
        )
        return cot, content, chunks[-1].finish_reason

    def test_thought_end_triggers_constraint(self):
        '''思考段可以随便写；<|thought_end|> 之后的第一个 token 必须服从 JSON 语法。'''
        word, object_start = self.word_id, self.token('{')
        script = [
            word,                        # 思考段：自由生成
            self.token('<|thought_end|>'),
            word,                        # 答案段：非法 JSON，必须被屏蔽
            object_start,                # 于是只能从这里开始写 JSON
        ]
        cot, content, finish = self.run_chat(script, max_new_tokens=8,
                                            response_format={'type': 'json_object'})
        self.assertEqual(cot.strip(), self.word.strip())    # 思考段未被约束
        self.assertTrue(content.lstrip().startswith('{'), repr(content))
        self.assertIn(finish, ('stop', 'length'))

    def test_tool_tokens_do_not_leak_into_answer(self):
        '''工具/分隔符 token 在 JSON 模式下既不出现，也不会被当成正文输出。'''
        word = self.word_id
        script = [
            word,
            self.token('<|thought_end|>'),
            self.token('<|tool_call_start|>'),      # 约束下不可选
            self.token('<|tool_name_divider|>'),    # 约束下不可选
            self.token('{'),
            self.token('"'),
            self.token('a'),
            self.token('"'),
            self.token(':'),
            self.token('1'),
            self.token('}'),
        ]
        _, content, finish = self.run_chat(script, max_new_tokens=24,
                                           response_format={'type': 'json_object'})
        self.assertNotIn('<|', content)
        self.assertNotIn('tool_call_start', content)
        self.assertTrue(content.lstrip().startswith('{'), repr(content))
        if finish == 'stop':
            self.assertEqual(json.loads(content), {'a': 1})

    def test_im_end_is_forced_only_when_complete(self):
        '''JSON 没写完时 <|im_end|> 被屏蔽；写完以后被强制。'''
        word, im_end = self.word_id, self.token('<|im_end|>')
        script = [
            word,
            self.token('<|thought_end|>'),
            im_end,                # 想在答案开头就结束 -> 被屏蔽
            self.token('{'),
            self.token('}'),       # {} 写完了
            im_end,                # 现在被强制
        ]
        _, content, finish = self.run_chat(script, max_new_tokens=16,
                                           response_format={'type': 'json_object'})
        self.assertEqual(finish, 'stop')
        self.assertEqual(json.loads(content), {})

    def test_text_mode_is_unconstrained(self):
        '''response_format 不开启时，答案段照旧自由生成（不会被 JSON 语法拦）。'''
        word = self.word_id
        script = [word, self.token('<|thought_end|>'), word, word]
        cot, content, _ = self.run_chat(script, max_new_tokens=5)
        self.assertEqual(cot.strip(), self.word.strip())
        self.assertEqual(content.strip(), (self.word * 2).strip())

    def test_thinking_disabled_starts_from_answer(self):
        '''enable_thinking=False 时不存在思考段，约束从第一个 token 起生效。'''
        model = _ScriptedModel(
            [self.word_id, self.token('{'), self.token('}')], self.vocab_size
        )
        chunks = list(chat(
            model=model, tokenizer=self.tokenizer, device='cpu',
            messages=[{'role': 'user', 'content': 'hello'}],
            max_new_tokens=8, temperature=1.0, enable_thinking=False,
            response_format={'type': 'json_object'},
        ))
        content = ''.join(c.content for c in chunks if not c.is_cot and c.finish_reason is None)
        self.assertTrue(content.lstrip().startswith('{'), repr(content))
        self.assertEqual(chunks[-1].finish_reason, 'stop')
        self.assertEqual(json.loads(content), {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
