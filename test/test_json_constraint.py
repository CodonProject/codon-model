import os
import sys
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import unittest

import torch

from codon.model.grammar import (
    JSONConstraint,
    JSONConstraintError,
    JSONGrammar,
    TokenVocab,
    UnsupportedSchemaError,
    build_json_constraint,
    build_token_vocab,
    constraint_from_response_format,
    normalize_schema,
    parse_response_format,
)
from codon.model.sampler import Sampler


# ---------------------------------------------------------------------------
# 一个最小可用的 tokenizer 替身（duck typing），避免测试依赖真实词表文件
# ---------------------------------------------------------------------------

class _StubFastTokenizer:
    def __init__(self, special_ids):
        self.all_special_ids = list(special_ids)

    def get_added_vocab(self):
        return {}


class _StubTokenizer:
    def __init__(self, texts, special_ids=(), safe_escape='<|safe_escape|>', vocab_size=None):
        self._texts = dict(texts)
        self.safe_escape = safe_escape
        self.safe_escape_id = None
        self.fast_tokenizer = _StubFastTokenizer(special_ids)
        self.vocab_size = int(vocab_size) if vocab_size else max(self._texts) + 1

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(self._texts.get(int(i), '') for i in ids)

    def token_to_id(self, token):
        for token_id, text in self._texts.items():
            if text == token:
                return token_id
        return None


_BASE_TOKENS = (
    list('{}[]",:.eE+-0123456789 \n\t\\/bfnrtu')
    + list('abcdefghijklmnopqrstuvwxyz')
    + ['é', '中']
)
_MULTI_TOKENS = [
    '{"', '": ', '"a"', ', "', 'true', 'false', 'null', '[]', '{}', '123',
    '"name"', '"age"', '"kind"', '"tags"', '"speech"', '"music"', '"x"',
    '": "', '", "', '\n  ', '  ', '<|safe_escape|>', '\ufffd',
]

#: 词表按顺序编号；特殊 token 与坏 token 放在里面用于验证过滤逻辑
_TOKENS = list(dict.fromkeys(_BASE_TOKENS + _MULTI_TOKENS))
TEXT_TO_ID = {text: index for index, text in enumerate(_TOKENS)}
BAD_MULTIBYTE_ID = TEXT_TO_ID['\ufffd']
ESCAPE_ID = TEXT_TO_ID['<|safe_escape|>']
EOS_ID = len(_TOKENS)
VOCAB_SIZE = EOS_ID + 1

#: 纯空白 token：随机游走时压低它们的概率，否则容易长时间停在空白上（不是语法问题）
WHITESPACE_IDS = tuple(
    token_id for token_id, text in enumerate(_TOKENS) if text and text.strip(' \t\n\r') == ''
)


def random_logits() -> torch.Tensor:
    logits = torch.randn(1, VOCAB_SIZE)
    for token_id in WHITESPACE_IDS:
        logits[0, token_id] = -1e9
    return logits


def make_tokenizer(**kwargs):
    texts = {index: text for index, text in enumerate(_TOKENS)}
    return _StubTokenizer(texts, special_ids=[EOS_ID], vocab_size=VOCAB_SIZE, **kwargs)


def make_vocab():
    return TokenVocab.from_tokenizer(make_tokenizer())


def brute_force_allowed_ids(vocab, grammar):
    '''每个 token 都真跑一遍 FSM（不做任何剪枝），作为 allowed_ids 的正确性参照。'''
    out = set()
    for token_id, text in vocab.texts.items():
        probe = grammar.clone(probe=True)
        if probe.feed_text(text):
            out.add(token_id)
    return out


# ---------------------------------------------------------------------------
# 测试用的 schema 与极简 schema 校验器（不依赖 jsonschema 包）
# ---------------------------------------------------------------------------

OBJECT_SCHEMA = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string', 'minLength': 1, 'maxLength': 8},
        'age': {'type': 'integer'},
        'kind': {'enum': ['speech', 'music']},
        'tags': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 2},
        'meta': {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}},
    },
    'required': ['name', 'kind'],
    'additionalProperties': False,
}


def check_schema(value, schema, path='$'):
    if 'const' in schema:
        assert value == schema['const'], f'{path}: expected const {schema["const"]!r}, got {value!r}'
    if 'enum' in schema:
        assert value in schema['enum'], f'{path}: {value!r} not in enum {schema["enum"]!r}'

    type_name = schema.get('type')
    if type_name == 'object':
        assert isinstance(value, dict), f'{path}: expected object, got {type(value).__name__}'
        props = schema.get('properties') or {}
        for key in schema.get('required', []):
            assert key in value, f'{path}: missing required key {key!r}'
        extra = schema.get('additionalProperties', True)
        for key, item in value.items():
            if key in props:
                check_schema(item, props[key], f'{path}.{key}')
            elif extra is False:
                raise AssertionError(f'{path}: unexpected extra key {key!r}')
            elif isinstance(extra, dict):
                check_schema(item, extra, f'{path}.{key}')
    elif type_name == 'array':
        assert isinstance(value, list), f'{path}: expected array, got {type(value).__name__}'
        if 'minItems' in schema:
            assert len(value) >= schema['minItems'], f'{path}: too few items'
        if 'maxItems' in schema:
            assert len(value) <= schema['maxItems'], f'{path}: too many items'
        items = schema.get('items')
        if items:
            for index, item in enumerate(value):
                check_schema(item, items, f'{path}[{index}]')
    elif type_name == 'string':
        assert isinstance(value, str), f'{path}: expected string, got {type(value).__name__}'
        if 'minLength' in schema:
            assert len(value) >= schema['minLength'], f'{path}: too short'
        if 'maxLength' in schema:
            assert len(value) <= schema['maxLength'], f'{path}: too long'
    elif type_name == 'integer':
        assert isinstance(value, int) and not isinstance(value, bool), f'{path}: expected integer'
    elif type_name == 'number':
        assert isinstance(value, (int, float)) and not isinstance(value, bool), f'{path}: expected number'
    elif type_name == 'boolean':
        assert isinstance(value, bool), f'{path}: expected boolean'
    elif type_name == 'null':
        assert value is None, f'{path}: expected null'


class TestJSONGrammar(unittest.TestCase):
    def test_accepts_valid_prefixes(self):
        for text in [
            '{', '{"a"', '{"a":', '{"a":1', '{"a":1}', '  {  }  ',
            '[', '[1', '[1,', '[1,2]', '[[[]]]', '[{"a":[1,true,null]}]',
            '"', '"abc', '"a\\', '"a\\n', '"a\\u00', '"a\\u0041"',
            't', 'tr', 'tru', 'true', 'false', 'null',
            '-', '-0', '-12', '0', '1', '12', '1.', '1.5', '1e', '1e-', '1e+10', '0.0',
        ]:
            grammar = JSONGrammar()
            self.assertTrue(grammar.feed_text(text), f'{text!r} should be a valid JSON prefix')

    def test_rejects_invalid(self):
        for text in [
            'x', '{1', '{"a" 1', '{"a":1,}', '[1,]', '[,1]', '[1 2]', '{,}',
            '"a\nb"', '"a\\x"', '"a\\u12"', '01', '1..2', '.5', '+1', '--1',
            'trux', 'nulll', '\u4e2d', 'True', 'NULL', '}', ']', ':', ',',
        ]:
            grammar = JSONGrammar()
            self.assertFalse(grammar.feed_text(text), f'{text!r} should not be a valid JSON prefix')

        # 被拒的字符会回滚，语法仍然停在最后一个合法前缀上
        grammar = JSONGrammar()
        self.assertTrue(grammar.feed_text('{"a": 1'))
        self.assertFalse(grammar.step('x'))
        self.assertTrue(grammar.step('}'))
        self.assertTrue(grammar.is_complete)

    def test_completeness(self):
        complete = ['{}', '[]', '{"a":1}', '[1,2,3]', '123', '-1.5e10', '0', '"x"', 'true', 'false', 'null', '{"a":{"b":[]}}']
        incomplete = ['', '{', '[', '{"a"', '{"a":1', '[1,', '"x', 'tru', '1.', '1e', '  ']
        for text in complete:
            grammar = JSONGrammar()
            self.assertTrue(grammar.feed_text(text), text)
            self.assertTrue(grammar.is_complete, f'{text!r} should be complete')
        for text in incomplete:
            grammar = JSONGrammar()
            self.assertTrue(grammar.feed_text(text), text)
            self.assertFalse(grammar.is_complete, f'{text!r} should not be complete')

    def test_text_and_reset(self):
        grammar = JSONGrammar()
        grammar.feed_text('{"a": 1}')
        self.assertEqual(grammar.document, '{"a": 1}')
        grammar.reset()
        self.assertEqual(grammar.document, '')
        self.assertFalse(grammar.is_complete)

    def test_clone_is_independent(self):
        grammar = JSONGrammar()
        grammar.feed_text('{"a"')
        clone = grammar.clone()
        self.assertTrue(clone.feed_text(':1}'))
        self.assertTrue(clone.is_complete)
        self.assertFalse(grammar.is_complete)
        self.assertEqual(grammar.document, '{"a"')

    def test_after_complete_only_whitespace(self):
        grammar = JSONGrammar()
        grammar.feed_text('{"a": 1}')
        self.assertTrue(grammar.feed_text(' \n\t'))
        self.assertFalse(grammar.step('x'))


class TestJSONGrammarSchema(unittest.TestCase):
    def make(self, schema):
        return JSONGrammar(schema)

    def test_strict_object_keys_and_required(self):
        grammar = self.make(OBJECT_SCHEMA)
        self.assertTrue(grammar.feed_text('{"name": "a"'))
        self.assertFalse(grammar.step('}'))            # 还缺 required 的 kind（状态已回滚）
        self.assertTrue(grammar.feed_text(', "kind": "speech"'))
        self.assertTrue(grammar.step('}'))
        self.assertTrue(grammar.is_complete)
        self.assertEqual(json.loads(grammar.document)['kind'], 'speech')

    def test_unknown_key_rejected(self):
        grammar = self.make(OBJECT_SCHEMA)
        self.assertTrue(grammar.step('{'))
        self.assertTrue(grammar.step('"'))
        self.assertTrue(grammar.step('n'))             # 'name' 的合法前缀
        self.assertFalse(grammar.step('o'))            # 但没有键名以 'no' 开头（状态回滚）
        self.assertTrue(grammar.feed_text('ame": "a"'))
        self.assertEqual(grammar.document, '{"name": "a"')

    def test_escaped_key_in_strict_object_is_rejected(self):
        # 严格对象（additionalProperties=false）的键名只接受字面量写法
        grammar = self.make(OBJECT_SCHEMA)
        self.assertTrue(grammar.step('{'))
        self.assertFalse(grammar.feed_text('"\\u006eame"'))
        self.assertTrue(grammar.feed_text('name": "a"'))   # 状态已回滚到键名内部，可继续写
        self.assertEqual(grammar.document, '{"name": "a"')

        # 自由键名的对象允许转义，且不会被误判成 properties 里出现过的键
        free = self.make({'type': 'object', 'properties': {'name': {'type': 'string'}}})
        self.assertTrue(free.step('{'))
        self.assertTrue(free.feed_text('"\\u006eame": "a"'))
        self.assertTrue(free.step('}'))
        self.assertTrue(free.is_complete)
        self.assertEqual(json.loads(free.document), {'name': 'a'})

    def test_integer_type_forbids_float(self):
        grammar = self.make(OBJECT_SCHEMA)
        self.assertTrue(grammar.feed_text('{"age": 1'))
        self.assertFalse(grammar.step('.'))
        self.assertTrue(grammar.feed_text(', "name": "a", "kind": "music"}'))
        self.assertTrue(grammar.is_complete)

        grammar2 = self.make({'type': 'integer'})
        self.assertTrue(grammar2.feed_text('12'))
        self.assertTrue(grammar2.is_complete)
        self.assertFalse(grammar2.step('.'))           # 已完整，只允许空白

    def test_enum_values(self):
        grammar = self.make({'enum': ['speech', 'music']})
        self.assertTrue(grammar.feed_text('"mu'))
        self.assertTrue(grammar.feed_text('sic"'))
        self.assertTrue(grammar.is_complete)
        bad = self.make({'enum': ['speech']})
        self.assertFalse(bad.feed_text('"musi'))
        good = self.make({'enum': ['speech']})
        self.assertTrue(good.feed_text('"spe'))
        self.assertTrue(good.feed_text('ech"'))
        self.assertTrue(good.is_complete)

    def test_enum_of_numbers_and_objects(self):
        grammar = self.make({'enum': [1, 2]})
        self.assertTrue(grammar.feed_text('1'))
        self.assertTrue(grammar.is_complete)
        bad = self.make({'enum': [1, 2]})
        self.assertFalse(bad.step('3'))

        obj = self.make({'const': {'a': [1]}})
        self.assertTrue(obj.feed_text('{"a": [1]}'))
        self.assertTrue(obj.is_complete)
        bad_obj = self.make({'const': {'a': [1]}})
        self.assertFalse(bad_obj.feed_text('{"a": [2]}'))

    def test_array_bounds(self):
        schema = {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 2, 'maxItems': 2}
        self.assertFalse(self.make(schema).step(']'))   # 顶层不是数组
        empty_ok = self.make({'type': 'array', 'items': {'type': 'integer'}})
        self.assertTrue(empty_ok.feed_text('[]'))       # 没有 minItems 时空数组合法
        self.assertTrue(empty_ok.is_complete)

        grammar = self.make(schema)
        self.assertTrue(grammar.feed_text('[1'))
        self.assertFalse(grammar.step(']'))            # 还差一个
        self.assertTrue(grammar.feed_text(', 2'))
        self.assertFalse(grammar.step(','))            # 已经到 maxItems
        self.assertTrue(grammar.step(']'))
        self.assertTrue(grammar.is_complete)

    def test_string_length_bounds(self):
        grammar = self.make({'type': 'string', 'minLength': 2, 'maxLength': 3})
        self.assertTrue(grammar.feed_text('"a'))
        self.assertFalse(grammar.step('"'))            # 太短（状态回滚）
        self.assertTrue(grammar.feed_text('bc'))       # 正好到 maxLength
        self.assertFalse(grammar.step('d'))            # 超长
        self.assertTrue(grammar.step('"'))
        self.assertTrue(grammar.is_complete)

    def test_type_restriction(self):
        grammar = self.make({'type': 'object'})
        self.assertTrue(grammar.step('{'))
        self.assertFalse(grammar.step('['))

        boolean = self.make({'type': 'boolean'})
        self.assertFalse(boolean.step('1'))
        self.assertTrue(boolean.step('t'))
        self.assertTrue(boolean.feed_text('rue'))
        self.assertTrue(boolean.is_complete)

        null = self.make({'type': 'null'})
        self.assertTrue(null.feed_text('null'))
        self.assertTrue(null.is_complete)
        self.assertFalse(self.make({'type': 'string'}).step('1'))

    def test_escape_and_unicode_in_string(self):
        grammar = self.make({'type': 'string'})
        self.assertTrue(grammar.feed_text('"a\\n\\u0041"'))
        self.assertTrue(grammar.is_complete)

    def test_unsupported_keywords_raise(self):
        for schema in [
            {'anyOf': [{'type': 'string'}]},
            {'oneOf': [{'type': 'string'}]},
            {'allOf': [{'type': 'string'}]},
            {'$ref': '#/definitions/x'},
            {'type': 'string', 'pattern': '^a'},
            {'type': 'number', 'minimum': 1},
            {'type': 'array', 'items': [{'type': 'string'}]},
            {'type': ['string', 'null']},
        ]:
            with self.assertRaises(UnsupportedSchemaError):
                normalize_schema(schema)

    def test_metadata_keywords_ignored(self):
        schema = normalize_schema({'type': 'string', 'title': 'x', 'description': 'y', 'format': 'date'})
        self.assertEqual(schema, {'type': 'string'})

    def test_unsatisfiable_schema_raises(self):
        with self.assertRaises(ValueError):
            normalize_schema({
                'type': 'object', 'properties': {'a': {'type': 'string'}},
                'required': ['b'], 'additionalProperties': False,
            })
        with self.assertRaises(ValueError):
            normalize_schema({'type': 'string', 'minLength': 5, 'maxLength': 2})
        with self.assertRaises(ValueError):
            normalize_schema({'enum': []})
        with self.assertRaises(ValueError):
            normalize_schema({'const': float('nan')})

    def test_bad_schema_types(self):
        with self.assertRaises(TypeError):
            normalize_schema('not a schema')
        with self.assertRaises(TypeError):
            normalize_schema({'type': 'object', 'properties': []})
        with self.assertRaises(ValueError):
            normalize_schema({'type': 'nope'})


class TestTokenVocab(unittest.TestCase):
    def test_bad_and_special_tokens_filtered(self):
        vocab = make_vocab()
        self.assertNotIn(BAD_MULTIBYTE_ID, vocab.texts)      # 多字节碎片（\ufffd）
        self.assertNotIn(ESCAPE_ID, vocab.texts)             # safe_escape
        self.assertNotIn(EOS_ID, vocab.texts)
        self.assertEqual(vocab.vocab_size, VOCAB_SIZE)
        self.assertEqual(vocab.texts[TEXT_TO_ID['{']], '{')

    def test_vocab_cached_on_tokenizer(self):
        tokenizer = make_tokenizer()
        first = build_token_vocab(tokenizer)
        second = build_token_vocab(tokenizer)
        self.assertIs(first, second)

    def test_allowed_ids_at_document_start(self):
        vocab = make_vocab()
        constraint = JSONConstraint(vocab, mode='object', eos_token_id=EOS_ID)
        allowed = set(constraint.allowed_token_ids(0))
        self.assertIn(TEXT_TO_ID['{'], allowed)
        self.assertIn(TEXT_TO_ID[' '], allowed)
        self.assertIn(TEXT_TO_ID['{"'], allowed)
        self.assertNotIn(TEXT_TO_ID['}'], allowed)           # 还不能直接闭合
        self.assertNotIn(TEXT_TO_ID['"a"'], allowed)         # 值只能是对象
        self.assertNotIn(EOS_ID, allowed)

    def test_string_state_is_fast_path(self):
        vocab = make_vocab()
        constraint = JSONConstraint(vocab, mode='object', eos_token_id=EOS_ID)
        constraint.commit_text('{"name": "')
        grammar = constraint.grammars[0]
        self.assertTrue(grammar.absorbs_plain_text())
        allowed = set(constraint.allowed_token_ids(0))
        self.assertTrue(set(vocab.plain_ids) <= allowed)     # 纯文字 token 全部合法
        self.assertIn(TEXT_TO_ID['"'], allowed)              # 可以闭合成字符串
        self.assertNotIn(TEXT_TO_ID['\n'], allowed)          # 字符串里不允许裸控制字符
        self.assertEqual(allowed, brute_force_allowed_ids(vocab, grammar))


class TestJSONConstraint(unittest.TestCase):
    def test_force_eos_and_block_eos(self):
        vocab = make_vocab()
        constraint = JSONConstraint(vocab, mode='object', eos_token_id=EOS_ID)
        sampler = Sampler(temperature=1.0, constraint=constraint)

        # 文档没写完时，即使 eos 的 logits 最高也不能选它
        logits = torch.full((1, VOCAB_SIZE), -10.0)
        logits[0, EOS_ID] = 100.0
        token = int(sampler(logits).item())
        self.assertNotEqual(token, EOS_ID)

        # 写完之后，马上强制 eos
        constraint.reset(1)
        constraint.commit_text('{"a": 1}')
        self.assertTrue(constraint.is_complete)
        token = int(sampler(random_logits()).item())
        self.assertEqual(token, EOS_ID)

    def test_advance_rejects_illegal_token(self):
        vocab = make_vocab()
        constraint = JSONConstraint(vocab, mode='object', eos_token_id=EOS_ID)
        with self.assertRaises(JSONConstraintError):
            constraint.advance(torch.tensor([[TEXT_TO_ID[']']]]))

    def test_allowed_ids_are_respected(self):
        vocab = make_vocab()
        constraint = JSONConstraint(vocab, mode='object', eos_token_id=EOS_ID)
        sampler = Sampler(temperature=1.0, constraint=constraint)
        for seed in range(5):
            torch.manual_seed(seed)
            constraint.reset(1)
            for _ in range(30):
                if constraint.is_complete:
                    break
                allowed = set(constraint.allowed_token_ids(0))
                token = int(sampler(random_logits()).item())
                self.assertIn(token, allowed)

    def test_batch_rows_are_independent(self):
        vocab = make_vocab()
        constraint = JSONConstraint(vocab, mode='object', eos_token_id=EOS_ID, batch_size=2)
        sampler = Sampler(temperature=1.0, constraint=constraint)
        constraint.commit_text('{"a": 1}', row=0)
        logits = torch.randn(2, VOCAB_SIZE)
        tokens = sampler(logits)
        self.assertEqual(int(tokens[0].item()), EOS_ID)
        self.assertNotEqual(int(tokens[1].item()), EOS_ID)

    def test_sampler_with_constraint_does_not_mutate(self):
        vocab = make_vocab()
        constraint = JSONConstraint(vocab, mode='object', eos_token_id=EOS_ID)
        plain = Sampler(temperature=1.0)
        bound = plain.with_constraint(constraint)
        self.assertIsNone(plain.constraint)
        self.assertIs(bound.constraint, constraint)
        self.assertIsNot(plain, bound)

    def test_build_json_constraint(self):
        tokenizer = make_tokenizer()
        constraint = build_json_constraint(tokenizer, schema=OBJECT_SCHEMA, eos_token_id=EOS_ID)
        self.assertIsInstance(constraint, JSONConstraint)
        self.assertEqual(constraint.root_schema['type'], 'object')
        self.assertIsNone(constraint_from_response_format(tokenizer, {'type': 'text'}))


class TestAllowedIdsExactness(unittest.TestCase):
    '''剪枝/快路径必须与「逐 token 全量校验」完全一致（否则会悄悄漏掉合法 token）。'''

    def walk_and_compare(self, vocab, constraint, seed, steps=40):
        torch.manual_seed(seed)
        sampler = Sampler(temperature=1.0, constraint=constraint)
        for step in range(steps):
            grammar = constraint.grammars[0]
            allowed = set(constraint.allowed_token_ids(0))
            expected = brute_force_allowed_ids(vocab, grammar)
            self.assertEqual(
                allowed, expected,
                f'seed={seed} step={step} state={grammar.state} '
                f'doc={grammar.document!r} missing={sorted(expected - allowed)} '
                f'extra={sorted(allowed - expected)}',
            )
            token = int(sampler(random_logits()).item())
            if token == EOS_ID:
                break

    def test_exactness_any_json(self):
        vocab = make_vocab()
        for seed in range(6):
            constraint = JSONConstraint(vocab, mode='json', eos_token_id=EOS_ID)
            self.walk_and_compare(vocab, constraint, seed)

    def test_exactness_object(self):
        vocab = make_vocab()
        for seed in range(6):
            constraint = JSONConstraint(vocab, mode='object', eos_token_id=EOS_ID)
            self.walk_and_compare(vocab, constraint, seed)

    def test_exactness_schema(self):
        vocab = make_vocab()
        for seed in range(8):
            constraint = JSONConstraint(vocab, schema=OBJECT_SCHEMA, eos_token_id=EOS_ID)
            self.walk_and_compare(vocab, constraint, seed, steps=60)

    def test_exactness_handcrafted_states(self):
        vocab = make_vocab()
        for schema, texts in [
            (None, ['{"a": [1, {"b": "x', '"a\\', '[[[', '-1.5e', '{"k": tr']),
            (OBJECT_SCHEMA, ['{"name": "', '{"name": "a", "tags": [', '{"kind": "sp']),
            ({'enum': ['speech', 'music']}, ['"sp', '"musi']),
            ({'type': 'array', 'items': {'type': 'integer'}}, ['[1, 2, 3']),
        ]:
            for text in texts:
                grammar = JSONGrammar(schema)
                self.assertTrue(grammar.feed_text(text), text)
                allowed = set(vocab.allowed_ids(grammar))
                self.assertEqual(allowed, brute_force_allowed_ids(vocab, grammar), text)


class TestForcedJSONFuzz(unittest.TestCase):
    '''随机 logits 下也必须产出可解析、且符合 schema 的 JSON。'''

    def run_fuzz(self, schema=None, mode=None, seeds=8, max_steps=512):
        vocab = make_vocab()
        for seed in range(seeds):
            torch.manual_seed(seed)
            constraint = JSONConstraint(vocab, schema=schema, mode=mode, eos_token_id=EOS_ID)
            sampler = Sampler(temperature=1.0, constraint=constraint)
            pieces = []
            for _ in range(max_steps):
                token = int(sampler(random_logits()).item())
                if token == EOS_ID:
                    break
                pieces.append(vocab.texts.get(token, ''))
            text = ''.join(pieces)
            self.assertTrue(constraint.is_complete, f'seed={seed}: {text!r} never completed')
            value = json.loads(text)                     # 必须能解析
            if schema is not None:
                check_schema(value, normalize_schema(schema))
            if mode == 'object':
                self.assertIsInstance(value, dict)
            yield seed, value

    def test_fuzz_any_json(self):
        list(self.run_fuzz(schema=None, mode='json', seeds=10))

    def test_fuzz_object(self):
        list(self.run_fuzz(schema=None, mode='object', seeds=10))

    def test_fuzz_schema(self):
        results = list(self.run_fuzz(schema=OBJECT_SCHEMA, seeds=12))
        kinds = {value['kind'] for _, value in results}
        self.assertTrue(kinds)                           # 至少跑出若干结果

    def test_fuzz_backtracks_into_nested(self):
        schema = {
            'type': 'object',
            'properties': {
                'a': {'type': 'array', 'items': {'type': 'object', 'properties': {'b': {'const': 1}}}},
            },
            'additionalProperties': False,
        }
        for _, value in self.run_fuzz(schema=schema, seeds=6):
            for item in value.get('a', []):
                self.assertIn(item, ({}, {'b': 1}))    # properties 里没有 required，空对象也合法


class TestResponseFormat(unittest.TestCase):
    def test_parse_variants(self):
        self.assertIsNone(parse_response_format(None))
        self.assertIsNone(parse_response_format({'type': 'text'}))
        self.assertEqual(parse_response_format({'type': 'json_object'}).mode, 'object')
        self.assertEqual(parse_response_format('json_array').mode, 'array')
        self.assertEqual(parse_response_format({'type': 'json'}).mode, 'json')

        spec = parse_response_format({
            'type': 'json_schema',
            'json_schema': {'name': 'x', 'strict': True, 'schema': OBJECT_SCHEMA},
        })
        self.assertEqual(spec.schema['type'], 'object')
        self.assertIn('properties', spec.schema)

        spec = parse_response_format({'type': 'json_object', 'schema': {'type': 'object'}})
        self.assertEqual(spec.schema['type'], 'object')

    def test_parse_errors(self):
        with self.assertRaises(ValueError):
            parse_response_format({'type': 'yaml'})
        with self.assertRaises(ValueError):
            parse_response_format({'type': 'json_schema'})
        with self.assertRaises(TypeError):
            parse_response_format(123)
        with self.assertRaises(UnsupportedSchemaError):
            parse_response_format({'type': 'json_schema', 'json_schema': {'schema': {'anyOf': []}}})


if __name__ == '__main__':
    unittest.main(verbosity=2)
