'''
真实词表上的强制 JSON 采样冒烟测试（需要联网，从 ModelScope / HuggingFace 拉 motif.vocab）。

    python test/test_json_real_vocab.py

检查三件事：
1. 真实词表的提取/过滤是否正确（无替换字符、无特殊 token）；
2. 剪枝结果与「逐 token 全量校验」完全一致；
3. 随机 logits 下仍能产出合法 JSON（并统计每一步筛选耗时）。
'''

import os
import sys
import json
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch

from codon.model.grammar import (
    JSONConstraint,
    TokenVocab,
    build_token_vocab,
    build_json_constraint,
    constraint_from_response_format,
)
from codon.model.sampler import Sampler


SCHEMA = {
    'type': 'object',
    'properties': {
        'title': {'type': 'string', 'maxLength': 32},
        'score': {'type': 'number'},
        'tags': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 4},
        'meta': {
            'type': 'object',
            'properties': {'ok': {'type': 'boolean'}, 'level': {'enum': ['low', 'high']}},
            'required': ['ok'],
            'additionalProperties': False,
        },
    },
    'required': ['title', 'meta'],
    'additionalProperties': False,
}


def brute_force_allowed_ids(vocab: TokenVocab, grammar, limit=None):
    out = set()
    for token_id, text in vocab.texts.items():
        if limit is not None and token_id >= limit:
            continue
        probe = grammar.clone(probe=True)
        if probe.feed_text(text):
            out.add(token_id)
    return out


def main() -> int:
    from codon.motif import MotifA1Tokenizer

    print('[*] loading tokenizer from remote ...')
    tokenizer = MotifA1Tokenizer().from_remote()
    print(f'[*] vocab_size = {tokenizer.vocab_size}')

    start = time.perf_counter()
    vocab = build_token_vocab(tokenizer)
    build_seconds = time.perf_counter() - start
    print(
        f'[*] vocab table built in {build_seconds:.2f}s: '
        f'{len(vocab.texts)} usable tokens '
        f'({len(vocab.plain_ids)} plain / {len(vocab.restricted)} restricted), '
        f'{len(vocab.first_chars)} distinct first chars'
    )

    bad = [tid for tid, text in vocab.texts.items() if not text or '\ufffd' in text]
    assert not bad, f'vocabulary still contains unusable tokens: {bad[:5]}'
    escape_id = tokenizer.token_to_id(tokenizer.safe_escape)
    assert escape_id not in vocab.texts, 'safe escape token must be filtered out'
    print('[*] filtering check: ok')

    # ---- 剪枝 vs 逐 token 全量校验 ----
    eos_id = tokenizer.token_to_id('[im_end]')
    for spec, texts in [
        (constraint_from_response_format(tokenizer, {'type': 'json_object'}, eos_token_id=eos_id),
         ['', '{', '{"a"', '{"a": "x', '{"a": [1, 2, {"b": ']),
        (build_json_constraint(tokenizer, schema=SCHEMA, eos_token_id=eos_id),
         ['{"title": "he', '{"title": "x", "meta": {', '{"title": "x", "meta": {"ok": tr']),
    ]:
        for text in texts:
            constraint = spec
            constraint.reset(1)
            if text:
                constraint.commit_text(text)
            grammar = constraint.grammars[0]
            allowed = set(constraint.allowed_token_ids(0))
            expected = brute_force_allowed_ids(vocab, grammar)
            assert allowed == expected, (
                f'{text!r}: missing={sorted(expected - allowed)[:5]} extra={sorted(allowed - expected)[:5]}'
            )
    print('[*] exactness check: ok (pruned set == full scan)')

    # ---- 随机 logits 下的强制 JSON + 耗时 ----
    # 真实词表里 99% 的 token 都是「纯文字」，纯随机游走会一直写在字符串里不闭合，
    # 所以这里给收尾字符（" } ]）加一点偏置，让随机游走能走完整个语法。
    closers = [
        token_id for token_id, text in vocab.texts.items()
        if text in ('"', '}', ']')
    ]

    for name, mode, schema in [
        ('json_object', 'object', None),
        ('json_schema', None, SCHEMA),
    ]:
        completed_runs = 0
        for seed in range(4):
            torch.manual_seed(seed)
            constraint = build_json_constraint(
                tokenizer, schema=schema, mode=mode, eos_token_id=eos_id
            )
            sampler = Sampler(temperature=1.0, constraint=constraint)
            pieces, total_seconds, worst_seconds, steps = [], 0.0, 0.0, 0
            for _ in range(600):
                t0 = time.perf_counter()
                logits = torch.randn(1, tokenizer.vocab_size)
                logits[0, eos_id] = -1e9
                logits[0, closers] += 6.0
                token = int(sampler(logits).item())
                elapsed = time.perf_counter() - t0
                total_seconds += elapsed
                worst_seconds = max(worst_seconds, elapsed)
                steps += 1
                if token == eos_id:
                    break
                pieces.append(vocab.texts.get(token, ''))

            text = ''.join(pieces)
            if constraint.is_complete:
                completed_runs += 1
                value = json.loads(text)               # 必须能解析
                if schema is not None:
                    assert isinstance(value, dict) and 'title' in value and 'meta' in value, value
                    assert value['meta'].get('ok') in (True, False), value
                else:
                    assert isinstance(value, dict)
            print(
                f'[*] {name} seed={seed}: complete={constraint.is_complete} steps={steps} '
                f'avg={total_seconds / steps * 1000:.1f}ms worst={worst_seconds * 1000:.1f}ms '
                f'chars={len(text)}'
            )
        assert completed_runs > 0, f'{name}: no run produced a complete JSON document'

    print('[*] real-vocab smoke test passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
