'''
端到端：MotifA1 + 真实 tokenizer，用 CausalLanguageModel.generate(constraint=...) 强制 JSON。

    python test/test_json_e2e.py

会从远端拉 MotifA1-SFT 的权重与词表（首次运行较慢）。
'''

import os
import sys
import json
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch

from codon.motif import MotifA1, MotifA1Tokenizer
from codon.utils.session import Session
from codon.model.grammar import build_json_constraint, constraint_from_response_format
from codon.model.sampler import Sampler
from codon.utils.generate import chat


SCHEMA = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string'},
        'age': {'type': 'integer'},
        'tags': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['name', 'age'],
    'additionalProperties': False,
}


def build_prompt(tokenizer, instruction, disable_thinking=True):
    session = Session(tokenizer)
    session.add_message({'role': 'system', 'content': 'You reply with JSON only.'})
    session.add_message({'role': 'user', 'content': instruction})
    session.add_generation_prompt(disable_thinking=disable_thinking)
    return session


def main() -> int:
    print('[*] loading MotifA1 + tokenizer from remote ...')
    tokenizer = MotifA1Tokenizer().from_remote()
    model = MotifA1().from_remote()
    model.eval()

    device = next(model.parameters()).device
    im_end = tokenizer.token_to_id('[im_end]')
    prompt_len = None

    # ---- 1) 无 schema：只要求是合法 JSON ----
    for mode, schema, instruction in [
        ('object', None, 'Give me a JSON object with a name and an age.'),
        (None, SCHEMA, 'Extract: Aiden is 30 years old and likes python, json.'),
    ]:
        session = build_prompt(tokenizer, instruction)
        ids = torch.tensor([session.input_ids], dtype=torch.long, device=device)
        prompt_len = ids.shape[1]
        constraint = build_json_constraint(tokenizer, schema=schema, mode=mode, eos_token_id=im_end)
        start = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                ids, max_new_tokens=256, temperature=0.5,
                sampler=Sampler(temperature=0.5),
                constraint=constraint, eos_token_id=im_end,
            )
        elapsed = time.perf_counter() - start
        new_ids = generated[0, prompt_len:].tolist()
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        print(f'[*] generate(mode={mode}, schema={schema is not None}) {elapsed:.1f}s -> {text!r}')
        value = json.loads(text)
        assert constraint.is_complete, 'constraint should have finished the document'
        if schema is not None:
            assert set(value) >= {'name', 'age'}, value

    # ---- 2) 自由采样对照：同样 prompt 下不约束时未必是合法 JSON ----
    session = build_prompt(tokenizer, 'Give me a JSON object with a name and an age.')
    ids = torch.tensor([session.input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        free = model.generate(ids, max_new_tokens=64, temperature=0.5, eos_token_id=im_end)
    free_text = tokenizer.decode(free[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
    print(f'[*] unconstrained -> {free_text!r}')

    # ---- 3) chat() 的 response_format 通路（答案段强制 + schema 校验） ----
    def run_chat(response_format, max_new_tokens, content='Extract: Bob is 41 and likes rust.'):
        chunks = list(chat(
            model=model, tokenizer=tokenizer, device=device,
            messages=[{'role': 'user', 'content': content}],
            max_new_tokens=max_new_tokens, temperature=0.5,
            response_format=response_format,
        ))
        text = ''.join(chunk.content for chunk in chunks if not chunk.is_cot)
        return text, chunks[-1].finish_reason

    text, reason = run_chat(
        {'type': 'json_schema', 'json_schema': {'schema': SCHEMA}}, max_new_tokens=512
    )
    print(f'[*] chat(response_format=json_schema) -> finish={reason} {text!r}')
    value = json.loads(text)
    assert set(value) >= {'name', 'age'}, value
    assert reason == 'stop', 'JSON 完整时应当报 stop'

    # 预算不够时：JSON 不完整，但结束原因必须是 length（调用方据此加大预算重试）
    text, reason = run_chat({'type': 'json_object'}, max_new_tokens=8)
    print(f'[*] chat(json_object, max_new_tokens=8) -> finish={reason} {text!r}')
    self_check = reason == 'length'
    assert self_check, f'预算用尽应当报 length，实际 {reason}'

    # text 模式：不做约束
    text_mode, reason = run_chat({'type': 'text'}, max_new_tokens=16, content='hi')
    print(f'[*] chat(response_format=text) -> finish={reason} {text_mode!r}')
    assert reason in ('stop', 'length')

    print('[*] e2e forced-JSON test passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
