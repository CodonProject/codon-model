import gzip
import json
import os
import random
import re
from glob import glob
from typing import Any, Optional, Sequence

import torch

from codon.utils.data.base import CodonDataset
from codon.utils.session import Session
from codon.utils.tokens import PackedTokenizer


_DEFAULT_SYSTEM_PROMPTS = [
    'You are a helpful assistant.',
    'You are a AI.',
    'Answer the user concisely and accurately.',
]

# 正则 / 常量（参考 convert_sft 的多格式解析经验）
_THINK_RE = re.compile(r'<think>(.*?)</think>', re.S)
_DEDICATED_COT_KEYS = ('reasoning_content', 'reasoning', 'hidden', 'thinking',
                       'chain_of_thought', 'cot', 'rationale')
_TEXT_BLOCK_TYPES = ('text', 'output_text', 'input_text', 'refusal', 'comment')
_THINK_BLOCK_TYPES = ('thinking', 'reasoning', 'reasoning_content', 'analysis')
_TOOL_BLOCK_TYPES = ('toolCall', 'tool_use', 'tool_use_block', 'tool_call')
_TOOL_RESULT_TYPES = ('tool_result', 'toolResult', 'tool_response')
_SKIP_BLOCK_TYPES = ('image', 'input_image', 'image_url', 'file')

# 原始消息 role -> 内部规范 role（human / model / tool_response）
_SOURCE_ROLE_MAP = {
    'user': 'human', 'human': 'human',
    'assistant': 'model', 'model': 'model',
    'tool': 'tool_response', 'tool_response': 'tool_response',
    'tool_result': 'tool_response', 'toolResult': 'tool_response',
    'function': 'tool_response',
}
_SYSTEM_ROLES = {'system', 'developer'}

_INPUT_KEYS = ('instruction', 'question', 'prompt', 'problem', 'input', 'context', 'query', 'user')
_OUTPUT_KEYS = ('output', 'response', 'answer', 'content', 'completion', 'result', 'assistant')


# --------------------------------------------------------------------------
# 通用文本 / 内容块解析（移植自 convert_sft 的解析经验）
# --------------------------------------------------------------------------

def _parse_args(v: Any) -> Any:
    '''把参数尽量解析成结构化对象；已是 str 非 JSON 则原样。'''
    if v is None:
        return ''
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s.startswith(('{', '[')):
        try:
            return json.loads(s)
        except Exception:
            return v
    return v


def _content_blocks(content: list) -> tuple:
    '''按 Anthropic/OpenAI 风格 content 块拆分 -> (texts, thinks, tool_calls)。'''
    texts, thinks, calls = [], [], []
    for b in content:
        if isinstance(b, str):
            texts.append(b)
            continue
        if not isinstance(b, dict):
            continue
        t = b.get('type') or ''
        if t in _TEXT_BLOCK_TYPES:
            v = b.get('text')
            if isinstance(v, str) and v:
                texts.append(v)
        elif t in _THINK_BLOCK_TYPES:
            v = b.get('thinking')
            if v is None:
                v = b.get('content')
            if isinstance(v, str) and v:
                thinks.append(v)
        elif t in _TOOL_BLOCK_TYPES:
            nm = b.get('name')
            if nm:
                calls.append({'name': str(nm), 'param': _parse_args(b.get('arguments', b.get('input')))})
        elif t in _TOOL_RESULT_TYPES:
            v = b.get('content')
            if isinstance(v, str) and v.strip():
                texts.append(v)
            elif isinstance(v, list):
                sub, _, _ = _content_blocks(v)
                texts.extend(sub)
        elif t in _SKIP_BLOCK_TYPES:
            continue
        else:
            v = b.get('text', b.get('content'))
            if isinstance(v, str) and v:
                texts.append(v)
    return texts, thinks, calls


def _content_to_text(c: Any) -> str:
    '''content（str / list-of-blocks / dict / 标量）转纯文本，丢弃 think / tool 段。'''
    if c is None:
        return ''
    if isinstance(c, str):
        return c
    if isinstance(c, (int, float, bool)):
        return str(c)
    if isinstance(c, list):
        texts, _, _ = _content_blocks(c)
        return '\n'.join(p for p in texts if p)
    if isinstance(c, dict):
        if isinstance(c.get('text'), str):
            return c['text']
        return json.dumps(c, ensure_ascii=False)
    return json.dumps(c, ensure_ascii=False)


def _split_think(content: str):
    '''把 '<think>...</think>' 段拆出 -> (cot, remaining)。容忍末尾未闭合 <think>。'''
    if not isinstance(content, str) or '<think>' not in content:
        return '', content
    out, rest = [], content
    while True:
        m = _THINK_RE.search(rest)
        if not m:
            break
        out.append(m.group(1).strip())
        rest = rest[:m.start()] + rest[m.end():]
    if out:
        return '\n\n'.join(x for x in out if x), rest.strip()
    i = rest.find('<think>')
    if i >= 0:
        return rest[i + len('<think>'):].strip(), rest[:i].strip()
    return '', content


def _extract_tool_calls(m: dict) -> list:
    '''OpenAI tool_calls / tool_call / 块内工具 -> [{'name','param'}]。'''
    raw = m.get('tool_calls')
    if raw is None:
        raw = m.get('tool_call')
    if not raw:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    out = []
    for t in raw:
        if not isinstance(t, dict):
            continue
        name, args = None, None
        fn = t.get('function')
        if isinstance(fn, dict):
            name = fn.get('name')
            args = fn.get('arguments')
        if name is None:
            name = t.get('name')
        if args is None:
            args = t.get('input', t.get('arguments'))
        if name:
            out.append({'name': str(name), 'param': _parse_args(args)})
    return out


def _normalize_tools(tools: Any) -> list:
    '''tools 字段（list / JSON str / None）归一化，尽量补 name 提示。'''
    if not tools:
        return []
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except Exception:
            return []
    if not isinstance(tools, list):
        return []
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        d = dict(t)
        fn = d.get('function')
        if isinstance(fn, dict) and fn.get('name') and 'name' not in d:
            d = {'name': fn['name']}
            d.update(t)
        out.append(d)
    return out


# --------------------------------------------------------------------------
# 各种 row / message 形态 -> 内部规范 group
# --------------------------------------------------------------------------

def _canonicalize_model_content(raw: Any):
    '''model 的 content（list 块 / str）-> (cot, content_text, tool_calls)。'''
    calls = []
    if isinstance(raw, list):
        texts, thinks, block_calls = _content_blocks(raw)
        cot = '\n\n'.join(x.strip() for x in thinks if x)
        content = '\n'.join(x for x in texts if x)
        calls = block_calls
    else:
        content = _content_to_text(raw)
        cot, content = _split_think(content)
    return cot, content, calls


def _msg_to_turn(m: dict) -> Optional[dict]:
    '''原始消息 dict -> 规范 turn（role: human/model/tool_response）。system 返回 None。'''
    if not isinstance(m, dict):
        return None
    role = str(m.get('role', ''))
    if role in _SYSTEM_ROLES:
        return None
    mapped = _SOURCE_ROLE_MAP.get(role)
    if mapped is None:
        return None  # 未知 role 忽略
    item: dict = {'role': mapped, 'content': '', 'cot': '', 'tool_call': []}
    calls = _extract_tool_calls(m)
    if mapped == 'model':
        # dedicated cot 字段优先；再 content 块 thinking；再 <think> 拆分
        cot = ''
        for k in _DEDICATED_COT_KEYS:
            v = m.get(k)
            if v is None:
                continue
            if isinstance(v, list):
                v = _content_to_text(v)
            elif not isinstance(v, str):
                v = str(v)
            if v.strip():
                cot = v.strip()
                break
        content = _content_to_text(m.get('content'))
        if cot and '<think>' in content:
            _, content = _split_think(content)
        elif not cot:
            cot, content = _canonicalize_model_content(m.get('content'))[0:2]
        if calls and content.startswith('<think>'):
            _, content = _split_think(content)
        item['cot'], item['content'] = cot, content
    else:
        item['content'] = _content_to_text(m.get('content'))
    item['tool_call'] = calls
    if m.get('name') is not None and mapped == 'tool_response':
        item['name'] = str(m['name'])
    return item


def _row_to_session_group(row: dict) -> Optional[dict]:
    '''把 session / messages / conversations / 单轮 形态行 -> 规范 group，失败返回 None。'''
    # 1) 新结构 session
    if isinstance(row.get('session'), list):
        turns = []
        for t in row['session']:
            role = str(t.get('role', ''))
            if role in ('system', 'developer'):
                continue
            mapped = _SOURCE_ROLE_MAP.get(role, role)
            if mapped not in ('human', 'model', 'tool_response'):
                continue
            item = {'role': mapped, 'content': _content_to_text(t.get('content')), 'cot': '', 'tool_call': []}
            if mapped == 'model':
                cot = t.get('cot')
                if not cot:
                    cot = t.get('reasoning_content')
                if isinstance(cot, str) and cot:
                    item['cot'] = cot
                    if '<think>' in item['content']:
                        _, item['content'] = _split_think(item['content'])
                else:
                    it_cot, item['content'], _ = _canonicalize_model_content(t.get('content'))
                    item['cot'] = it_cot
                tc = t.get('tool_call')
                if tc:
                    if isinstance(tc, dict):
                        tc = [tc]
                    item['tool_call'] = _normalize_tc_list(tc)
                if t.get('name') is not None:
                    item['name'] = str(t['name'])
            elif t.get('name') is not None:
                item['name'] = str(t['name'])
            turns.append(item)
        tools = _normalize_tools(row.get('tool'))
        return {'system': _render_system(row.get('system', ''), tools), 'turns': turns}

    # 2) OpenAI/agentic messages
    msgs = row.get('messages')
    if msgs is None:
        msgs = row.get('conversations')
    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs)
        except Exception:
            return None
    if isinstance(msgs, list):
        system_parts, turns = [], []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get('role', ''))
            if role in _SYSTEM_ROLES:
                c = _content_to_text(m.get('content'))
                if c.strip():
                    system_parts.append(c)
                continue
            # ShareGPT conversations: {from: 'human'/'gpt', value: ...}
            if role == '' and m.get('from'):
                g = str(m.get('from'))
                role = 'user' if g in ('human', 'user') else ('assistant' if g in ('gpt', 'assistant') else role)
                m = {'role': role, 'content': m.get('value')}
            if 'tool_calls' not in m and 'tool_call' not in m and isinstance(m.get('content'), str):
                pass
            item = _msg_to_turn(m)
            if item is not None:
                turns.append(item)
        if not turns:
            return None
        tools = _normalize_tools(row.get('tools', row.get('tool', row.get('functions'))))
        return {'system': _render_system('\n\n'.join(system_parts), tools), 'turns': turns}

    # 3) 单轮列（alpaca / QA / MotifSFT input-content 等）返回标记，交给 merge 池
    return _turn_row_hint(row)


def _normalize_tc_list(tc: Any) -> list:
    out = []
    for t in tc:
        if isinstance(t, dict):
            if t.get('name') is not None:
                out.append({'name': str(t['name']), 'param': _parse_args(t.get('param', t.get('arguments')))})
            elif isinstance(t.get('function'), dict):
                out.append({'name': str(t['function'].get('name', '')),
                            'param': _parse_args(t.get('function', {}).get('arguments'))})
    return out


def _turn_row_hint(row: dict) -> Optional[str]:
    '''检测单轮列形态：命中返回 'turn'（供 merge 池），否则 None。'''
    if any(k in row for k in _INPUT_KEYS) or any(k in row for k in _OUTPUT_KEYS):
        # MotifSFT input/content 也属此类
        return 'turn'
    return None


def _human_from_row(row: dict) -> str:
    for k in _INPUT_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ''


def _model_from_row(row: dict):
    '''单轮行 -> (human, cot, content)。自动处理 output 内 <think> 与 dedicated cot。'''
    human = _human_from_row(row)
    out = None
    # 选一个文本输出列；结构化的（list/dict）优先 content 字段
    for k in _OUTPUT_KEYS:
        if k in row and row.get(k) is not None and k != 'content':
            v = row[k]
            if isinstance(v, str):
                out = v
                break
            if isinstance(v, list):
                texts, _, _ = _content_blocks(v)
                if texts:
                    out = '\n'.join(texts)
                    break
    if out is None and 'content' in row:
        v = row['content']
        if isinstance(v, str):
            out = v
        elif isinstance(v, list):
            texts, thinks, calls = _content_blocks(v)
            if texts:
                out = '\n'.join(texts)
    if out is None:
        return None
    cot = ''
    for k in _DEDICATED_COT_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            cot = v.strip()
            break
    if cot and '<think>' in out:
        _, out = _split_think(out)
    elif not cot:
        cot, out = _split_think(out)
    return human, cot, out


def _render_system(system: Any, tools: list) -> str:
    seg = []
    s = _content_to_text(system)
    if s:
        seg.append(s)
    if tools:
        seg.append('Available tools: ' + json.dumps(tools, ensure_ascii=False))
    return '\n\n'.join(seg)


# --------------------------------------------------------------------------
# CodonSFT
# --------------------------------------------------------------------------

class CodonSFT(CodonDataset):
    '''
    统一 SFT 数据集：folder 内 jsonl / gzip-jsonl / .json / .parquet 每行自动识别多种格式：

    1) 新结构 session（完整会话 / 工具对话）::

        {"type":"SFT","system":"...","tool":[...],
         "session":[{"role":"human|model|tool_response","content","cot","tool_call":[...]}]}

    2) OpenAI / agentic ``messages``（或 ShareGPT ``conversations``）：role 可为
       system/developer/user/assistant/tool/tool_result；content 支持文本或内容块
       （text / thinking / tool_use…）；assistant 支持 ``tool_calls`` 与 dedicated
       reasoning 字段；``<think>…</think>`` 自动拆成 cot。

    3) 单轮问答列：``instruction/question/prompt/problem/input`` +
       ``output/answer/response/completion/content``（含 MotifSFT 的 input/content），
       answer 内嵌 ``<think>`` 或独立 cot 字段自动拆出思考。这类行默认按
       ``two/three_turn_prob`` 做伪多对话组群（``merge_turns=False`` 则每条独立）。

    输出与 MotifSFT 一致的预批 dict。会话级 role 统一 human/user / model / tool_response。
    '''

    def __init__(
        self,
        folder: str,
        tokenizer: PackedTokenizer,
        pad_length: int,
        batch_size: int,
        two_turn_prob: float = 0.2,
        three_turn_prob: float = 0.1,
        system_prompts: Optional[Sequence[str]] = None,
        pattern: str = '*.jsonl',
        recursive: bool = True,
        seed: int = 42,
        merge_turns: bool = True,        # 伪多对话组装开关（对单轮问答行）
    ) -> None:
        if two_turn_prob < 0 or three_turn_prob < 0:
            raise ValueError('turn probabilities must be non-negative')
        if two_turn_prob + three_turn_prob > 1.0:
            raise ValueError('two_turn_prob + three_turn_prob must not exceed 1.0')
        if pad_length <= 0:
            raise ValueError('pad_length must be positive')
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')

        self.tokenizer = tokenizer
        self.pad_length = pad_length
        self.batch_size = batch_size
        self.two_turn_prob = two_turn_prob
        self.three_turn_prob = three_turn_prob
        self.system_prompts = (
            list(system_prompts) if system_prompts else list(_DEFAULT_SYSTEM_PROMPTS)
        )
        self.seed = seed
        self.merge_turns = merge_turns
        self.pattern = pattern
        self.recursive = recursive

        rows = self._load_rows(folder)
        if not rows:
            raise RuntimeError(f'no samples loaded from {folder!r}')

        full_groups, turn_rows = [], []
        for r in rows:
            if not isinstance(r, dict):
                continue
            g = _row_to_session_group(r)
            if isinstance(g, dict):
                full_groups.append(g)
            else:
                turn_rows.append(r)

        self.groups = list(full_groups)
        if turn_rows:
            random.Random(seed).shuffle(turn_rows)
            self.groups.extend(self._build_turn_groups(turn_rows, random.Random(seed + 1)))

    # ------------------------------------------------------------------ 读取
    @staticmethod
    def _iter_jsonl(path: str):
        opener = gzip.open if path.endswith('.gz') else open
        with opener(path, 'rt', encoding='utf-8', errors='replace') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

    @staticmethod
    def _iter_json(path: str):
        '''.json：整文件为单个 dict 或 list[dict]。'''
        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            o = json.load(fh)
        if isinstance(o, list):
            yield from (x for x in o if isinstance(x, dict))
        elif isinstance(o, dict):
            yield o

    @staticmethod
    def _iter_parquet(path: str):
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError('读取 .parquet 需要 pyarrow，请 pip install pyarrow')
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2048):
            for r in batch.to_pylist():
                yield r

    def _load_rows(self, folder: str) -> list:
        rows = []
        suffixes = ('.jsonl', '.json', '.gz', '.parquet')
        paths = []
        if self.recursive:
            for cur, _, fs in os.walk(folder):
                for fn in fs:
                    if fn.lower().endswith(suffixes):
                        paths.append(os.path.join(cur, fn))
        else:
            paths = [
                os.path.join(folder, fn) for fn in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, fn)) and fn.lower().endswith(suffixes)
            ]
        for p in sorted(paths):
            low = p.lower()
            if low.endswith('.jsonl') or low.endswith('.gz'):
                reader = self._iter_jsonl
            elif low.endswith('.json'):
                reader = self._iter_json
            else:
                reader = self._iter_parquet
            try:
                rows.extend(list(reader(p)))
            except Exception as e:
                print(f'[!] skip {os.path.basename(p)}: {e}')
        return rows

    def _build_turn_groups(self, turn_rows: list, rng: random.Random) -> list:
        groups = []
        n_total = len(turn_rows)
        i = 0
        while i < n_total:
            if self.merge_turns:
                r = rng.random()
                remain = n_total - i
                if r < self.three_turn_prob and remain >= 3:
                    n = 3
                elif r < self.three_turn_prob + self.two_turn_prob and remain >= 2:
                    n = 2
                else:
                    n = 1
            else:
                n = 1
            turns = []
            for row in turn_rows[i:i + n]:
                parsed = _model_from_row(row)
                if parsed is None:
                    continue
                human, cot, content = parsed
                if human:
                    turns.append({'role': 'human', 'content': human, 'cot': '', 'tool_call': []})
                if content or cot:
                    turns.append({'role': 'model', 'content': content, 'cot': cot, 'tool_call': []})
            if turns:
                groups.append({'system': rng.choice(self.system_prompts), 'turns': turns})
            i += n
        return groups

    # -------------------------------------------------------------- 编码 / 采样
    def _add_turn(self, session: Session, turn: dict) -> None:
        role = turn.get('role')
        if role == 'human':
            session.add_message({'role': 'user', 'content': turn.get('content', '')})
        elif role == 'model':
            msg: dict = {'role': 'model', 'content': turn.get('content', '')}
            if turn.get('cot'):
                msg['reasoning_content'] = turn['cot']
            tc = turn.get('tool_call')
            if tc:
                norm = []
                for t in tc:
                    nm = t.get('name')
                    if not nm:
                        continue
                    arg = t.get('param')
                    if not isinstance(arg, str):
                        arg = json.dumps(arg, ensure_ascii=False)
                    norm.append({'function': {'name': nm, 'arguments': arg}})
                msg['tool_calls'] = norm
            session.add_message(msg)
        elif role == 'tool_response':
            msg = {'role': 'tool_response', 'content': turn.get('content', '')}
            if turn.get('name'):
                msg['name'] = turn['name']
            session.add_message(msg)
        else:
            raise ValueError(f'unknown canonical role {role!r}')

    def _build_session(self, group: dict) -> Session:
        session = Session(self.tokenizer)
        if group.get('system'):
            session.add_message({'role': 'system', 'content': group['system']})
        for turn in group['turns']:
            self._add_turn(session, turn)
        return session

    def _build_sample(self, group: dict) -> dict[str, Any]:
        session = self._build_session(group)
        tensors = session.to_tensors(pad_to=self.pad_length)
        p = self.pad_length
        if tensors['input_ids'].size(0) > p:
            tensors = {
                k: (v[:p].contiguous() if torch.is_tensor(v) else v)
                for k, v in tensors.items()
            }
        return tensors

    def __len__(self) -> int:
        return len(self.groups) // self.batch_size

    def __getitem__(self, idx: int) -> dict[str, Any]:
        length = len(self)
        if idx < 0:
            idx += length
        if idx < 0 or idx >= length:
            raise IndexError(f'index {idx} out of range for {length} batches')

        start = idx * self.batch_size
        end = start + self.batch_size
        samples = [self._build_sample(g) for g in self.groups[start:end]]
        out: dict[str, Any] = {}
        for key in samples[0]:
            vals = [s[key] for s in samples]
            if torch.is_tensor(vals[0]):
                out[key] = torch.stack(vals)
            elif vals[0] is not None:
                out[key] = [v for item in vals for v in item]
        return out
