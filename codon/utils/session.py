from codon.utils.tokens import PackedTokenizer
from dataclasses import dataclass, field
from typing import Optional, Union, Literal, Sequence
import copy
import torch
import json

MaskPolicy = Union[Literal['all', 'none', 'content', 'thought', 'fim'], Sequence[bool]]

@dataclass
class Message:
    ids: list[int]
    ignore_mask: list[bool] = field(default_factory=list)
    role: Optional[str] = None

    def __post_init__(self):
        if not self.ignore_mask:
            self.ignore_mask = [False] * len(self.ids)
        if len(self.ignore_mask) != len(self.ids):
            raise ValueError(
                f'ignore_mask length {len(self.ignore_mask)} != ids length {len(self.ids)}'
            )

    def __len__(self) -> int:
        return len(self.ids)

    def _resolve_range(self, begin: int, end: int, include: tuple[bool, bool]) -> tuple[int, int]:
        lo = begin if include[0] else begin + 1
        hi = end + 1 if include[1] else end
        return max(0, lo), min(len(self.ids), hi)

    def mask_all(self) -> 'Message':
        self.ignore_mask = [True] * len(self.ids)
        return self

    def unmask_all(self) -> 'Message':
        self.ignore_mask = [False] * len(self.ids)
        return self

    def mask_before(self, index: int) -> 'Message':
        for i in range(max(0, min(index, len(self.ids)))):
            self.ignore_mask[i] = True
        return self

    def mask_after(self, index: int) -> 'Message':
        for i in range(max(0, index), len(self.ids)):
            self.ignore_mask[i] = True
        return self

    def mask_between(
        self, begin: int, end: int,
        include_boundaries: tuple[bool, bool] = (False, False)
    ) -> 'Message':
        lo, hi = self._resolve_range(begin, end, include_boundaries)
        for i in range(lo, hi):
            self.ignore_mask[i] = True
        return self

    def unmask_between(
        self, begin: int, end: int,
        include_boundaries: tuple[bool, bool] = (False, False)
    ) -> 'Message':
        lo, hi = self._resolve_range(begin, end, include_boundaries)
        for i in range(lo, hi):
            self.ignore_mask[i] = False
        return self

    def insert_before(self, index: int, data: list[int], ignore_mask: bool = True) -> 'Message':
        mask = [ignore_mask] * len(data) if isinstance(ignore_mask, bool) else list(ignore_mask)
        self.ids = self.ids[:index] + list(data) + self.ids[index:]
        self.ignore_mask = self.ignore_mask[:index] + mask + self.ignore_mask[index:]
        return self

    def insert_after(self, index: int, data: list[int], ignore_mask: bool = True) -> 'Message':
        return self.insert_before(index + 1, data, ignore_mask)

    def find(self, token_id: int, start: int = 0) -> int:
        try:
            return self.ids.index(token_id, start)
        except ValueError:
            return -1

    def find_last(self, token_id: int) -> int:
        for i in range(len(self.ids) - 1, -1, -1):
            if self.ids[i] == token_id:
                return i
        return -1


class Session:
    _ROLE_ALIAS = {
        'assistant': 'model',
        'instruction': 'system',
        'developer': 'system',
    }

    def __init__(self, tokenizer: PackedTokenizer):
        self.tokenizer = tokenizer
        self.messages: list[Message] = []

        self._tokens = {
            'im_start': '[im_start]', 'im_end': '[im_end]',
            'system': '[system]', 'user': '[user]',
            'model': '[model]', 'tool': '[tool]',
            'fim': '[fim]',
            'cot_start': '[cot_start]', 'cot_end': '[cot_end]',
            'fim_pre': '[fim_pre]', 'fim_mid': '[fim_mid]', 'fim_suf': '[fim_suf]',
            'pad': '[pad]',
        }
        self._ids: dict[str, Optional[int]] = {}
        self._resolve_specials()

        self.policy: dict[str, MaskPolicy] = {
            'system': 'all',
            'user': 'all',
            'tool': 'all',
            'model': 'content',
            'fim': 'fim',
        }

    def _resolve_specials(self) -> None:
        self._ids = {k: self.tokenizer.token_to_id(v) for k, v in self._tokens.items()}

    def set_token(self, mapping: dict) -> 'Session':
        self._tokens.update(mapping)
        self._resolve_specials()
        return self

    def set_policy(self, role: str, policy: MaskPolicy) -> 'Session':
        self.policy[self._ROLE_ALIAS.get(role, role)] = policy
        return self

    def _encode_message(self, message: dict) -> list[int]:
        msg = copy.deepcopy(message)
        role = self._ROLE_ALIAS.get(msg.get('role', 'user'), msg.get('role', 'user'))
        msg['role'] = role
        if role == 'model':
            msg.setdefault('thought', None)
            msg.setdefault('reasoning_content', None)
        return self.tokenizer.apply_chat_template(
            [msg],
            add_generation_prompt=False,
            tokenize=True,
        )

    def _apply_policy(self, msg: Message, policy: MaskPolicy) -> None:
        if isinstance(policy, (list, tuple)):
            if len(policy) != len(msg):
                raise ValueError(f'explicit mask length {len(policy)} != message length {len(msg)}')
            msg.ignore_mask = [bool(x) for x in policy]
            return

        if policy == 'all':
            msg.mask_all()
        elif policy == 'none':
            msg.unmask_all()
        elif policy == 'content':
            msg.mask_all()
            anchor = self._ids.get('cot_start')
            idx = msg.find(anchor) if anchor is not None else -1
            if idx < 0:
                role_id = self._ids.get(msg.role or '')
                idx = msg.find(role_id) if role_id is not None else 0
            msg.unmask_between(idx, len(msg), include_boundaries=(True, True))
        elif policy == 'thought':
            msg.mask_all()
            s_id, e_id = self._ids.get('cot_start'), self._ids.get('cot_end')
            if s_id is None or e_id is None:
                return
            s = msg.find(s_id)
            e = msg.find(e_id, max(s, 0))
            if s >= 0 and e > s:
                msg.unmask_between(s, e, include_boundaries=(False, False))
        elif policy == 'fim':
            msg.mask_all()
            mid_id = self._ids.get('fim_mid')
            end_id = self._ids.get('im_end')
            if mid_id is None or end_id is None:
                return
            mid_idx = msg.find(mid_id)
            end_idx = msg.find_last(end_id)
            if mid_idx >= 0 and end_idx > mid_idx:
                msg.unmask_between(mid_idx, end_idx, include_boundaries=(True, True))
        else:
            raise ValueError(f'unknown mask policy: {policy!r}')

    def _pop_gen_prompt(self) -> Optional[Message]:
        if self.messages and self.messages[-1].role == '__gen_prompt__':
            return self.messages.pop()
        return None
    
    def _append(self, msg: Message) -> None:
        gen = self._pop_gen_prompt()
        self.messages.append(msg)
        if gen is not None:
            self.messages.append(gen)

    def add_message(
        self, message: dict,
        mask: Optional[MaskPolicy] = None,
    ) -> 'Session':
        if 'tool_calls' in message.keys() and isinstance(message['tool_calls'], str):
            message['tool_calls'] = json.loads(message['tool_calls'])
        role = self._ROLE_ALIAS.get(message.get('role', 'user'), message.get('role', 'user'))
        ids = self._encode_message(message)
        msg = Message(ids=ids, role=role)
        self._apply_policy(msg, mask if mask is not None else self.policy.get(role, 'all'))
        self._append(msg)
        return self

    def add_messages(self, messages: list[dict]) -> 'Session':
        for m in messages:
            self.add_message(m)
        return self

    def add_generation_prompt(
        self,
        enable_thinking: bool = False,
        disable_thinking: bool = False
    ) -> 'Session':
        if enable_thinking and disable_thinking:
            raise ValueError('enable_thinking and disable_thinking cannot both be True')
        self._pop_gen_prompt()
        parts = ['im_start', 'model']
        if enable_thinking:
            parts.append('cot_start')
        elif disable_thinking:
            parts.extend(['cot_start', 'cot_end'])
        ids: list[int] = []
        for key in parts:
            tid = self._ids.get(key)
            if tid is None:
                raise ValueError(f'token {self._tokens.get(key, key)!r} not found in vocab')
            ids.append(tid)
        msg = Message(ids=ids, role='__gen_prompt__')
        msg.mask_all()
        self.messages.append(msg)
        return self

    def pop(self, index: int = -1) -> Message:
        return self.messages.pop(index)

    def clear(self) -> 'Session':
        self.messages.clear()
        return self

    def pad_to(
        self,
        length: int,
        side: Literal['left', 'right'] = 'right',
    ) -> 'Session':
        current_len = len(self)
        if current_len >= length:
            return self
        pad_id = self._ids.get('pad')
        if pad_id is None:
            raise ValueError('pad token not found in tokenizer')
        delta = length - current_len
        pad_msg = Message(
            ids=[pad_id] * delta,
            ignore_mask=[True] * delta,
            role='__padding__',
        )
        if side == 'right':
            self.messages.append(pad_msg)
        elif side == 'left':
            self.messages.insert(0, pad_msg)
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        return self

    @property
    def input_ids(self) -> list[int]:
        out: list[int] = []
        for m in self.messages:
            out.extend(m.ids)
        return out

    @property
    def ignore_mask(self) -> list[bool]:
        out: list[bool] = []
        for m in self.messages:
            out.extend(m.ignore_mask)
        return out

    @property
    def labels(self) -> list[int]:
        return [-100 if ig else tid for tid, ig in zip(self.input_ids, self.ignore_mask)]

    def to_tensors(
        self,
        device: Union[str, torch.device] = 'cpu',
        pad_to: Optional[int] = None,
        batch_dim: bool = False,
    ) -> dict[str, torch.Tensor]:
        ids = self.input_ids
        lbl = self.labels
        att = [1] * len(ids)

        if pad_to is not None and len(ids) < pad_to:
            pad_id = self._ids.get('pad') or 0
            delta = pad_to - len(ids)
            ids += [pad_id] * delta
            lbl += [-100] * delta
            att += [0] * delta

        t = lambda xs: torch.tensor(xs, dtype=torch.long, device=device)
        out = {'input_ids': t(ids), 'labels': t(lbl), 'attention_mask': t(att)}
        if batch_dim:
            out = {k: v.unsqueeze(0) for k, v in out.items()}
        return out

    def decode(self, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(self.input_ids, skip_special_tokens=skip_special_tokens)

    def __len__(self) -> int:
        return sum(len(m) for m in self.messages)

    def __iter__(self):
        return iter(self.messages)

    def __getitem__(self, idx: int) -> Message:
        return self.messages[idx]