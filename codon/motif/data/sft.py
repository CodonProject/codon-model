import json
import os
import random
from glob import glob
from typing import Any, Optional, Sequence

import torch

from codon.utils.data.base import CodonDataset
from codon.utils.session   import Session
from codon.utils.tokens    import PackedTokenizer


_DEFAULT_SYSTEM_PROMPTS = [
    'You are a helpful assistant.',
    'You are a AI.',
    'Answer the user concisely and accurately.',
]


class MotifSFT(CodonDataset):
    '''
    SFT dataset built from a folder of jsonl files.

    Each jsonl row is expected to carry the fields `input`, `cot`, `content`.
    All rows are globally shuffled once, then consumed sequentially to form
    dialog groups:
      - with probability `three_turn_prob` three consecutive rows are merged
        into a three-turn sys-user-model-user-model-user-model sample,
      - with probability `two_turn_prob` two rows are merged into a two-turn
        sys-user-model-user-model sample,
      - otherwise the row becomes a single sys-user-model sample.
    Every group is prefixed with a randomly sampled system prompt.

    Each sample is tokenized via `Session`, padded to `pad_length`, and
    anything longer is truncated. `__getitem__` returns a pre-batched dict
    of tensors with shape [batch_size, pad_length].
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

        self.samples = self._load_jsonl(folder, pattern, recursive)
        if not self.samples:
            raise RuntimeError(f'no jsonl samples loaded from {folder!r}')

        random.Random(seed).shuffle(self.samples)

        self.groups = self._build_groups(random.Random(seed + 1))

    @staticmethod
    def _load_jsonl(folder: str, pattern: str, recursive: bool) -> list[dict]:
        if recursive:
            paths = sorted(
                glob(os.path.join(folder, '**', pattern), recursive=True)
            )
        else:
            paths = sorted(glob(os.path.join(folder, pattern)))

        rows: list[dict] = []
        for path in paths:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
        return rows

    def _build_groups(self, rng: random.Random) -> list[dict]:
        groups: list[dict] = []
        n_total = len(self.samples)
        i = 0
        while i < n_total:
            r = rng.random()
            remain = n_total - i
            if r < self.three_turn_prob and remain >= 3:
                n = 3
            elif r < self.three_turn_prob + self.two_turn_prob and remain >= 2:
                n = 2
            else:
                n = 1
            groups.append({
                'system': rng.choice(self.system_prompts),
                'turns': self.samples[i:i + n],
            })
            i += n
        return groups

    def _build_session(self, group: dict) -> Session:
        session = Session(self.tokenizer)
        session.add_message({'role': 'system', 'content': group['system']})
        for turn in group['turns']:
            session.add_message({
                'role': 'user',
                'content': turn.get('input', ''),
            })
            model_msg: dict = {
                'role': 'model',
                'content': turn.get('content', ''),
            }
            cot = turn.get('cot')
            if cot:
                model_msg['reasoning_content'] = cot
            session.add_message(model_msg)
        return session

    def _build_sample(self, group: dict) -> dict[str, Any]:
        session = self._build_session(group)
        tensors = session.to_tensors(pad_to=self.pad_length)
        p = self.pad_length
        if tensors['input_ids'].size(0) > p:
            # 仅截断张量字段；images 等 list 字段保持原值（纯文本 SFT 为空）。
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
        # Session.to_tensors 的 images 键是 list[Tensor]（纯文本为空 list），
        # 不能 torch.stack；只对张量字段做 batch 堆叠，list 字段按样本展平拼接。
        out: dict[str, Any] = {}
        for key in samples[0]:
            vals = [s[key] for s in samples]
            if torch.is_tensor(vals[0]):
                out[key] = torch.stack(vals)
            elif vals[0] is not None:
                out[key] = [v for item in vals for v in item]
        return out