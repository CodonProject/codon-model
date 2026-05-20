import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Generator, Dict, Iterator, Union

from codon.utils.session import Session

# Data containers

@dataclass
class ChatChunk:
    '''
    A streamed fragment of a chat response.

    Attributes:
        content (str): Decoded text of this chunk (may be empty for pure events).
        is_cot (bool): Whether this chunk belongs to the chain-of-thought segment.
        cot_started (bool): Event flag set on the chunk that opens [cot_start].
        cot_ended (bool): Event flag set on the chunk that closes [cot_end].
        finished (bool): Event flag set on the final [im_end] chunk.
        token_ids (List[int]): Token ids carried by this chunk.
    '''
    content: str = ''
    is_cot: bool = False
    cot_started: bool = False
    cot_ended: bool = False
    finished: bool = False
    token_ids: List[int] = field(default_factory=list)

    def __add__(self, other: 'ChatChunk') -> 'ChatChunk':
        if not isinstance(other, ChatChunk):
            return NotImplemented
        return ChatChunk(
            content=self.content + other.content,
            is_cot=other.is_cot,
            cot_started=self.cot_started or other.cot_started,
            cot_ended=self.cot_ended or other.cot_ended,
            finished=self.finished or other.finished,
            token_ids=self.token_ids + other.token_ids,
        )

    __radd__ = __add__

    def to_message(self) -> Dict:
        if self.is_cot:
            return {'role': 'model', 'reasoning_content': self.content}
        return {'role': 'model', 'content': self.content}


@dataclass
class ChatResponse:
    '''
    A fully accumulated chat response.

    Attributes:
        content (str): Final answer text (post-[cot_end]).
        reasoning_content (str): Chain-of-thought text.
        token_ids (List[int]): All generated token ids (incl. specials).
        finish_reason (str): One of 'stop' (saw [im_end]) or 'length'.
    '''
    content: str = ''
    reasoning_content: str = ''
    token_ids: List[int] = field(default_factory=list)
    finish_reason: str = 'length'

    def to_message(self) -> Dict:
        msg: Dict = {'role': 'model', 'content': self.content}
        if self.reasoning_content:
            msg['reasoning_content'] = self.reasoning_content
        return msg

    @classmethod
    def from_chunks(cls, chunks: Iterator[ChatChunk]) -> 'ChatResponse':
        content_parts, cot_parts, ids = [], [], []
        finish = 'length'
        for c in chunks:
            ids.extend(c.token_ids)
            if c.content:
                (cot_parts if c.is_cot else content_parts).append(c.content)
            if c.finished:
                finish = 'stop'
        return cls(
            content=''.join(content_parts),
            reasoning_content=''.join(cot_parts),
            token_ids=ids,
            finish_reason=finish,
        )

# Token-level streaming generator

@torch.no_grad()
def generate_stream(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 4,
    eos_token_id: Optional[int] = None,
    bad_token_ids: Optional[List[int]] = None,
    bad_token_bias: float = -3.0,
    newline_token_ids: Optional[List[int]] = None,
    max_consecutive_newlines: int = 2,
) -> Generator[int, None, None]:
    '''Yield generated token ids one at a time.'''
    generated = input_ids.clone()
    nl_set = set(newline_token_ids or [])

    for _ in range(max_new_tokens):
        out = model(generated)
        logits = out.logits[:, -1, :].float()

        if repetition_penalty and repetition_penalty != 1.0:
            seen_set = list(set(generated[0].tolist()))
            scores = logits[0, seen_set]
            scores = torch.where(
                scores > 0, scores / repetition_penalty, scores * repetition_penalty
            )
            logits[0, seen_set] = scores

        if bad_token_ids:
            logits[0, bad_token_ids] += bad_token_bias

        if nl_set and max_consecutive_newlines > 0:
            tail = generated[0, -max_consecutive_newlines:].tolist()
            if len(tail) >= max_consecutive_newlines and all(t in nl_set for t in tail):
                for nl_id in nl_set:
                    logits[0, nl_id] = -float('inf')

        if no_repeat_ngram_size and generated.size(1) >= no_repeat_ngram_size:
            seq = generated[0].tolist()
            n = no_repeat_ngram_size
            prefix = tuple(seq[-(n - 1):])
            banned = set()
            for i in range(len(seq) - n + 1):
                if tuple(seq[i:i + n - 1]) == prefix:
                    banned.add(seq[i + n - 1])
            for tid in banned:
                logits[0, tid] = -float('inf')

        if temperature and temperature > 0:
            logits = logits / temperature

        if top_k and top_k > 0:
            kth = torch.topk(logits, top_k).values[:, -1, None]
            logits = torch.where(
                logits < kth, torch.full_like(logits, -float('inf')), logits
            )

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, -float('inf'))
            logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_id = next_token.item()

        generated = torch.cat([generated, next_token], dim=1)
        yield next_id

        if eos_token_id is not None and next_id == eos_token_id:
            break


@torch.no_grad()
def generate(model, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
    '''Non-streaming wrapper. Returns the full token tensor [1, T_full].'''
    out_ids = input_ids[0].tolist()
    for tid in generate_stream(model, input_ids, **kwargs):
        out_ids.append(tid)
    return torch.tensor(
        [out_ids], dtype=input_ids.dtype, device=input_ids.device
    )


# Chat layer

def _detect_newline_ids(tokenizer) -> List[int]:
    nl_ids = []
    vocab = tokenizer.fast_tokenizer.get_vocab()
    for tok, tid in vocab.items():
        if 'Ċ' in tok and tok.replace('Ċ', '') == '':
            nl_ids.append(tid)
    return nl_ids


def _build_chat_session(tokenizer, messages, system_prompt) -> Session:
    sess = Session(tokenizer)
    has_system = any(
        m.get('role') in ('system', 'instruction', 'developer') for m in messages
    )
    if system_prompt is not None and not has_system:
        sess.add_message({'role': 'system', 'content': system_prompt})
    for msg in messages:
        sess.add_message(msg)
    sess.add_generation_prompt()
    return sess


def chat_stream(
    model,
    tokenizer,
    device,
    messages: List[Dict],
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 512,
    **gen_kwargs,
) -> Generator[ChatChunk, None, None]:
    '''
    Stream a chat completion as ChatChunks.

    The model decides on its own whether to open [cot_start]; this generator
    tracks the CoT state machine and yields chunks tagged with `is_cot`.
    Special tokens are emitted as event-only chunks (empty content) with
    `cot_started`, `cot_ended`, or `finished` set.
    '''
    sess = _build_chat_session(tokenizer, messages, system_prompt)
    p_tensor = torch.tensor([sess.input_ids], dtype=torch.long, device=device)

    cot_start_id = tokenizer.token_to_id('[cot_start]')
    cot_end_id   = tokenizer.token_to_id('[cot_end]')
    eos_id       = tokenizer.token_to_id('[im_end]')

    if 'newline_token_ids' not in gen_kwargs:
        gen_kwargs['newline_token_ids'] = _detect_newline_ids(tokenizer)

    model.eval()

    in_cot = False
    pending: List[int] = []

    def _flush(is_cot_now: bool):
        if not pending:
            return None
        text = tokenizer.decode(pending, skip_special_tokens=False)
        chunk = ChatChunk(content=text, is_cot=is_cot_now, token_ids=list(pending))
        pending.clear()
        return chunk

    for tid in generate_stream(
        model, p_tensor,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_id,
        **gen_kwargs,
    ):
        if tid == cot_start_id:
            flushed = _flush(in_cot)
            if flushed is not None:
                yield flushed
            in_cot = True
            yield ChatChunk(is_cot=True, cot_started=True, token_ids=[tid])
            continue

        if tid == cot_end_id:
            flushed = _flush(True)
            if flushed is not None:
                yield flushed
            in_cot = False
            yield ChatChunk(is_cot=False, cot_ended=True, token_ids=[tid])
            continue

        if tid == eos_id:
            flushed = _flush(in_cot)
            if flushed is not None:
                yield flushed
            yield ChatChunk(is_cot=in_cot, finished=True, token_ids=[tid])
            return

        pending.append(tid)
        text = tokenizer.decode(pending, skip_special_tokens=False)
        # Wait for complete UTF-8 codepoints before emitting.
        if '\ufffd' not in text:
            yield ChatChunk(content=text, is_cot=in_cot, token_ids=list(pending))
            pending.clear()

    # Reached max_new_tokens without [im_end]: flush any tail.
    flushed = _flush(in_cot)
    if flushed is not None:
        yield flushed


def chat(
    model,
    tokenizer,
    device,
    messages: List[Dict],
    system_prompt: Optional[str] = None,
    stream: bool = False,
    max_new_tokens: int = 512,
    **gen_kwargs,
) -> Union[ChatResponse, Generator[ChatChunk, None, None]]:
    '''
    Chat completion.
      - stream=True : returns a generator of ChatChunk.
      - stream=False: returns an aggregated ChatResponse.

    `messages` is an OpenAI-style history: [{'role': ..., 'content': ...}, ...].
    `system_prompt` is injected only if no system message is already present.
    '''
    if stream:
        return chat_stream(
            model, tokenizer, device,
            messages=messages,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )

    chunks = chat_stream(
        model, tokenizer, device,
        messages=messages,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        **gen_kwargs,
    )
    return ChatResponse.from_chunks(chunks)

# Convenience probe (kept for backward compat with training loop)

def chat_probe(
    model,
    tokenizer,
    device,
    user_prompt: str,
    system_prompt: str = 'You are a helpful assistant.',
    max_new_tokens: int = 256,
):
    resp = chat(
        model, tokenizer, device,
        messages=[{'role': 'user', 'content': user_prompt}],
        system_prompt=system_prompt,
        stream=False,
        max_new_tokens=max_new_tokens,
        temperature=0.7, top_k=50, top_p=0.9,
        repetition_penalty=1.15, no_repeat_ngram_size=4,
        max_consecutive_newlines=2,
    )
    if resp.reasoning_content:
        print(f'[cot_start]{resp.reasoning_content}[cot_end]', end='')
    print(resp.content)
    return resp