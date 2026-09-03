import torch
from typing import Any, Dict, Iterable, Iterator, List, Union, TYPE_CHECKING

from codon.utils.data.base import CodonDataset, CodonIterableDataset
if TYPE_CHECKING:
    from codon.motif.data import MotifPrev1


class ChunkedTokenStream(CodonIterableDataset):
    '''
    Stream `(input_ids, labels)` pairs by packing variable-length token
    records, EOS-separated, into fixed-size chunks.

    Resume preserves the in-flight token buffer and the upstream cursor, so
    no record processed before a checkpoint is re-encoded on resume. Upstream
    skipping reuses the source's native seek when available
    (:meth:`CodonDataset.compose` with ``seek``) and falls back to
    ``next()``-based skipping for generic iterables.

    Conforms to :class:`CodonIterableDataset` and the Stateful protocol.

    Attributes:
        data (Union[Iterable[Any], CodonDataset]): Upstream record source.
        chunk_len (int): Tokens per emitted chunk.
        batch_size (int): Rows per emitted micro-batch.
        seq_len (int): Per-row sequence length.
        eos_token_id (int): Separator id appended after every record.
    '''

    def __init__(
        self,
        data: Union[Iterable[Any], 'MotifPrev1', CodonDataset],
        chunk_len: int,
        batch_size: int,
        seq_len: int,
        eos_token_id: int,
    ) -> None:
        '''
        Initialize the chunked token stream.

        Args:
            data (Union[Iterable[Any], MotifPrev1, CodonDataset]): Upstream
                source. A :class:`CodonDataset` (e.g., :class:`MotifPrev1`)
                is iterated via ``compose(seek=...)`` for zero-overhead
                resume. Any other iterable is iterated with ``next()``-based
                skipping; each yielded item may be a token-id list, a record
                dict containing ``'input_ids'``, or a batch (list of either).
            chunk_len (int): Length of each emitted chunk in tokens.
            batch_size (int): Rows per yielded micro-batch.
            seq_len (int): Per-row sequence length after reshape.
            eos_token_id (int): EOS id used to separate records.

        Raises:
            ValueError: If ``chunk_len != batch_size * (seq_len + 1)``.
        '''
        expected = batch_size * (seq_len + 1)
        if chunk_len != expected:
            raise ValueError(
                f'chunk_len ({chunk_len}) must equal '
                f'batch_size * (seq_len + 1) = {expected}.'
            )
        self.data = data
        self.chunk_len = chunk_len
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id
        self._token_buffer: List[int] = []
        self._upstream_offset: int = 0

    def state_dict(self) -> Dict[str, Any]:
        '''
        Snapshot the live token buffer and the upstream cursor.

        Returns:
            Dict[str, Any]: Picklable state with ``token_buffer`` and
                ``upstream_offset``.
        '''
        return {
            'token_buffer': list(self._token_buffer),
            'upstream_offset': int(self._upstream_offset),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        '''
        Restore the live token buffer and the upstream cursor.

        Args:
            state (Dict[str, Any]): State previously produced by
                :meth:`state_dict`.
        '''
        self._token_buffer = list(state.get('token_buffer', []))
        self._upstream_offset = int(state.get('upstream_offset', 0))

    @staticmethod
    def _extract_ids(record: Any) -> List[int]:
        '''
        Pull token ids out of a single upstream record.

        Args:
            record (Any): Either a dict containing ``input_ids`` or a list
                of ints.

        Returns:
            List[int]: Token id list for the record.
        '''
        if isinstance(record, dict):
            return list(record['input_ids'])
        return list(record)

    @staticmethod
    def _normalize_item(item: Any) -> Iterator[List[int]]:
        '''
        Coerce one upstream item into one or more token-id lists.

        Args:
            item (Any): A record, a record dict, or a batch (list of records
                or record dicts).

        Returns:
            Iterator[List[int]]: One token-id list per record in ``item``.
        '''
        if isinstance(item, dict):
            yield list(item['input_ids'])
            return
        if (
            isinstance(item, (list, tuple))
            and len(item) > 0
            and isinstance(item[0], (list, tuple, dict))
        ):
            for record in item:
                yield ChunkedTokenStream._extract_ids(record)
            return
        yield list(item)

    def _drain_buffer(self) -> Iterator[Any]:
        '''
        Emit as many full chunks as the current token buffer allows.

        Returns:
            Iterator[Any]: Stream of ``(input_ids, labels)`` tensor tuples.
        '''
        while len(self._token_buffer) >= self.chunk_len:
            chunk = self._token_buffer[:self.chunk_len]
            self._token_buffer = self._token_buffer[self.chunk_len:]
            tensor = torch.tensor(chunk, dtype=torch.long).view(
                self.batch_size, self.seq_len + 1
            )
            yield (
                tensor[:, :-1].contiguous(),
                tensor[:, 1:].contiguous(),
            )

    def iter_from(self, offset: int) -> Iterator[Any]:
        '''
        Yield ``(input_ids, labels)`` pairs, resuming from internal state.

        ``offset`` is interpreted in upstream items (records for a
        :class:`CodonDataset`, ``next()`` units otherwise) and is merged
        with the internal cursor. The previously loaded token buffer is
        reused as-is, so no record processed before the cut is re-encoded.

        Args:
            offset (int): Lower bound on upstream items to skip before
                streaming.

        Returns:
            Iterator[Any]: Stream of ``(input_ids, labels)`` tensor tuples.
        '''
        self._upstream_offset = max(self._upstream_offset, int(offset))

        if isinstance(self.data, CodonDataset):
            wrapper = self.data.compose(seek=self._upstream_offset)
            for i in range(len(wrapper)):
                # A map index may yield one record *or* a batch of records
                # (e.g. TextFileDataset with a tokenizer returns one segment
                # list per file). Flatten them like the generic branch below;
                # a single EOS per batch keeps resume offsets aligned.
                for ids in self._normalize_item(wrapper[i]):
                    self._token_buffer.extend(ids)
                    self._token_buffer.append(self.eos_token_id)
                self._upstream_offset += 1
                yield from self._drain_buffer()
            return

        upstream_iter = iter(self.data)
        for _ in range(self._upstream_offset):
            try:
                next(upstream_iter)
            except StopIteration:
                return
        for item in upstream_iter:
            for ids in self._normalize_item(item):
                self._token_buffer.extend(ids)
                self._token_buffer.append(self.eos_token_id)
            self._upstream_offset += 1
            yield from self._drain_buffer()
