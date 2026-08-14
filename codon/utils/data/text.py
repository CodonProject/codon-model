import os
import random
import bisect
from typing import List, Optional, Dict, Any, Union

import pyarrow.parquet as pq
from tqdm import tqdm   # 进度条

from codon.utils.data.base import CodonDataset
from codon.utils.tokens import PackedTokenizer


class TextFileDataset(CodonDataset):
    """
    Map-style dataset that loads text from either:
        - A directory of .txt files (original behavior)
        - A single .parquet file (lazy or preloaded)

    Args:
        root (str): Root directory to scan for .txt files, OR a path to a .parquet file.
        recursive (bool): If True, scan subdirectories recursively (ignored for Parquet).
        encoding (str): File encoding to use when reading .txt files (ignored for Parquet).
        return_path (bool): If True, __getitem__ returns (path, content);
            for Parquet, path is "parquet_row_<idx>".
        tokenizer (Optional[PackedTokenizer]): Tokenizer to apply to file content.
            If None, raw text is returned.
        seq_len (Optional[int]): Length of each token segment. Required if tokenizer given.
        drop_last (bool): If True, discard the last segment if length < seq_len.
        shuffle (bool): If True, shuffle the file list at initialization (ignored for Parquet).
        seed (Optional[int]): Random seed for shuffling.
        cache_size (int): Number of row groups to keep in cache (for Parquet lazy mode only).
        lazy_mode (bool): If True (default), load data on demand.
            If False, preload all data at initialization (shows progress bar).
    """

    def __init__(
        self,
        root: str,
        recursive: bool = False,
        encoding: str = 'utf-8',
        return_path: bool = False,
        tokenizer: Optional[PackedTokenizer] = None,
        seq_len: Optional[int] = None,
        drop_last: bool = True,
        shuffle: bool = False,
        seed: Optional[int] = None,
        cache_size: int = 4,
        lazy_mode: bool = True,          # new parameter
    ) -> None:
        self.root = os.path.abspath(root)
        self.recursive = recursive
        self.encoding = encoding
        self.return_path = return_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.cache_size = cache_size
        self.lazy_mode = lazy_mode

        if tokenizer is not None and seq_len is None:
            raise ValueError("seq_len must be provided when tokenizer is given.")

        # Determine mode: Parquet or TXT
        self.is_parquet = False
        self.parquet_path = None
        self.total_rows = 0
        self.row_group_offsets = []
        self.parquet_file = None
        self.row_group_cache = {}
        self.cache_order = []

        self.files = []          # list of file paths (or dummy for parquet)
        self.data = None         # preloaded data when lazy_mode=False

        if os.path.isfile(self.root) and self.root.lower().endswith('.parquet'):
            self.is_parquet = True
            self.parquet_path = self.root
            self._init_parquet()
            if not self.lazy_mode:
                self._preload_parquet()
        else:
            if not os.path.isdir(self.root):
                raise NotADirectoryError(f"'{self.root}' is not a directory or a .parquet file.")
            self._scan_txt_files()
            if not self.lazy_mode:
                self._preload_txt_files()

    # ---------- initialization helpers ----------

    def _init_parquet(self) -> None:
        """Open Parquet file, read metadata, build row_group_offsets."""
        self.parquet_file = pq.ParquetFile(self.parquet_path)
        self.total_rows = self.parquet_file.metadata.num_rows
        offsets = [0]
        for i in range(self.parquet_file.num_row_groups):
            rg = self.parquet_file.metadata.row_group(i)
            offsets.append(offsets[-1] + rg.num_rows)
        self.row_group_offsets = offsets[:-1]
        self.files = [self.parquet_path] * self.total_rows

    def _scan_txt_files(self) -> None:
        """Collect .txt files from root directory."""
        if self.recursive:
            for dirpath, _, filenames in os.walk(self.root):
                for fname in filenames:
                    if fname.lower().endswith('.txt'):
                        self.files.append(os.path.join(dirpath, fname))
        else:
            for entry in os.scandir(self.root):
                if entry.is_file() and entry.name.lower().endswith('.txt'):
                    self.files.append(entry.path)
        self.files.sort()
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(self.files)

    def _preload_parquet(self) -> None:
        """Read all 'content' from Parquet into self.data with progress bar."""
        if self.parquet_file is None:
            self.parquet_file = pq.ParquetFile(self.parquet_path)
        total = self.total_rows
        self.data = []
        # Determine batch size for progress reporting
        batch_size = max(1000, total // 100)   # at most 100 batches
        with tqdm(total=total, desc="Loading Parquet", unit=" rows") as pbar:
            for batch in self.parquet_file.iter_batches(batch_size=batch_size, columns=['content']):
                # batch is a RecordBatch; convert 'content' column to list of strings
                col = batch.column('content')
                # pyarrow.array to Python list
                texts = col.to_pylist()
                self.data.extend(texts)
                pbar.update(len(texts))
        # Ensure length matches
        if len(self.data) != total:
            # In case some rows are None, replace with ""
            self.data = ["" if x is None else x for x in self.data]

    def _preload_txt_files(self) -> None:
        """Read all .txt files into self.data with progress bar."""
        self.data = []
        with tqdm(total=len(self.files), desc="Loading TXT files", unit="file") as pbar:
            for path in self.files:
                try:
                    with open(path, 'r', encoding=self.encoding) as f:
                        text = f.read()
                except Exception:
                    text = ""   # fallback to empty string
                self.data.append(text)
                pbar.update(1)

    # ---------- dataset methods ----------

    def __len__(self) -> int:
        if self.is_parquet:
            return self.total_rows
        return len(self.files)

    def __getitem__(self, idx: int) -> Union[str, List[List[int]], tuple]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range (0-{len(self)-1})")

        if not self.lazy_mode:
            # Preloaded mode: content is already in self.data
            content = self.data[idx]
            if self.is_parquet:
                path = f"parquet_row_{idx}"
            else:
                path = self.files[idx]
        else:
            # Lazy mode: load on demand
            if self.is_parquet:
                rg_idx = bisect.bisect_right(self.row_group_offsets, idx) - 1
                offset = idx - self.row_group_offsets[rg_idx]
                table = self._get_row_group_table(rg_idx)
                content = table['content'][offset].as_py()
                if content is None:
                    content = ""
                path = f"parquet_row_{idx}"
            else:
                path = self.files[idx]
                try:
                    with open(path, 'r', encoding=self.encoding) as f:
                        content = f.read()
                except Exception as e:
                    raise IOError(f"Failed to read '{path}': {e}")

        # Tokenize if requested
        if self.tokenizer is not None:
            token_ids = self.tokenizer.encode(content, apply_safe_rule=False, add_special_tokens=False)
            segments = []
            for i in range(0, len(token_ids), self.seq_len):
                seg = token_ids[i:i + self.seq_len]
                if self.drop_last and len(seg) < self.seq_len:
                    continue
                segments.append(seg)
            content = segments

        if self.return_path:
            return path, content
        return content

    def _get_row_group_table(self, rg_idx: int):
        """Lazy cache for row groups (only used when lazy_mode=True)."""
        if rg_idx in self.row_group_cache:
            return self.row_group_cache[rg_idx]
        table = self.parquet_file.read_row_group(rg_idx, columns=['content'])
        # Simple LRU
        if len(self.row_group_cache) >= self.cache_size:
            oldest = self.cache_order.pop(0)
            self.row_group_cache.pop(oldest, None)
        self.row_group_cache[rg_idx] = table
        self.cache_order.append(rg_idx)
        return table

    # ---------- stateful protocol ----------

    def state_dict(self) -> Dict[str, Any]:
        state = {
            'root': self.root,
            'recursive': self.recursive,
            'encoding': self.encoding,
            'return_path': self.return_path,
            'seq_len': self.seq_len,
            'drop_last': self.drop_last,
            'shuffle': self.shuffle,
            'seed': self.seed,
            'cache_size': self.cache_size,
            'lazy_mode': self.lazy_mode,
            'is_parquet': self.is_parquet,
        }
        if self.is_parquet:
            state['parquet_path'] = self.parquet_path
            state['total_rows'] = self.total_rows
        else:
            state['files'] = self.files
        # We do NOT save preloaded data (too large)
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        # Restore parameters
        self.encoding = state.get('encoding', self.encoding)
        self.return_path = state.get('return_path', self.return_path)
        self.seq_len = state.get('seq_len', self.seq_len)
        self.drop_last = state.get('drop_last', self.drop_last)
        self.shuffle = state.get('shuffle', self.shuffle)
        self.seed = state.get('seed', self.seed)
        self.cache_size = state.get('cache_size', self.cache_size)
        self.lazy_mode = state.get('lazy_mode', self.lazy_mode)

        is_parquet = state.get('is_parquet', False)
        if is_parquet:
            self.is_parquet = True
            self.parquet_path = state.get('parquet_path')
            self.total_rows = state.get('total_rows', 0)
            # Re-open file and rebuild metadata
            self.parquet_file = pq.ParquetFile(self.parquet_path)
            self._init_parquet()   # resets row_group_offsets, files
            if not self.lazy_mode:
                self._preload_parquet()
        else:
            self.is_parquet = False
            stored_files = state.get('files')
            if stored_files is not None:
                self.files = list(stored_files)
            if not self.lazy_mode:
                self._preload_txt_files()
        # root and recursive are not altered to avoid changing scan scope