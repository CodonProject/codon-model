from codon.utils.data.base import CodonDataset
from codon.utils.tokens import PackedTokenizer
from transformers import PreTrainedTokenizerFast

import os
import bisect
import pyarrow.parquet as pq

from pathlib import Path
from typing  import Any, Optional, Dict, Union
from tqdm    import tqdm


class MotifPrev1(CodonDataset):
    '''
    A concrete implementation of CodonDataset for loading Motif data from a directory.
    Optimized with O(1) lazy loading via Parquet metadata or full memory mapping initialization.
    '''

    def __init__(self, path: str, mode: str = 'lazy') -> None:
        if not os.path.isdir(path):
            raise NotADirectoryError(f'{path}')

        if mode not in ['lazy', 'full']:
            raise ValueError('')

        self.path = path
        self.mode = mode
        self.tokenizer: Optional[PreTrainedTokenizerFast] = None
        
        self.file_paths = sorted(list(Path(path).glob('*.parquet')))
        if not self.file_paths:
            raise FileNotFoundError(f'{path}')

        self.cum_sizes = []
        self._table_cache: Dict[int, Any] = {}
        current_total = 0
        
        with tqdm(total=len(self.file_paths), desc=f'Loading Dataset ({mode})', leave=False) as pbar:
            for idx, fp in enumerate(self.file_paths):
                if mode == 'lazy':
                    meta = pq.read_metadata(fp)
                    current_total += meta.num_rows
                elif mode == 'full':
                    table = pq.read_table(fp, memory_map=True)
                    self._table_cache[idx] = table
                    current_total += table.num_rows
                
                self.cum_sizes.append(current_total)
                pbar.update(1)
            
        self.total_rows = current_total

    def set_tokenizer(self, tokenizer: Union[PreTrainedTokenizerFast, PackedTokenizer]) -> 'MotifPrev1':
        if isinstance(tokenizer, PackedTokenizer):
            tokenizer = tokenizer.fast_tokenizer
        self.tokenizer = tokenizer
        return self

    def __len__(self) -> int:
        return self.total_rows

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.total_rows:
            raise IndexError(f'Index {idx} out of bounds for dataset length {self.total_rows}.')

        file_idx = bisect.bisect_right(self.cum_sizes, idx)
        local_idx = idx if file_idx == 0 else idx - self.cum_sizes[file_idx - 1]
        
        if file_idx not in self._table_cache:
            self._table_cache[file_idx] = pq.read_table(self.file_paths[file_idx], memory_map=True)
            
        table = self._table_cache[file_idx]
        
        content_str = table.column('content')[local_idx].as_py()
        
        record: Dict[str, Any] = {
            'content': content_str
        }
        
        if 'tag' in table.column_names:
            record['tag'] = table.column('tag')[local_idx].as_py()
            
        if self.tokenizer is not None:
            record['input_ids'] = self.tokenizer.encode(content_str)
            
        return record