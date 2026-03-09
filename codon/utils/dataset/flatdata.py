import json
import random
import pandas as pd
import pyarrow.parquet as pq

from typing import Any, Dict, Optional, Union, Callable

from .base import CodonDataset


class FlatColumnDataset(CodonDataset):
    '''
    A dataset wrapper that provides access to a specific column of a FlatDataset.

    This class allows treating a single column of a structured dataset (like CSV,
    JSONL, or Parquet) as a standalone dataset.

    Attributes:
        dataset (FlatDataset): The underlying source dataset.
        column (str): The name of the column to access.
    '''

    def __init__(self, source: Union['FlatDataset', str], column: str, **kwargs):
        '''
        Initializes the FlatColumnDataset.

        Args:
            source (Union[FlatDataset, str]): The source dataset instance or a
                file path to create a new FlatDataset.
            column (str): The name of the column to retrieve.
            **kwargs: Additional arguments passed to FlatDataset constructor if
                source is a path.

        Raises:
            TypeError: If source is not a FlatDataset instance or a string.
            KeyError: If the specified column does not exist in the source dataset.
        '''
        if isinstance(source, str):
            self.dataset = FlatDataset(source, **kwargs)
        elif isinstance(source, FlatDataset):
            self.dataset = source
        else:
            raise TypeError("Source must be a FlatDataset instance or a file path string")
            
        self.column = column
        
        # Verify column exists
        if self.column not in self.dataset.fields:
             raise KeyError(f"Column '{self.column}' not found in dataset fields: {self.dataset.fields}")

    def __len__(self) -> int:
        '''
        Returns the length of the dataset.

        Returns:
            int: The number of rows in the dataset.
        '''
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        '''
        Retrieves the value of the specified column at the given index.

        Args:
            idx (int): The index of the row.

        Returns:
            Any: The value in the configured column at the given index.
        '''
        return self.dataset.get_value(idx, self.column)

    def to_flat_dataset(self) -> 'MappedFlatDataset':
        '''
        Converts the column data to a FlatDataset.

        This method extracts the column values and creates a new MappedFlatDataset
        that treats each dictionary value as a row. Supports chaining for nested
        dictionary structures.

        Returns:
            MappedFlatDataset: A new dataset where each row is the dictionary value
                from the specified column.

        Raises:
            TypeError: If any value in the column is not a dictionary.
        '''
        def extract_column_as_row(row: Dict[str, Any]) -> Dict[str, Any]:
            '''
            Extracts the column value from a row and validates it is a dict.

            Args:
                row (Dict[str, Any]): The source row.

            Returns:
                Dict[str, Any]: The column value (must be a dictionary).

            Raises:
                TypeError: If the column value is not a dictionary.
            '''
            col_value = row[self.column]
            if not isinstance(col_value, dict):
                raise TypeError(
                    f"Column '{self.column}' value must be a dictionary, "
                    f"got {type(col_value).__name__}"
                )
            return col_value

        return MappedFlatDataset(
            parent_dataset=self.dataset,
            map_fn=extract_column_as_row,
            in_memory=self.dataset.in_memory
        )


class FlatDataset(CodonDataset):
    '''
    A dataset implementation for flat file formats (JSONL, CSV, Parquet).

    This class supports both in-memory loading (for smaller datasets) and
    lazy loading (for larger datasets) to efficiently handle data access.

    Attributes:
        path (str): The file path to the dataset.
        in_memory (bool): Whether to load the entire dataset into memory.
        shuffle (bool): Whether to shuffle the data indices.
    '''

    def __init__(
        self,
        path: str,
        in_memory: bool = False,
        shuffle: bool = False
    ):
        '''
        Initializes the FlatDataset.

        Args:
            path (str): The file path to the dataset (supports .jsonl, .csv, .parquet).
            in_memory (bool): If True, loads all data into memory. If False,
                uses lazy loading (offsets for text files, row groups for Parquet).
                Defaults to False.
            shuffle (bool): If True, shuffles the access indices. Defaults to False.
        '''
        self.path = path
        self.in_memory = in_memory
        self.shuffle = shuffle
        self._data = []
        self._offsets = []
        self._indices = []
        self._file_type = self._detect_file_type(path)
        self._length = 0
        self._columns = []
        
        # Parquet specific
        self._pq_file = None
        self._pq_meta = None

        if self.in_memory:
            self._load_all()
        else:
            self._setup_lazy_loading()
        
        if self.shuffle:
            self._indices = list(range(self._length))
            random.shuffle(self._indices)
        else:
            self._indices = range(self._length)

    def _detect_file_type(self, path: str) -> str:
        '''
        Detects the file type based on the file extension.

        Args:
            path (str): The file path.

        Returns:
            str: The detected file type ('jsonl', 'csv', or 'parquet').

        Raises:
            ValueError: If the file extension is not supported.
        '''
        if path.endswith('.jsonl'):
            return 'jsonl'
        elif path.endswith('.csv'):
            return 'csv'
        elif path.endswith('.parquet'):
            return 'parquet'
        else:
            raise ValueError(f"Unsupported file type: {path}")

    def _load_all(self):
        '''
        Loads the entire dataset into memory.
        '''
        if self._file_type == 'jsonl':
            with open(self.path, 'r', encoding='utf-8') as f:
                self._data = [json.loads(line) for line in f]
            if self._data:
                self._columns = list(self._data[0].keys())
        elif self._file_type == 'csv':
            df = pd.read_csv(self.path)
            self._data = df.to_dict('records')
            self._columns = df.columns.tolist()
        elif self._file_type == 'parquet':
            df = pd.read_parquet(self.path)
            self._data = df.to_dict('records')
            self._columns = df.columns.tolist()
        
        self._length = len(self._data)

    def _setup_lazy_loading(self):
        '''
        Sets up lazy loading by calculating file offsets or metadata.
        '''
        if self._file_type == 'jsonl':
            with open(self.path, 'rb') as f:
                offset = 0
                for line in f:
                    self._offsets.append(offset)
                    offset += len(line)
            self._length = len(self._offsets)
            # Peek first line for columns
            if self._length > 0:
                with open(self.path, 'r', encoding='utf-8') as f:
                    self._columns = list(json.loads(f.readline()).keys())
                    
        elif self._file_type == 'csv':
            with open(self.path, 'rb') as f:
                # Read header
                header_line = f.readline()
                self._columns = header_line.decode('utf-8').strip().split(',')
                offset = len(header_line)
                while True:
                    current_offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    self._offsets.append(current_offset)
            self._length = len(self._offsets)
            
        elif self._file_type == 'parquet':
            self._pq_file = pq.ParquetFile(self.path)
            self._pq_meta = self._pq_file.metadata
            self._length = self._pq_meta.num_rows
            self._columns = self._pq_file.schema.names

    @property
    def fields(self) -> list[str]:
        '''
        Returns the list of column names in the dataset.

        Returns:
            list[str]: A list of column names.
        '''
        return self._columns

    def __len__(self) -> int:
        '''
        Returns the number of rows in the dataset.

        Returns:
            int: The total number of rows.
        '''
        return self._length

    def __getitem__(self, idx: Union[int, str]) -> Union[Dict[str, Any], 'FlatColumnDataset']:
        '''
        Retrieves a row by index or a column wrapper by name.

        Args:
            idx (Union[int, str]): The row index (int) or column name (str).

        Returns:
            Union[Dict[str, Any], FlatColumnDataset]: If idx is an int, returns
            the row data as a dictionary. If idx is a str, returns a
            FlatColumnDataset wrapper for that column.

        Raises:
            TypeError: If idx is not an int or str.
            IndexError: If the numeric index is out of range.
        '''
        if isinstance(idx, str):
            return FlatColumnDataset(self, idx)
        
        if not isinstance(idx, int):
            raise TypeError(f"Index must be int or str, got {type(idx)}")

        if idx < 0 or idx >= self._length:
            raise IndexError("Index out of range")
        
        real_idx = self._indices[idx]
        return self.get_value(real_idx)

    def get_value(self, idx: int, column: Optional[str] = None) -> Any:
        '''
        Retrieves the value at the specified index, optionally for a specific column.

        Args:
            idx (int): The real index of the row (after shuffling logic).
            column (Optional[str]): The specific column name to retrieve.
                If None, returns the entire row.

        Returns:
            Any: The value of the column or the full row dictionary.

        Raises:
            RuntimeError: If an unexpected file type handling path is reached.
        '''
        # Handle in-memory access
        if self.in_memory:
            row = self._data[idx]
            if column:
                return row[column]
            return row

        # Handle lazy loading
        if self._file_type == 'jsonl':
            with open(self.path, 'r', encoding='utf-8') as f:
                f.seek(self._offsets[idx])
                row = json.loads(f.readline())
                if column:
                    return row[column]
                return row
                
        elif self._file_type == 'csv':
            with open(self.path, 'r', encoding='utf-8') as f:
                f.seek(self._offsets[idx])
                line = f.readline()
                # Simple CSV parsing
                values = line.strip().split(',')
                row = dict(zip(self._columns, values))
                if column:
                    return row[column]
                return row
                
        elif self._file_type == 'parquet':
            # Map idx to row group
            row_group_index = 0
            row_in_group = idx
            for i in range(self._pq_file.num_row_groups):
                num_rows = self._pq_meta.row_group(i).num_rows
                if row_in_group < num_rows:
                    row_group_index = i
                    break
                row_in_group -= num_rows
            
            # Optimization: Read specific column only if requested
            cols_to_read = [column] if column else self._columns
            
            # Read just the necessary columns from that row group
            table = self._pq_file.read_row_group(row_group_index, columns=cols_to_read)
            
            if column:
                # If single column requested, return the value directly
                return table.column(column)[row_in_group].as_py()
            else:
                # Return full row dict
                row_data = {}
                for col_name in self._columns:
                    val = table.column(col_name)[row_in_group].as_py()
                    row_data[col_name] = val
                return row_data
        
        raise RuntimeError("Should not reach here")


class MappedFlatDataset(FlatDataset):
    '''
    A dataset that applies a mapping function to rows from a parent FlatDataset.

    This class enables lazy transformation of data without materializing all rows
    into memory. It supports chaining transformations for nested data structures.

    Attributes:
        parent_dataset (FlatDataset): The source dataset to read from.
        map_fn (Callable): Function that transforms each row.
        in_memory (bool): Whether to cache mapped data in memory.
    '''

    def __init__(
        self,
        parent_dataset: FlatDataset,
        map_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        in_memory: bool = False,
        shuffle: bool = False
    ):
        '''
        Initializes the MappedFlatDataset.

        Args:
            parent_dataset (FlatDataset): The source dataset to read from.
            map_fn (Callable): Function that transforms each row dictionary.
                Should accept a dict and return a dict.
            in_memory (bool): If True, caches all mapped rows in memory.
                Defaults to False.
            shuffle (bool): If True, shuffles the access indices. Defaults to False.
        '''
        self.parent_dataset = parent_dataset
        self.map_fn = map_fn
        self.in_memory = in_memory
        self.shuffle = shuffle
        self._data = []
        self._indices = []
        self._length = parent_dataset._length
        self._columns = []
        self._file_type = 'mapped'

        # Peek first row to determine columns
        if self._length > 0:
            first_row = self.map_fn(parent_dataset.get_value(0))
            self._columns = list(first_row.keys())

        if self.in_memory:
            self._load_all_mapped()

        if self.shuffle:
            self._indices = list(range(self._length))
            random.shuffle(self._indices)
        else:
            self._indices = range(self._length)

    def _load_all_mapped(self):
        '''
        Loads all mapped rows into memory.
        '''
        self._data = []
        for i in range(self._length):
            row = self.parent_dataset.get_value(i)
            mapped_row = self.map_fn(row)
            self._data.append(mapped_row)

    def get_value(self, idx: int, column: Optional[str] = None) -> Any:
        '''
        Retrieves the value at the specified index, optionally for a specific column.

        Args:
            idx (int): The real index of the row.
            column (Optional[str]): The specific column name to retrieve.
                If None, returns the entire row.

        Returns:
            Any: The value of the column or the full row dictionary.
        '''
        if self.in_memory:
            row = self._data[idx]
            if column:
                return row[column]
            return row

        # Lazy loading: fetch from parent and apply mapping
        parent_row = self.parent_dataset.get_value(idx)
        mapped_row = self.map_fn(parent_row)
        if column:
            return mapped_row[column]
        return mapped_row
