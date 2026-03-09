import json
import os
import csv

from enum        import Enum, auto
from dataclasses import dataclass
from datetime    import datetime
from typing      import Union

import pandas as pd

from .base     import CodonDataset
from .flatdata import FlatDataset

from codon.utils.safecode import safecode

class FileType(Enum):
    PARQUET = auto()
    JSONL   = auto()
    CSV     = auto()

@dataclass
class CorpusData:
    '''
    Represents a single corpus data entry with content, token count, and UUID.

    Attributes:
        content (str): The text content.
        num_token (int): The number of tokens (characters) in the content.
        uuid (str): Unique identifier for this entry.
    '''
    content: str
    num_token: int
    uuid: str

class CorpusDataset(CodonDataset):
    '''
    A dataset for managing linguistic corpora with token counting.

    This class maintains a folder of corpus files (PARQUET, JSONL, CSV) and
    tracks metadata in a JSON configuration file. Token count is calculated
    based on character count. Supports Key-Value access with both string keys
    (filename:row_number) and integer keys (global row index).

    Attributes:
        folder_path (str): Path to the folder containing corpus files.
        config_path (str): Path to the configuration JSON file.
        _config (dict): Configuration dictionary loaded from JSON.
        _total_token (int): Total token count across all files.
        _file_index (list): List of file metadata for index mapping.
        _cumulative_rows (list): Cumulative row counts for index mapping.
    '''

    def __init__(
        self,
        folder_path: str,
        file_type: FileType = None,
        file_limit: int = 2 * 1024 * 1024 * 1024
    ) -> None:
        '''
        Initializes the CorpusDataset.

        Args:
            folder_path (str): Path to the folder containing corpus files.
            file_type (FileType): The file type for storing data. If not provided,
                will attempt to read from config.json. Required for new datasets.
            file_limit (int): Maximum file size in bytes. Defaults to 2GB.

        Raises:
            ValueError: If file_type is not provided and config.json does not exist.
        '''
        self.folder_path = folder_path
        self.config_path = os.path.join(folder_path, 'config.json')
        self.file_limit = file_limit
        self._config: dict = {}
        self._total_token: int = 0
        self._file_index: list = []
        self._cumulative_rows: list = [0]
        self._current_file_size: int = 0
        self._current_file_idx: int = 0
        self.file_type: FileType = file_type
        self._flat_dataset_cache: dict = {}
        self._parquet_buffer: list = []
        self._file_name_to_idx: dict = {}

        # Create folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        # Load or initialize config
        self._load_config()

        # Determine file_type
        if self.file_type is None:
            if 'file_type' in self._config:
                self.file_type = FileType[self._config['file_type']]
            else:
                raise ValueError(
                    'file_type must be specified when creating a new CorpusDataset'
                )

        # Store file_type in config
        self._config['file_type'] = self.file_type.name
        self._save_config()

    def _load_config(self) -> None:
        '''
        Loads configuration from JSON file or creates a new one.
        '''
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
        else:
            self._config = {
                'version': '1.0',
                'total_token': 0,
                'files': []
            }
            self._save_config()

        # Rebuild file index and cumulative rows
        self._rebuild_index()

    def _save_config(self) -> None:
        '''
        Saves configuration to JSON file.
        '''
        self._config['total_token'] = self._total_token
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)

    def _rebuild_index(self) -> None:
        '''
        Rebuilds the file index and cumulative row counts.
        '''
        self._file_index = []
        self._cumulative_rows = [0]
        self._total_token = 0
        self._file_name_to_idx = {}

        for file_info in self._config.get('files', []):
            self._file_index.append(file_info)
            self._file_name_to_idx[file_info['filename']] = len(self._file_index) - 1
            self._total_token += file_info.get('num_token', 0)
            self._cumulative_rows.append(
                self._cumulative_rows[-1] + file_info.get('num_rows', 0)
            )

    def _detect_file_type(self, file_path: str) -> FileType:
        '''
        Detects file type based on extension.

        Args:
            file_path (str): Path to the file.

        Returns:
            FileType: The detected file type.

        Raises:
            ValueError: If file type is not supported.
        '''
        if file_path.endswith('.parquet'):
            return FileType.PARQUET
        elif file_path.endswith('.jsonl'):
            return FileType.JSONL
        elif file_path.endswith('.csv'):
            return FileType.CSV
        else:
            raise ValueError(f'Unsupported file type: {file_path}')

    def _count_tokens(self, content: str) -> int:
        '''
        Counts tokens in content based on character count.

        Args:
            content (str): The text content.

        Returns:
            int: The number of tokens (characters).
        '''
        return len(content)

    def _load_file_data(self, file_path: str, file_type: FileType) -> list:
        '''
        Loads data from a file.

        Args:
            file_path (str): Path to the file.
            file_type (FileType): Type of the file.

        Returns:
            list: List of dictionaries representing rows.

        Raises:
            ValueError: If file type is not supported.
        '''
        if file_type == FileType.PARQUET:
            df = pd.read_parquet(file_path)
            return df.to_dict('records')
        elif file_type == FileType.JSONL:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        elif file_type == FileType.CSV:
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        else:
            raise ValueError(f'Unsupported file type: {file_type}')

    def add(self, data: str) -> None:
        '''
        Adds a string data entry to the dataset. If adding the data would exceed
        the file size limit, automatically creates a new file.

        Args:
            data (str): The text content to add.
        '''
        # Calculate token count (character count)
        num_token = self._count_tokens(data)
        data_size = len(data.encode('utf-8'))

        # Check if adding this data would exceed limit
        if self._current_file_size + data_size > self.file_limit and len(self._config['files']) > 0:
            # Flush parquet buffer before starting new file
            if self.file_type == FileType.PARQUET and self._parquet_buffer:
                current_filename = f'corpus_{self._current_file_idx:03d}.{self._get_file_extension()}'
                current_file_path = os.path.join(self.folder_path, current_filename)
                self._flush_parquet_buffer(current_file_path)
            # Start a new file
            self._current_file_idx += 1
            self._current_file_size = 0

        # Get or create current file
        current_filename = f'corpus_{self._current_file_idx:03d}.{self._get_file_extension()}'
        current_file_path = os.path.join(self.folder_path, current_filename)

        # Generate UUID for this entry
        entry_uuid = safecode(8)

        # Prepare row data
        row_data = {'content': data, 'uuid': entry_uuid}

        # Append to file
        if self.file_type == FileType.JSONL:
            with open(current_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(row_data, ensure_ascii=False) + '\n')
        elif self.file_type == FileType.CSV:
            file_exists = os.path.exists(current_file_path)
            with open(current_file_path, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['content', 'uuid'])
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_data)
        elif self.file_type == FileType.PARQUET:
            # For parquet, accumulate in memory and write periodically
            self._parquet_buffer.append(row_data)
            # Write to parquet every 1000 rows or when buffer is large
            if len(self._parquet_buffer) >= 1000:
                self._flush_parquet_buffer(current_file_path)

        # Update file info
        self._update_file_info(current_filename, num_token, data_size)

    def _get_file_extension(self) -> str:
        '''
        Gets the file extension based on file_type.

        Returns:
            str: The file extension without the dot.

        Raises:
            ValueError: If file type is not supported.
        '''
        if self.file_type == FileType.PARQUET:
            return 'parquet'
        elif self.file_type == FileType.JSONL:
            return 'jsonl'
        elif self.file_type == FileType.CSV:
            return 'csv'
        else:
            raise ValueError(f'Unsupported file type: {self.file_type}')

    def _flush_parquet_buffer(self, file_path: str) -> None:
        '''
        Flushes accumulated parquet data to file.

        Args:
            file_path (str): Path to the parquet file.
        '''
        if not self._parquet_buffer:
            return

        df = pd.DataFrame(self._parquet_buffer)
        if os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        df.to_parquet(file_path, index=False)
        self._parquet_buffer = []

    def _update_file_info(self, filename: str, num_token: int, data_size: int) -> None:
        '''
        Updates file information in config.

        Args:
            filename (str): The filename.
            num_token (int): Number of tokens added.
            data_size (int): Size of data in bytes.
        '''
        # Find or create file info
        file_info = None
        for info in self._config['files']:
            if info['filename'] == filename:
                file_info = info
                break

        if file_info is None:
            file_info = {
                'filename': filename,
                'file_type': self.file_type.name,
                'num_rows': 0,
                'num_token': 0,
                'created_at': datetime.now().isoformat()
            }
            self._config['files'].append(file_info)

        file_info['num_rows'] += 1
        file_info['num_token'] += num_token
        self._total_token += num_token
        self._current_file_size += data_size

        # Save config and rebuild index
        self._save_config()
        self._rebuild_index()

    def add_from_file(self, file_path: str, fields: list[str]) -> None:
        '''
        Adds data from a file by reading specified fields and concatenating them.

        Args:
            file_path (str): Path to the source file.
            fields (list[str]): List of field names to read and concatenate.
                If a field doesn't exist, it is skipped.

        Raises:
            FileNotFoundError: If the file does not exist.
        '''
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File not found: {file_path}')

        file_type = self._detect_file_type(file_path)
        data = self._load_file_data(file_path, file_type)

        # Process each row
        for row in data:
            # Concatenate specified fields
            content_parts = []
            for field in fields:
                if field in row:
                    content_parts.append(str(row[field]))

            if content_parts:
                concatenated_content = ''.join(content_parts)
                self.add(concatenated_content)

    def _get_or_create_flat_dataset(self, file_path: str) -> FlatDataset:
        '''
        Gets or creates a FlatDataset instance for lazy loading.

        This method caches FlatDataset instances to avoid recreating them
        for repeated access to the same file.

        Args:
            file_path (str): Path to the corpus file.

        Returns:
            FlatDataset: A FlatDataset instance for lazy loading the file.
        '''
        if file_path not in self._flat_dataset_cache:
            self._flat_dataset_cache[file_path] = FlatDataset(
                file_path,
                in_memory=False,
                shuffle=False
            )
        return self._flat_dataset_cache[file_path]

    def get(self, key: Union[int, str]) -> CorpusData:
        '''
        Retrieves a corpus data entry by key using lazy loading.

        Args:
            key (Union[int, str]): The key to retrieve. Can be:
                - int: Global row index (0, 1, 2, ...)
                - str: "filename:row_number" format (e.g., "corpus_001.parquet:5")

        Returns:
            CorpusData: The corpus data at the specified key.

        Raises:
            IndexError: If index is out of range.
            KeyError: If string key format is invalid.
        '''
        # Handle string key format: "filename:row_number"
        if isinstance(key, str):
            if ':' not in key:
                raise KeyError(f'Invalid key format: {key}. Expected "filename:row_number"')
            filename, row_str = key.rsplit(':', 1)
            try:
                row_in_file = int(row_str)
            except ValueError:
                raise KeyError(f'Invalid row number in key: {key}')

            # Find file by filename using optimized lookup
            if filename not in self._file_name_to_idx:
                raise KeyError(f'File not found: {filename}')

            file_idx = self._file_name_to_idx[filename]
            file_info = self._file_index[file_idx]

            if row_in_file < 0 or row_in_file >= file_info['num_rows']:
                raise IndexError(f'Row {row_in_file} out of range for {filename}')
        else:
            # Handle integer key: global row index
            if key < 0 or key >= self._cumulative_rows[-1]:
                raise IndexError(f'Index {key} out of range')

            # Find which file contains this index
            file_idx = 0
            for i, cumulative_row in enumerate(self._cumulative_rows[1:], 1):
                if key < cumulative_row:
                    file_idx = i - 1
                    break

            row_in_file = key - self._cumulative_rows[file_idx]
            file_info = self._file_index[file_idx]

        # Use FlatDataset for lazy loading
        file_path = os.path.join(self.folder_path, file_info['filename'])
        flat_dataset = self._get_or_create_flat_dataset(file_path)
        row = flat_dataset.get_value(row_in_file)

        # Find content column (try common names)
        content_column = None
        for col_name in ['content', 'text', 'data', 'corpus']:
            if col_name in row:
                content_column = col_name
                break

        if content_column is None:
            # Use first column if no standard name found
            content_column = list(row.keys())[0]

        content = str(row[content_column])
        num_token = self._count_tokens(content)
        uuid = str(row.get('uuid', ''))

        return CorpusData(content=content, num_token=num_token, uuid=uuid)

    def __len__(self) -> int:
        '''
        Returns the total number of entries in the dataset.

        Returns:
            int: The total number of entries.
        '''
        return self._cumulative_rows[-1] if self._cumulative_rows else 0

    def __getitem__(self, key: Union[int, str]) -> CorpusData:
        '''
        Retrieves an entry by key using bracket notation.

        Args:
            key (Union[int, str]): The key to retrieve (int index or str "filename:row").

        Returns:
            CorpusData: The corpus data at the specified key.
        '''
        return self.get(key)
