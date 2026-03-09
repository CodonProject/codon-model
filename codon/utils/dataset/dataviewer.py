'''
Data viewer module for previewing dataset fields.

This module provides utilities to inspect the structure and schema
of various dataset file formats including JSONL, Parquet, and CSV.
'''

import os
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


class DataViewer:
    '''
    A utility class for previewing and inspecting dataset files.

    Supports JSONL, Parquet, and CSV file formats. Provides methods
    to view fields, schema, and sample data.

    Attributes:
        file_path (str): Path to the dataset file.
        file_type (str): Type of the file ('jsonl', 'parquet', 'csv').
    '''

    def __init__(self, file_path: str) -> None:
        '''
        Initialize the DataViewer with a file path.

        Args:
            file_path (str): Path to the dataset file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is not supported.
        '''
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File not found: {file_path}')

        self.file_path = file_path
        self.file_type = self._detect_file_type(file_path)
        self._df: pd.DataFrame | None = None

    def _detect_file_type(self, file_path: str) -> str:
        '''
        Detect file type based on extension.

        Args:
            file_path (str): Path to the file.

        Returns:
            str: File type ('jsonl', 'parquet', or 'csv').

        Raises:
            ValueError: If file extension is not supported.
        '''
        ext = os.path.splitext(file_path)[1].lower()
        type_map = {
            '.jsonl': 'jsonl',
            '.parquet': 'parquet',
            '.csv': 'csv',
        }
        if ext not in type_map:
            raise ValueError(
                f'Unsupported file type: {ext}. '
                f'Supported types: {list(type_map.keys())}'
            )
        return type_map[ext]

    def _load_data(self, nrows: int | None = None) -> pd.DataFrame:
        '''
        Load data from file into a pandas DataFrame.

        Args:
            nrows (int | None): Number of rows to load. If None, loads all rows.

        Returns:
            pd.DataFrame: Loaded data.
        '''
        if self._df is not None and nrows is None:
            return self._df

        if self.file_type == 'jsonl':
            df = pd.read_json(self.file_path, lines=True, nrows=nrows)
        elif self.file_type == 'parquet':
            parquet_file = pq.ParquetFile(self.file_path)
            if nrows is not None:
                df = parquet_file.read_row_group(0).to_pandas()[:nrows]
            else:
                df = parquet_file.read().to_pandas()
        elif self.file_type == 'csv':
            df = pd.read_csv(self.file_path, nrows=nrows)
        else:
            raise ValueError(f'Unknown file type: {self.file_type}')

        if nrows is None:
            self._df = df
        return df

    def get_fields(self) -> list[str]:
        '''
        Get list of field names (columns) in the dataset.

        Returns:
            list[str]: List of field names.
        '''
        df = self._load_data(nrows=1)
        return list(df.columns)

    def get_schema(self) -> dict[str, str]:
        '''
        Get schema information with field names and their data types.

        Returns:
            dict[str, str]: Dictionary mapping field names to data types.
        '''
        df = self._load_data(nrows=1)
        schema = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            schema[col] = dtype
        return schema

    def preview(self, nrows: int = 5) -> pd.DataFrame:
        '''
        Preview the first N rows of the dataset.

        Args:
            nrows (int): Number of rows to preview. Default is 5.

        Returns:
            pd.DataFrame: First N rows of the dataset.
        '''
        return self._load_data(nrows=nrows).head(nrows)

    def get_stats(self) -> dict[str, Any]:
        '''
        Get basic statistics about the dataset.

        Returns:
            dict[str, Any]: Dictionary containing:
                - num_rows: Total number of rows
                - num_columns: Total number of columns
                - file_size: File size in bytes
                - memory_usage: Memory usage in bytes (if data loaded)
        '''
        df = self._load_data()
        file_size = os.path.getsize(self.file_path)
        memory_usage = df.memory_usage(deep=True).sum()

        return {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'file_size': file_size,
            'memory_usage': memory_usage,
        }

    def __repr__(self) -> str:
        '''
        Return string representation of the DataViewer.

        Returns:
            str: String representation showing file path and type.
        '''
        return f'DataViewer(file={self.file_path}, type={self.file_type})'


def preview_fields(file_path: str, nrows: int = 5) -> None:
    '''
    Convenience function to preview fields and sample data from a file.

    Args:
        file_path (str): Path to the dataset file.
        nrows (int): Number of rows to preview. Default is 5.
    '''
    viewer = DataViewer(file_path)

    print(f'File: {viewer.file_path}')
    print(f'Type: {viewer.file_type}')
    print()

    print('Schema:')
    schema = viewer.get_schema()
    for field, dtype in schema.items():
        print(f'  {field}: {dtype}')
    print()

    print(f'Preview (first {nrows} rows):')
    print(viewer.preview(nrows))
    print()

    print('Statistics:')
    stats = viewer.get_stats()
    for key, value in stats.items():
        print(f'  {key}: {value}')
