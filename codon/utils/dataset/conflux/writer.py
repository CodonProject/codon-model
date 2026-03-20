import os
import io
import json
import tarfile
from typing import Dict, Any, Optional

from .base import ConfluxDataset, SchemaItem

class ConfluxWriter:
    '''
    Writer for serializing and storing data samples into a Conflux dataset.

    Attributes:
        dataset (ConfluxDataset): The dataset instance being written to.
        config (DatasetConfig): The dataset configuration.
        schema (Dict[str, SchemaItem]): The dataset schema.
    '''

    def __init__(self, dataset: ConfluxDataset) -> None:
        '''
        Initializes the ConfluxWriter.

        Args:
            dataset (ConfluxDataset): The target dataset instance.
        '''
        self.dataset = dataset
        self.config = dataset.manifest.config
        self.schema = dataset.manifest.schema
        
        self._current_shard_idx = len(dataset.manifest.shards)
        self._current_samples = 0
        self._current_chars = 0
        self._current_bytes = 0
        
        self._global_sample_idx = dataset.manifest.statistics.total_samples
        
        self._tar_handle: Optional[tarfile.TarFile] = None
        self._current_shard_id = ''
        self._current_shard_path = ''

        if self.config.compression_mode == 'uncompressed' and self._current_shard_idx > 0:
            last_shard = dataset.manifest.shards[-1]
            
            if last_shard.samples < self.config.max_samples_per_shard:
                self._current_shard_idx -= 1
                self._current_shard_id = last_shard.shard_id
                self._current_shard_path = os.path.join(self.dataset._root, last_shard.path)
                os.makedirs(self._current_shard_path, exist_ok=True)
                
                self._current_samples = last_shard.samples
                self._current_chars = last_shard.characters
                self._current_bytes = last_shard.size_bytes
                
                self.dataset.manifest.statistics.total_shards -= 1
                self.dataset.manifest.statistics.total_samples -= last_shard.samples
                self.dataset.manifest.statistics.total_characters -= last_shard.characters
                self.dataset.manifest.statistics.total_size_bytes -= last_shard.size_bytes
                self.dataset.manifest.shards.pop()
                self.dataset.save()

    def _open_new_shard(self) -> None:
        '''
        Opens a new shard for writing based on the compression mode.
        '''
        mode = self.config.compression_mode
        self._current_shard_id = f'shard_{self._current_shard_idx:05d}'
        
        if mode in ['tar.gz', 'tar']:
            ext = '.tar.gz' if mode == 'tar.gz' else '.tar'
            filename = self._current_shard_id + ext
            self._current_shard_path = os.path.join(self.dataset._root, filename)
            write_mode = 'w:gz' if mode == 'tar.gz' else 'w'
            self._tar_handle = tarfile.open(self._current_shard_path, mode=write_mode)
            
        elif mode == 'uncompressed':
            self._current_shard_path = os.path.join(self.dataset._root, self._current_shard_id)
            os.makedirs(self._current_shard_path, exist_ok=True)
            self._tar_handle = None

        self._current_samples = 0
        self._current_chars = 0
        self._current_bytes = 0

    def _close_current_shard(self) -> None:
        '''
        Closes the current shard and updates the dataset manifest.
        '''
        if self._current_samples == 0:
            return
            
        if self._tar_handle is not None:
            self._tar_handle.close()
            self._tar_handle = None
            
        rel_path = os.path.relpath(self._current_shard_path, self.dataset._root)
        
        self.dataset.add_shard(
            shard_id=self._current_shard_id, 
            relative_path=rel_path,
            samples=self._current_samples, 
            characters=self._current_chars, 
            size_bytes=self._current_bytes
        )
        self._current_shard_idx += 1

    def _serialize_value(self, value: Any, schema_item: SchemaItem) -> bytes:
        '''
        Serializes a given value into bytes based on its type and schema.

        Args:
            value (Any): The data value to serialize.
            schema_item (SchemaItem): The schema definition for the value.

        Returns:
            bytes: The serialized byte representation of the value.
            
        Raises:
            TypeError: If the data type is unsupported for serialization.
        '''
        if isinstance(value, bytes):
            return value
            
        elif isinstance(value, str):
            encoded = value.encode('utf-8')
            self._current_chars += len(value)
            return encoded
            
        elif isinstance(value, (dict, list)):
            json_str = json.dumps(value, ensure_ascii=False)
            self._current_chars += len(json_str)
            return json_str.encode('utf-8')
            
        else:
            type_name = type(value).__name__
            type_module = type(value).__module__
            
            if 'PIL' in type_module or type_name == 'Image':
                mem_file = io.BytesIO()
                ext = schema_item.extension.lower()
                img_format = 'JPEG' if ext in ['jpg', 'jpeg'] else ext.upper()
                value.save(mem_file, format=img_format)
                return mem_file.getvalue()
                
            elif 'numpy' in type_module and type_name == 'ndarray':
                mem_file = io.BytesIO()
                import numpy as np
                np.save(mem_file, value)
                return mem_file.getvalue()
                
            elif 'torch' in type_module and type_name == 'Tensor':
                mem_file = io.BytesIO()
                if schema_item.extension == 'npy':
                    import numpy as np
                    np.save(mem_file, value.detach().cpu().numpy())
                else:
                    import torch
                    torch.save(value, mem_file)
                return mem_file.getvalue()
                
            else:
                raise TypeError(f"Unsupported data type '{type_module}.{type_name}' for extension '{schema_item.extension}'.")

    def write(self, data: Dict[str, Any], sample_key: Optional[str] = None) -> str:
        '''
        Writes a single sample dictionary to the dataset.

        Args:
            data (Dict[str, Any]): Dictionary mapping schema keys to data values.
            sample_key (Optional[str]): Explicit key for the sample.

        Returns:
            str: The sample key used for storing the data.
            
        Raises:
            ValueError: If a key in data is not defined in the schema.
        '''
        if sample_key is None:
            sample_key = f'sample_{self._global_sample_idx:07d}'
            
        if self._current_samples >= self.config.max_samples_per_shard or self._current_shard_id == '':
            if self._current_shard_id != '':
                self._close_current_shard()
            self._open_new_shard()

        for key, value in data.items():
            if key not in self.schema:
                raise ValueError(f"Key '{key}' is not defined in dataset schema.")
                
            schema_item = self.schema[key]
            filename = f'{sample_key}.{key}.{schema_item.extension}'
            
            raw_bytes = self._serialize_value(value, schema_item)
            byte_size = len(raw_bytes)
            self._current_bytes += byte_size

            if self.config.compression_mode in ['tar.gz', 'tar']:
                tarinfo = tarfile.TarInfo(name=filename)
                tarinfo.size = byte_size
                if self._tar_handle is not None:
                    self._tar_handle.addfile(tarinfo, io.BytesIO(raw_bytes))
            else:
                file_path = os.path.join(self._current_shard_path, filename)
                with open(file_path, 'wb') as f:
                    f.write(raw_bytes)
                    
        self._current_samples += 1
        self._global_sample_idx += 1
        
        return sample_key

    def __enter__(self) -> 'ConfluxWriter':
        '''
        Enters the runtime context related to this object.

        Returns:
            ConfluxWriter: The current writer instance.
        '''
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        '''
        Exits the runtime context related to this object, closing the current shard.

        Args:
            exc_type (Any): The type of the exception that caused the context to be exited.
            exc_val (Any): The instance of the exception.
            exc_tb (Any): A traceback object encoding the stack trace.
        '''
        self._close_current_shard()
