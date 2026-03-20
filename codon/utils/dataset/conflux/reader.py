import torch
import os
import io
import json
import tarfile

from typing import Dict, Any, Optional, List, Iterator, TYPE_CHECKING
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

if TYPE_CHECKING:
    from .base import ConfluxDataset, Shard, Task


class ConfluxReader(IterableDataset):
    '''
    Iterable dataset for reading Conflux datasets.

    Attributes:
        dataset (ConfluxDataset): The dataset instance being read.
        manifest (Manifest): The dataset manifest.
        schema (Dict[str, SchemaItem]): The dataset schema.
        task_name (Optional[str]): The name of the active task.
        task (Optional[Task]): The active task definition.
        required_keys (set): The set of schema keys required for the task.
    '''

    def __init__(self, dataset: 'ConfluxDataset', task_name: Optional[str] = None) -> None:
        '''
        Initializes the ConfluxReader.

        Args:
            dataset (ConfluxDataset): The target dataset instance.
            task_name (Optional[str]): Optional task name to filter modalities.
        '''
        super().__init__()
        self.dataset = dataset
        self.manifest = dataset.manifest
        self.schema = self.manifest.schema
        
        self.task_name = task_name
        self.task: Optional['Task'] = None
        self.required_keys = set()
        
        if task_name:
            if task_name not in self.manifest.tasks:
                raise ValueError(f"Task '{task_name}' not found in manifest.")
            self.task = self.manifest.tasks[task_name]
            self.required_keys = set(self.task.inputs + self.task.targets)
        else:
            self.required_keys = set(self.schema.keys())

    def _get_worker_shards(self) -> List['Shard']:
        '''
        Splits the dataset shards among DataLoader workers.

        Returns:
            List[Shard]: The list of shards assigned to the current worker.
        '''
        shards = self.manifest.shards
        worker_info = get_worker_info()
        
        if worker_info is None:
            return shards
        
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        return [s for i, s in enumerate(shards) if i % num_workers == worker_id]

    def _deserialize_bytes(self, raw_bytes: bytes, extension: str) -> Any:
        '''
        Deserializes byte streams into Python or PyTorch objects based on extension.

        Args:
            raw_bytes (bytes): The raw data bytes.
            extension (str): The file extension determining the format.

        Returns:
            Any: The deserialized data object.
        '''
        ext = extension.lower()
        
        if ext == 'txt':
            return raw_bytes.decode('utf-8')
            
        elif ext == 'json':
            return json.loads(raw_bytes.decode('utf-8'))
            
        elif ext in ['png', 'jpg', 'jpeg']:
            from PIL import Image
            return Image.open(io.BytesIO(raw_bytes)).convert('RGB')
            
        elif ext == 'npy':
            import numpy as np
            return np.load(io.BytesIO(raw_bytes))
            
        elif ext in ['pt', 'pth']:
            return torch.load(io.BytesIO(raw_bytes), weights_only=True)
            
        return raw_bytes

    def _format_output(self, sample_data: Dict[str, Any]) -> Any:
        '''
        Formats the output sample according to task definitions and schema requirements.

        Args:
            sample_data (Dict[str, Any]): The raw deserialized sample data.

        Returns:
            Any: Formatted data (either a dict or a tuple of (inputs, targets)) or None if skipped.
        '''
        for key in self.required_keys:
            if key not in sample_data:
                if self.schema[key].required:
                    return None
                sample_data[key] = None

        if not self.task:
            return sample_data
            
        inputs = {k: sample_data[k] for k in self.task.inputs}
        targets = {k: sample_data[k] for k in self.task.targets}
        return inputs, targets

    def _read_from_tar(self, shard_path: str) -> Iterator[Any]:
        '''
        Streamingly extracts samples from a tar or tar.gz shard.

        Args:
            shard_path (str): Path to the tar archive.

        Yields:
            Iterator[Any]: Formatted samples.
        '''
        mode = 'r|gz' if shard_path.endswith('.gz') else 'r|'
        
        current_sample_key = None
        current_sample_data = {}
        
        with tarfile.open(shard_path, mode=mode) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                    
                parts = member.name.split('.')
                if len(parts) < 3:
                    continue
                    
                sample_key, modality, ext = parts[0], parts[1], parts[-1]
                
                if current_sample_key != sample_key:
                    if current_sample_key is not None:
                        formatted = self._format_output(current_sample_data)
                        if formatted is not None:
                            yield formatted
                            
                    current_sample_key = sample_key
                    current_sample_data = {}

                if modality not in self.required_keys:
                    continue
                    
                f = tar.extractfile(member)
                if f is not None:
                    raw_bytes = f.read()
                    current_sample_data[modality] = self._deserialize_bytes(raw_bytes, ext)
            
            if current_sample_key is not None:
                formatted = self._format_output(current_sample_data)
                if formatted is not None:
                    yield formatted

    def _read_from_uncompressed(self, shard_path: str) -> Iterator[Any]:
        '''
        Reads samples from an uncompressed directory shard.

        Args:
            shard_path (str): Path to the directory.

        Yields:
            Iterator[Any]: Formatted samples.
        '''
        files = os.listdir(shard_path)
        sample_groups = {}
        for f in files:
            parts = f.split('.')
            if len(parts) < 3:
                continue
            sample_key, modality = parts[0], parts[1]
            if modality not in self.required_keys:
                continue
            
            if sample_key not in sample_groups:
                sample_groups[sample_key] = []
            sample_groups[sample_key].append(f)
            
        for sample_key in sorted(sample_groups.keys()):
            current_sample_data = {}
            for filename in sample_groups[sample_key]:
                modality = filename.split('.')[1]
                ext = filename.split('.')[-1]
                file_path = os.path.join(shard_path, filename)
                
                with open(file_path, 'rb') as f:
                    raw_bytes = f.read()
                    current_sample_data[modality] = self._deserialize_bytes(raw_bytes, ext)
                    
            formatted = self._format_output(current_sample_data)
            if formatted is not None:
                yield formatted

    def __iter__(self) -> Iterator[Any]:
        '''
        Iterates over the dataset shards and yields samples.

        Yields:
            Iterator[Any]: Formatted samples.
        '''
        worker_shards = self._get_worker_shards()
        
        for shard in worker_shards:
            shard_path = os.path.join(self.dataset._root, shard.path)
            
            if os.path.isdir(shard_path):
                yield from self._read_from_uncompressed(shard_path)
            else:
                yield from self._read_from_tar(shard_path)

    def loader(self, batch_size: int = 1, **kwargs: Any) -> DataLoader:
        '''
        Creates a PyTorch DataLoader from this wrapped dataset.
        Note: IterableDatasets do not support the 'shuffle' argument. 
        Args:
            batch_size (int): How many samples per batch to load. Defaults to 1.
            **kwargs: Additional keyword arguments to pass to DataLoader (e.g., num_workers).
        Returns:
            DataLoader: A PyTorch DataLoader instance.
        '''
        kwargs.pop('shuffle', None)
        return DataLoader(self, batch_size=batch_size, **kwargs)