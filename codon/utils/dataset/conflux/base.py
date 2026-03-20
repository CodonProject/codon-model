import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .writer import ConfluxWriter
    from .reader import ConfluxReader

@dataclass
class DatasetInfo:
    '''
    Information about the dataset.

    Attributes:
        name (str): Name of the dataset.
        version (str): Version of the dataset.
        description (str): Description of the dataset.
        creation_date (str): Date of creation.
    '''
    name: str = 'unnamed_dataset'
    version: str = '0.0.1'
    description: str = ''
    creation_date: str = ''

@dataclass
class DatasetConfig:
    '''
    Configuration for dataset storage and sharding.

    Attributes:
        compression_mode (str): Compression mode ('tar.gz', 'tar', 'uncompressed').
        max_samples_per_shard (int): Maximum number of samples allowed per shard.
    '''
    compression_mode: str = 'tar.gz'
    max_samples_per_shard: int = 10000

@dataclass
class Statistics:
    '''
    Dataset statistics.

    Attributes:
        total_samples (int): Total number of samples in the dataset.
        total_shards (int): Total number of shards.
        total_size_bytes (int): Total size in bytes.
        total_characters (int): Total characters in text fields.
    '''
    total_samples: int = 0
    total_shards: int = 0
    total_size_bytes: int = 0
    total_characters: int = 0

@dataclass
class SchemaItem:
    '''
    Definition of a single data field in the dataset schema.

    Attributes:
        modality (str): Modality of the data.
        extension (str): File extension associated with the data.
        decoder (str): Decoder to use for loading the data.
        required (bool): Whether this field is required for all samples.
        dtype (Optional[str]): Data type, if applicable.
        shape (Optional[List[int]]): Shape of the data, if applicable.
        description (str): Description of the field.
    '''
    modality: str
    extension: str
    decoder: str = 'auto'
    required: bool = True
    dtype: Optional[str] = None
    shape: Optional[List[int]] = None
    description: str = ''

@dataclass
class Task:
    '''
    Definition of a machine learning task supported by the dataset.

    Attributes:
        inputs (List[str]): List of schema keys used as inputs.
        targets (List[str]): List of schema keys used as targets.
        description (str): Description of the task.
        type (str): Type of the task.
    '''
    inputs: List[str]
    targets: List[str]
    description: str = ''
    type: str = 'general'

@dataclass
class Shard:
    '''
    Information about a single data shard.

    Attributes:
        shard_id (str): Unique identifier for the shard.
        path (str): Relative path to the shard file or directory.
        samples (int): Number of samples in the shard.
        characters (int): Number of text characters in the shard.
        size_bytes (int): Size of the shard in bytes.
        checksum (Optional[str]): Checksum of the shard for integrity verification.
    '''
    shard_id: str
    path: str
    samples: int
    characters: int = 0
    size_bytes: int = 0
    checksum: Optional[str] = None

@dataclass
class Manifest:
    '''
    The root metadata structure of the dataset.

    Attributes:
        dataset_info (DatasetInfo): General information.
        config (DatasetConfig): Dataset configuration.
        statistics (Statistics): Dataset statistics.
        schema (Dict[str, SchemaItem]): Dataset schema definitions.
        tasks (Dict[str, Task]): Defined tasks.
        shards (List[Shard]): List of shards.
    '''
    dataset_info: DatasetInfo = field(default_factory=DatasetInfo)
    config: DatasetConfig = field(default_factory=DatasetConfig)
    statistics: Statistics = field(default_factory=Statistics)
    schema: Dict[str, SchemaItem] = field(default_factory=dict)
    tasks: Dict[str, Task] = field(default_factory=dict)
    shards: List[Shard] = field(default_factory=list)


class ConfluxDataset:
    '''
    Main class for managing a Conflux dataset.

    Attributes:
        _root (str): Root path of the dataset.
        _manifest_path (str): Path to the manifest.json file.
        manifest (Manifest): The dataset manifest object.
    '''

    def __init__(self, root_path: str) -> None:
        '''
        Initializes the ConfluxDataset.

        Args:
            root_path (str): The root directory where the dataset is stored.
        '''
        self._root = root_path
        self._manifest_path = os.path.join(root_path, 'manifest.json')
        self.manifest = Manifest()
        
        if not os.path.exists(root_path):
            os.makedirs(root_path, exist_ok=True)
            self.save()
        elif os.path.exists(self._manifest_path):
            self._load()
    
    def _load(self) -> None:
        '''
        Loads the dataset manifest from the manifest.json file.
        '''
        with open(self._manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.manifest.dataset_info = DatasetInfo(**{k: v for k, v in data.get('dataset_info', {}).items() if k in DatasetInfo.__annotations__})
        self.manifest.config = DatasetConfig(**{k: v for k, v in data.get('config', {}).items() if k in DatasetConfig.__annotations__})
        self.manifest.statistics = Statistics(**{k: v for k, v in data.get('statistics', {}).items() if k in Statistics.__annotations__})

        for k, v in data.get('schema', {}).items():
            self.manifest.schema[k] = SchemaItem(**{ik: iv for ik, iv in v.items() if ik in SchemaItem.__annotations__})
        for k, v in data.get('tasks', {}).items():
            self.manifest.tasks[k] = Task(**{ik: iv for ik, iv in v.items() if ik in Task.__annotations__})
            
        self.manifest.shards = [Shard(**{ik: iv for ik, iv in s.items() if ik in Shard.__annotations__}) for s in data.get('shards', [])]
        
        self.save()

    def save(self) -> 'ConfluxDataset':
        '''
        Saves the current manifest state to the manifest.json file.

        Returns:
            ConfluxDataset: The current dataset instance.
        '''
        with open(self._manifest_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.manifest), f, indent=4, ensure_ascii=False)
        return self
    
    def set_compression_mode(self, compression_mode: str) -> 'ConfluxDataset':
        '''
        Sets the compression mode for the dataset.

        Args:
            compression_mode (str): The compression mode to use.

        Returns:
            ConfluxDataset: The current dataset instance.
            
        Raises:
            AssertionError: If an invalid compression mode is provided.
        '''
        assert compression_mode in ['tar.gz', 'tar', 'uncompressed'], 'Invalid compression mode.'
        self.manifest.config.compression_mode = compression_mode
        self.save()
        return self

    def set_info(self, name: str, version: str = '0.0.1', description: str = '', compression_mode: str = 'tar.gz', max_samples_per_shard: int = 10000) -> 'ConfluxDataset':
        '''
        Sets general information and configuration for the dataset.

        Args:
            name (str): Name of the dataset.
            version (str): Version of the dataset.
            description (str): Description of the dataset.
            compression_mode (str): Compression mode to use.
            max_samples_per_shard (int): Maximum samples per shard.

        Returns:
            ConfluxDataset: The current dataset instance.
            
        Raises:
            AssertionError: If an invalid compression mode is provided.
        '''
        assert compression_mode in ['tar.gz', 'tar', 'uncompressed'], 'Invalid compression mode.'
        self.manifest.config.compression_mode = compression_mode
        self.manifest.config.max_samples_per_shard = max_samples_per_shard
        self.manifest.dataset_info.name = name
        self.manifest.dataset_info.version = version
        self.manifest.dataset_info.description = description
        self.save()
        return self

    def add_schema(self, key_name: str, modality: str, extension: str, required: bool = True, **kwargs) -> 'ConfluxDataset':
        '''
        Adds a new schema item to the dataset.

        Args:
            key_name (str): The key name for the data field.
            modality (str): The modality of the data.
            extension (str): The file extension for the data.
            required (bool): Whether the field is required.
            **kwargs: Additional properties for the SchemaItem.

        Returns:
            ConfluxDataset: The current dataset instance.
        '''
        valid_kwargs = {k: v for k, v in kwargs.items() if k in SchemaItem.__annotations__}
        item = SchemaItem(modality=modality, extension=extension, required=required, **valid_kwargs)
        self.manifest.schema[key_name] = item
        self.save()
        return self

    def add_task(self, task_name: str, inputs: List[str], targets: List[str], description: str = '', task_type: str = 'general') -> 'ConfluxDataset':
        '''
        Adds a new task definition to the dataset.

        Args:
            task_name (str): Name of the task.
            inputs (List[str]): List of schema keys used as inputs.
            targets (List[str]): List of schema keys used as targets.
            description (str): Description of the task.
            task_type (str): Type of the task.

        Returns:
            ConfluxDataset: The current dataset instance.

        Raises:
            ValueError: If an input or target key is not defined in the schema.
        '''
        for k in inputs + targets:
            if k not in self.manifest.schema:
                raise ValueError(f"Key '{k}' used in task '{task_name}' is not defined in schema.")
        
        task = Task(inputs=inputs, targets=targets, description=description, type=task_type)
        self.manifest.tasks[task_name] = task
        self.save()
        return self

    def add_shard(self, shard_id: str, relative_path: str, samples: int, characters: int = 0, size_bytes: int = 0) -> 'ConfluxDataset':
        '''
        Records a new shard in the dataset manifest and updates statistics.

        Args:
            shard_id (str): Unique identifier for the shard.
            relative_path (str): Relative path to the shard.
            samples (int): Number of samples in the shard.
            characters (int): Number of text characters in the shard.
            size_bytes (int): Size of the shard in bytes.

        Returns:
            ConfluxDataset: The current dataset instance.
        '''
        shard = Shard(shard_id=shard_id, path=relative_path, samples=samples, characters=characters, size_bytes=size_bytes)
        self.manifest.shards.append(shard)
        self.manifest.statistics.total_shards = len(self.manifest.shards)
        self.manifest.statistics.total_samples += samples
        self.manifest.statistics.total_characters += characters
        self.manifest.statistics.total_size_bytes += size_bytes
        self.save()
        return self
    
    def open(self) -> 'ConfluxWriter':
        '''
        Opens a writer for the dataset to append new data.

        Returns:
            ConfluxWriter: A writer instance associated with this dataset.
        '''
        from .writer import ConfluxWriter
        return ConfluxWriter(self)
    
    def compose(self, task_name: str=None) -> 'ConfluxReader':
        from .reader import ConfluxReader
        return ConfluxReader(self, task_name=task_name)