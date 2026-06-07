# Dataset Documentation

## Overview

Codon provides flexible dataset classes for various data formats and access patterns.

## Base Classes

### CodonDataset

Base class for map-style datasets.

```python
class CodonDataset(CodonBasicDataset):
    @property
    def row(self) -> int
    
    def __len__(self) -> int
    def __getitem__(self, idx: Any) -> Any
    
    def compose(
        self,
        collate_fn: Optional[Callable] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        seek: int = 0
    ) -> TorchDatasetWrapper
```

#### Example Usage

```python
from codon.utils.data import CodonDataset

class MyDataset(CodonDataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

dataset = MyDataset([1, 2, 3, 4, 5])
wrapper = dataset.compose(shuffle=True)
dataloader = wrapper.loader(batch_size=2)
```

---

### CodonIterableDataset

Base class for iterable-style datasets.

```python
class CodonIterableDataset(CodonBasicDataset):
    def iter_from(self, offset: int) -> Iterator[Any]
    def __iter__(self) -> Iterator[Any]
    
    def compose(
        self,
        collate_fn: Optional[Callable] = None,
        seek: int = 0
    ) -> TorchIterableDatasetWrapper
```

---

## Flat File Datasets

### FlatDataset

Dataset for JSONL, CSV, Parquet files.

#### Constructor

```python
FlatDataset(
    path: str,
    in_memory: bool = False,
    shuffle: bool = False
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| path | str | - | File path (.jsonl, .csv, .parquet) |
| in_memory | bool | False | Load all into memory |
| shuffle | bool | False | Shuffle access order |

#### Example Usage

```python
from codon.utils.data import FlatDataset

# JSONL dataset
dataset = FlatDataset('data.jsonl', in_memory=True)
row = dataset[0]  # Returns dict
print(row.keys())

# Access column
text_column = dataset['text']  # Returns FlatColumnDataset

# Create DataLoader
wrapper = dataset.compose(shuffle=True)
loader = wrapper.loader(batch_size=32)
```

---

### FlatColumnDataset

Access a single column as a dataset.

```python
from codon.utils.data import FlatDataset

dataset = FlatDataset('data.jsonl')
texts = dataset['text']  # Get text column

for text in texts:
    print(text)
```

---

### MappedFlatDataset

Apply transformation to rows.

```python
from codon.utils.data import FlatDataset, MappedFlatDataset

def transform(row):
    return {
        'input': row['text'].lower(),
        'label': row['category']
    }

dataset = FlatDataset('data.jsonl')
mapped = MappedFlatDataset(dataset, transform, in_memory=True)
```

---

## Image Datasets

### ImageDataset

Dataset for image files with optional manifest.

#### Constructor

```python
ImageDataset(
    path: Union[str, Path],
    transforms: Optional[Compose] = None,
    extensions: Optional[Tuple[str, ...]] = None,
    loader: Optional[Callable] = None,
    return_path: bool = False,
    manifest_path: Optional[Union[str, Path]] = None,
    cache_metadata: bool = False
)
```

#### Example Usage

```python
from codon.utils.data import ImageDataset
from torchvision.transforms import Compose, Resize, ToTensor

transforms = Compose([
    Resize((224, 224)),
    ToTensor()
])

dataset = ImageDataset(
    path='./images/',
    transforms=transforms,
    cache_metadata=True
)

item = dataset[0]
print(f"Image shape: {item.image.shape}")
print(f"Label: {item.label}")

# Get statistics
stats = dataset.get_statistics(sample_size=1000)
print(f"Mean: {stats['mean']}")
print(f"Std: {stats['std']}")
```

---

### TarImageDataset

Dataset for images in TAR archives.

```python
from codon.utils.data import TarImageDataset

dataset = TarImageDataset(
    tar_path='images.tar',
    transforms=transforms
)
```

---

## Chunked Token Stream

### ChunkedTokenStream

Stream packed token chunks for language model training.

#### Constructor

```python
ChunkedTokenStream(
    data: Union[Iterable, CodonDataset],
    chunk_len: int,
    batch_size: int,
    seq_len: int,
    eos_token_id: int
)
```

**Constraint:** `chunk_len = batch_size * (seq_len + 1)`

#### Example Usage

```python
from codon.utils.data import ChunkedTokenStream, FlatDataset

source = FlatDataset('tokens.jsonl')
stream = ChunkedTokenStream(
    data=source,
    chunk_len=8192,
    batch_size=8,
    seq_len=1023,
    eos_token_id=0
)

for inputs, labels in stream:
    print(f"Inputs: {inputs.shape}")   # [8, 1023]
    print(f"Labels: {labels.shape}")   # [8, 1023]
```

---

## Stateful Protocol

Datasets can implement state_dict/load_state_dict for checkpointing:

```python
class MyStatefulDataset(CodonDataset):
    def state_dict(self) -> Dict[str, Any]:
        return {'offset': self._offset}
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._offset = state['offset']
```

---

## Notes

1. **Lazy Loading**: FlatDataset supports lazy loading for large files.
2. **Parquet Optimization**: Reads only needed columns/row groups.
3. **Image Caching**: Can cache metadata to speed up initialization.
4. **TAR Benefits**: Reduces I/O overhead for many small files.