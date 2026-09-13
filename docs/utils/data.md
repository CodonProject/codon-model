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

All image datasets return an `ImageDatasetItem`:

| field | meaning |
| --- | --- |
| `image` | the decoded (and possibly transformed) image; a `[N, C, H, W]` tensor or a list when a row carries several images |
| `label` | class index, raw label, label vector, or `None` |
| `path` | file path (`ImageDataset` with `return_path=True`) or the `path_key` column |
| `row` | row/sample number in the source data, absolute even for a `split()` view |
| `parquet_path` | the shard a row came from |
| `key`, `sub_index`, `image_keys` | which image column(s) produced the images |

Shared helpers: `load_image(cell)` decodes bytes / struct / path / URL, and
`is_image_payload` / `is_image_reference` / `image_cell_values` /
`image_cell_kind` are the building blocks used by column detection. All three
datasets share the same training helpers: `split(rank, world_size)`,
`prefetch()`, `get_statistics()`, `sample_weights()`, `collate_dict()`,
`summary()`, `state_dict()` / `load_state_dict()`, and DataLoader batching via
`compose().loader(...)`.

### ImageDataset

Dataset over image files on disk. The layout is auto-detected: `root/class/*.jpg`
labels by folder (`ImageFolder` style), a flat folder yields unlabeled samples, and
a file, a glob, or a list of files/directories all work. Pass several class
directories (`['data/cats', 'data/dogs']`) and they become the classes themselves.

#### Constructor

```python
ImageDataset(
    path: Union[str, Path, Sequence[Union[str, Path]]],
    transforms: Optional[Compose] = None,
    extensions: Optional[Tuple[str, ...]] = None,
    loader: Optional[Callable] = None,
    return_path: bool = False,
    manifest_path: Optional[Union[str, Path]] = None,
    cache_metadata: bool = False,
    classes: Optional[Sequence[Any]] = None,
    in_memory: bool = False,
    num_threads: int = 16,
    verify: bool = False
)
```

* `path` — directory, single image, glob pattern, or a sequence of those.
* `manifest_path` — CSV of `path,label` pairs; relative paths resolve against
  `path` when it is a directory, otherwise against the manifest itself. Blank
  lines, `#` comments, and rows without a label are supported.
* `cache_metadata` — index paths, sizes, and mtimes once and cache them to
  `.codon_image_cache.pkl`; a changed file *and* an added or removed file both
  invalidate it.
* `classes` — explicit class list, which overrides the discovered order; labels
  matching a name are mapped to its index.
* `in_memory` — decode, transform, and keep every image (small datasets only).
* `num_threads` — threads used by `prefetch()`, `verify()`, `get_statistics()`.
* `verify` — open every file once at initialization; `verify()` returns
  `[(path, error), ...]` for the unreadable ones.

`get_statistics()` reports the 0-1 scale used by `Normalize`; pass
`scale='byte'` for 0-255. Statistics ignore `transforms` and measure raw pixels.

#### Example Usage

```python
from codon.data import ImageDataset
from torchvision.transforms import Compose, Resize, ToTensor

transforms = Compose([Resize((224, 224)), ToTensor()])

dataset = ImageDataset(
    path='./images/',                # class folders are detected
    transforms=transforms,
    cache_metadata=True,
)

item = dataset[0]
print(item.image.shape, item.label, item.row)   # torch.Size([3, 224, 224]) 0 0

print(dataset.classes, dataset.num_classes)     # ['cats', 'dogs'] 2
print(dataset.labels)                           # per-sample labels
print(dataset.get_image(0).size)                # (W, H), no transforms
print(dataset.with_path(0).path)                # path regardless of return_path
print(dataset.summary())                        # samples, labeled, classes, ...
stats = dataset.get_statistics(sample_size=1000)
print(stats['mean'], stats['std'])
```

Helpers for training:

```python
dataset.prefetch()                               # decode ahead, warm the page cache
shard = dataset.split(rank=0, world_size=8)      # this rank's samples
loader = dataset.compose(collate_fn=dataset.collate_dict()).loader(
    batch_size=32, shuffle=True, num_workers=8,
)
for batch in loader:
    images = batch['image']                      # [32, 3, 224, 224]
    labels = batch['label']                      # [32]
```

Other forms:

```python
ImageDataset('data/**/*.jpg')                    # glob
ImageDataset(['data/cats', 'data/dogs'])         # one class per directory
ImageDataset('data', manifest_path='labels.csv') # CSV of path,label
ImageDataset('data', extensions=('.png',))       # restrict extensions
ImageDataset('data', classes=['dogs', 'cats'])   # explicit class order
```

`state_dict()` / `load_state_dict()` round-trip the index and label mapping, and
the dataset is picklable for DataLoader workers.

---

### TarImageDataset

Dataset for images in TAR archives.

```python
from codon.data import TarImageDataset

dataset = TarImageDataset(
    tar_path='images.tar',
    transforms=transforms
)
```

---

### ParquetImageDataset

Dataset for images stored in one or more Parquet shards. Only the row group that
contains the requested row is read, so multi-GB shards stay usable without
preloading them.

An image cell may hold:

* raw encoded bytes (`binary`) with a PNG/JPEG/WebP payload,
* a `struct` with `bytes` / `path` fields (the HuggingFace `datasets.Image` layout),
* a string path or URL pointing at an image file,
* a `list` of any of those (several images in one row).

```python
ParquetImageDataset(
    path: Union[str, Path, List[str], List[Path]],
    image_key: Optional[Union[str, Sequence[str]]] = None,
    label_key: Optional[Union[str, List[str]]] = None,
    transforms: Optional[Compose] = None,
    return_path: bool = False,
    path_key: Optional[str] = None,
    classes: Optional[Sequence[Any]] = None,
    index_mode: bool = True,
    mode: Optional[str] = None,
    loader: Optional[Callable] = None,
    cache_size: int = 2,
    columns: Optional[Sequence[str]] = None,
    num_threads: int = 4,
    cache_metadata: bool = False,
    multi_image: str = 'auto',
    max_images: Optional[int] = None
)
```

* `path` — a `.parquet` file, a directory of `*.parquet` shards, or a list of both.
  Rows are addressed globally across every shard.
* `image_key` — one column or several, or `None` to auto-detect; pass
  `detect_image_keys()`-style detection results or read `dataset.image_keys`.
  Detection looks at the schema (`binary`/`struct`/`string`/`list` columns only),
  samples values, and decodes them, so a `caption` text column or a `thumbnail_hash`
  binary column is not mistaken for image data.
* `label_key` — one column, or several for a label vector. `None` returns `None`.
* `classes` — explicit class list; matching labels are mapped to indices. Without
  it, string labels need `index_mode=False` to be sorted into a class list, while
  integer labels stay dense as-is (`index_mode=True`) or are remapped (`False`).
* `cache_size` — row groups kept in the LRU cache; `0` disables caching.
* `cache_metadata` — persist the shard index and class mapping next to the data
  (see `refresh_metadata_cache()`), skipping the scan on re-initialization.
* `mode` — optional PIL conversion, e.g. `'RGB'`; `None` keeps the decoded mode.
* `multi_image` — how a row with several images is combined: `'auto'` stacks when
  the transforms already return equally shaped tensors (else a list), `'stack'`
  always stacks, `'list'` always returns a list. Single-image rows are unaffected.
* `max_images` — cap the number of images taken from one row.

#### Example Usage

```python
from codon.data import ParquetImageDataset

dataset = ParquetImageDataset(
    path='data/images',            # a directory of *.parquet shards
    label_key='label',             # image_key detected automatically
    transforms=Compose([Resize((224, 224)), ToTensor()]),
    cache_size=4,
)

print(dataset.image_keys)         # ['image'] - detected, not guessed by you
item = dataset[0]
print(item.image.shape, item.label, item.row, item.key)

print(dataset.summary())          # shards, rows, row groups, image_key, ...
print(dataset.classes)            # ['cat', 'dog'] (or None for free-form labels)
print(dataset.get_row(0))         # raw columns, image left undecoded
print(dataset.get_statistics())   # {'mean': [...], 'std': [...]}
```

Rows carrying several images:

```python
# A list column, e.g. video frames or augmented views.
clip = ParquetImageDataset(
    path='data/clips',
    image_key='frames',
    transforms=Compose([Resize((224, 224)), ToTensor()]),
)
clip[0].image.shape               # torch.Size([N, 3, 224, 224])

# Two separate image columns, stacked in column order.
pair = ParquetImageDataset(path='data/pairs', image_key=['left', 'right'])
pair[0].image.shape               # torch.Size([2, 3, H, W])

# Keep them separate, or bound the count per row.
ParquetImageDataset(path='data/clips', image_key='frames', multi_image='list')
ParquetImageDataset(path='data/clips', image_key='frames', max_images=8)
```

Inspect detection without relying on it:

```python
probe = ParquetImageDataset(path='data/images', label_key='label')
probe.detect_image_keys()                          # ranked candidates
probe.detect_image_keys(['image', 'caption'])      # score specific columns
probe.detect_image_keys(strict=False)              # [] instead of raising
```

Helpers for training:

```python
# Warm the row-group cache in parallel before the first epoch.
dataset.prefetch()

# One rank's rows, no overlap and no extra memory.
shard = dataset.split(rank=0, world_size=8)

# Batching needs dicts, because default_collate does not accept the dataclass.
loader = dataset.compose(collate_fn=dataset.collate_dict()).loader(
    batch_size=32, shuffle=True, num_workers=8,
)
for batch in loader:
    images = batch['image']       # [32, 3, 224, 224]
    labels = batch['label']       # [32]
```

`state_dict()` / `load_state_dict()` persist the shard list, the class mapping,
the image columns, and the row addressing, and the dataset is picklable, so
DataLoader workers reopen their own parquet handles after spawn. `close()` (or a
`with` block) releases them explicitly.

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