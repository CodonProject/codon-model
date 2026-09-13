import torch
import concurrent.futures
import bisect
import glob
import hashlib
import os, io
import pickle
import tarfile
import tempfile
import threading

from collections import OrderedDict
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass

from pathlib import Path
from typing  import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pyarrow as pa
import pyarrow.parquet as pq

from PIL import Image
from torchvision.transforms import Compose, ToTensor

from .base import CodonDataset

@dataclass
class ImageRecord:
    '''
    One indexed image: its file path and resolved label.

    Attributes:
        path (Path): Absolute path of the image file.
        label (Any): Class index, raw label, or None when the image is unlabeled.
    '''
    path: Path
    label: Any = None

@dataclass
class ImageDatasetItem:
    '''
    A data class representing a single item from the image datasets.

    Attributes:
        image (Any): The decoded (and potentially transformed) image. When a row
            carries several images and ``multi_image`` stacks them, this is a
            tensor of shape ``[N, C, H, W]``; otherwise it is a list of images.
        label (Any): The integer class index, the raw label, a label vector, or
            None when no label is available.
        path (Optional[Path]): The original file path or path within the tar.
        path_key (Optional[str]): The configured column name holding the path.
        row (Optional[int]): The row index inside the source data.
        parquet_path (Optional[Path]): The parquet shard the row was read from.
        key (Optional[str]): The image column this image came from.
        sub_index (Optional[int]): Position inside a multi-image row, or None.
        image_keys (Tuple[str, ...]): All image columns this item was built from.
    '''
    image: Any
    label: Any = None
    path: Optional[Path] = None
    path_key: Optional[str] = None
    row: Optional[int] = None
    parquet_path: Optional[Path] = None
    key: Optional[str] = None
    sub_index: Optional[int] = None
    image_keys: Tuple[str, ...] = ()

def default_loader(path: Path) -> Image.Image:
    '''
    Default image loader using PIL.

    Args:
        path (Path): Path to the image file.

    Returns:
        Image.Image: The loaded image in RGB mode.
    '''
    return Image.open(path).convert('RGB')

def opencv_loader(path: Path) -> Image.Image:
    '''
    Faster image loader using OpenCV if available, falling back to PIL.

    Args:
        path (Path): Path to the image file.

    Returns:
        Image.Image: The loaded image converted to PIL RGB.
    '''
    try:
        import cv2
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)
    except (ImportError, Exception):
        return default_loader(path)


# shared image helpers

_IMAGE_MAGIC: Tuple[bytes, ...] = (
    b'\xff\xd8\xff',         # JPEG
    b'\x89PNG\r\n\x1a\n',    # PNG
    b'GIF87a',               # GIF
    b'GIF89a',
    b'BM',                   # BMP
    b'II*\x00', b'MM\x00*',  # TIFF
    b'\x00\x00\x01\x00',     # ICO
)

_IMAGE_PATH_SUFFIXES: Tuple[str, ...] = (
    '.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif', '.tif', '.tiff', '.ppm', '.pgm',
)

_IMAGE_URL_PREFIXES: Tuple[str, ...] = ('http://', 'https://', 's3://', 'gs://', 'file://')


def _freeze(value: Any) -> Any:
    '''
    Converts a nested value into a hashable, order-insensitive surrogate.

    Args:
        value (Any): A list, tuple, or dict value.

    Returns:
        Any: A hashable surrogate usable as a dict key.
    '''
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _opencv_decode(payload: bytes) -> Optional[Image.Image]:
    '''
    Attempts to decode encoded image bytes with OpenCV.

    Args:
        payload (bytes): Encoded PNG/JPEG/WebP bytes.

    Returns:
        Optional[Image.Image]: The decoded RGB image, or None when OpenCV is
        unavailable or decoding fails.
    '''
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    try:
        buffer = np.frombuffer(payload, dtype=np.uint8)
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if decoded is None:
            return None
        return Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
    except Exception:
        return None



def is_image_payload(payload: Any) -> bool:
    '''
    Checks whether bytes carry a known image signature.

    Args:
        payload (Any): A candidate payload.

    Returns:
        bool: True when the header matches JPEG/PNG/GIF/BMP/TIFF/ICO/WebP.
    '''
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        return False
    data = bytes(payload)
    if not data:
        return False
    if any(data.startswith(magic) for magic in _IMAGE_MAGIC):
        return True
    return data[:4] == b'RIFF' and data[8:12] == b'WEBP'


def is_image_reference(value: Any) -> bool:
    '''
    Checks whether a value looks like a path or URL of an image file.

    Args:
        value (Any): A candidate string.

    Returns:
        bool: True when the value carries an image suffix or URL prefix.
    '''
    if not isinstance(value, str) or not value:
        return False
    lowered = value.lower()
    return lowered.endswith(_IMAGE_PATH_SUFFIXES) or lowered.startswith(_IMAGE_URL_PREFIXES)


def image_cell_values(cell: Any) -> List[Any]:
    '''
    Flattens an image cell into the list of payloads it carries.

    A cell holding one image yields a single-element list; a cell holding a list
    (or a struct of lists) of images yields one entry per image. ``None`` entries
    are dropped, so partially padded rows work.

    Args:
        cell (Any): The raw column value.

    Returns:
        List[Any]: Zero or more payloads (bytes, paths, or decoded images).
    '''
    if cell is None:
        return []
    if isinstance(cell, (bytes, bytearray, memoryview, str, Image.Image, dict)):
        return [cell]
    if isinstance(cell, (list, tuple)):
        values: List[Any] = []
        for item in cell:
            if item is None:
                continue
            if isinstance(item, (list, tuple, dict)):
                values.extend(image_cell_values(item))
            else:
                values.append(item)
        return values
    return [cell]


def image_payload_kind(value: Any, probe: bool = True) -> str:
    '''
    Classifies a single image payload.

    Args:
        value (Any): A cell value or one element of a multi-image cell.
        probe (bool): When True, unrecognised bytes are handed to PIL to decide
            whether they encode an image. Defaults to True.

    Returns:
        str: ``'image'``, ``'struct'``, ``'path'``, ``'bytes'``, or ``'unknown'``.
    '''
    if isinstance(value, Image.Image):
        return 'image'
    if isinstance(value, (bytes, bytearray, memoryview)):
        if is_image_payload(value):
            return 'image'
        if not probe:
            return 'bytes'
        try:
            with Image.open(io.BytesIO(bytes(value))) as handle:
                handle.verify()
            return 'image'
        except Exception:
            return 'bytes'
    if isinstance(value, dict):
        payload = value.get('bytes', value.get('data'))
        reference = value.get('path')
        if payload is not None and image_payload_kind(payload, probe=probe) == 'image':
            return 'struct'
        if isinstance(reference, str) and is_image_reference(reference):
            return 'struct'
        return 'unknown'
    if isinstance(value, str):
        return 'path' if is_image_reference(value) else 'unknown'
    return 'unknown'


def image_cell_kind(cell: Any, probe: bool = True, sample: int = 4) -> str:
    '''
    Classifies a whole cell from its leading payloads.

    Args:
        cell (Any): The raw column value.
        probe (bool): Passed through to :func:`image_payload_kind`.
        sample (int): Maximum payloads inspected. Defaults to 4.

    Returns:
        str: The strongest kind found, or ``'unknown'``.
    '''
    values = image_cell_values(cell)[:max(1, sample)]
    if not values:
        return 'unknown'
    priority = {'image': 3, 'struct': 2, 'path': 1, 'bytes': -1, 'unknown': -2}
    best = 'bytes'
    for value in values:
        kind = image_payload_kind(value, probe=probe)
        if priority.get(kind, -3) > priority.get(best, -3):
            best = kind
    return best


def load_image(cell: Any, loader: Callable = default_loader) -> Image.Image:
    '''
    Decodes one image payload into a PIL image, without applying transforms.

    Args:
        cell (Any): Encoded bytes, a ``{'bytes','path'}`` struct, a path or URL, or
            an already decoded PIL image.
        loader (Callable): Loader used for filesystem paths. Defaults to
            :func:`default_loader`.

    Returns:
        Image.Image: The decoded image.

    Raises:
        RuntimeError: If the payload cannot be decoded or the file is missing.
        TypeError: If the payload type is unsupported.
    '''
    if isinstance(cell, Image.Image):
        return cell

    if isinstance(cell, dict):
        payload = cell.get('bytes', cell.get('data'))
        reference = cell.get('path')
        if payload is not None:
            try:
                return load_image(payload, loader)
            except Exception:
                if reference is None:
                    raise
        if reference is None:
            raise TypeError(
                f'Image struct carries neither "bytes" nor "path": keys={sorted(cell.keys())}'
            )
        return load_image(reference, loader)

    if isinstance(cell, (bytes, bytearray, memoryview)):
        data = bytes(cell)
        if not data:
            raise RuntimeError('The image payload is empty.')
        try:
            with Image.open(io.BytesIO(data)) as handle:
                handle.load()
                return handle.copy()
        except Exception:
            decoded = _opencv_decode(data)
            if decoded is None:
                raise
            return decoded

    if isinstance(cell, str):
        if cell.startswith(('http://', 'https://')):
            try:
                import urllib.request
                with urllib.request.urlopen(cell) as response:
                    return load_image(response.read(), loader)
            except Exception as error:
                raise RuntimeError(f'Failed to fetch image from {cell}: {error}') from error
        candidate = Path(cell)
        if not candidate.exists():
            raise RuntimeError(f'Referenced image file does not exist: {cell}')
        return loader(candidate)

    if isinstance(cell, (list, tuple)) and cell and all(isinstance(b, int) for b in cell):
        return load_image(bytes(cell), loader)

    if cell is None:
        raise TypeError('The image cell is NULL in this row.')

    raise TypeError(
        f'Unsupported image cell of type {type(cell).__name__}; expected bytes, an '
        f'image struct, or a path.'
    )


def _looks_like_number(text: str) -> bool:
    '''
    Checks whether a string holds an integer or float literal.

    Args:
        text (str): The candidate text.

    Returns:
        bool: True for a numeric literal.
    '''
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def collate_image_items(batch: List[Any]) -> Any:
    '''
    Collates a batch of image dataset items field by field.

    ``torch.utils.data.default_collate`` accepts dicts, tensors, and numbers, but
    not dataclass instances, so a DataLoader over :class:`ImageDataset` or
    :class:`ParquetImageDataset` needs this: every field is batched on its own and
    the item type is rebuilt, which keeps ``batch.image`` / ``batch.label``
    working exactly like ``ImageDataset`` always did. A field that is ``None`` for
    any sample stays ``None`` for the whole batch instead of raising.

    Args:
        batch (List[Any]): The items of one batch, as produced by
            ``__getitem__``.

    Returns:
        Any: The same item type with every field batched.
    '''
    from torch.utils.data._utils.collate import default_collate

    if not batch:
        return batch

    first = batch[0]
    if not is_dataclass(first):
        return default_collate(batch)

    names = [field.name for field in dataclass_fields(first)]
    collected: Dict[str, Any] = {}
    for name in names:
        values = [getattr(item, name) for item in batch]
        if any(value is None for value in values):
            collected[name] = None
            continue
        if all(isinstance(value, Path) for value in values):
            collected[name] = [str(value) for value in values]
            continue
        collected[name] = default_collate(values)
    return type(first)(**collected)


class ImageDataset(CodonDataset):
    '''
    A map-style dataset over image files on disk, with manifest support.

    The layout is detected from the input:

    * **class folders** — ``root/class_a/*.jpg`` labels each image by its parent
      directory, like ``torchvision.datasets.ImageFolder``;
    * **flat folder** — ``root/*.jpg`` yields every image with label ``None``;
    * **explicit paths** — a file, a glob pattern (``'data/**/*.png'``), or a list
      of files and directories;
    * **manifest** — a CSV of ``path,label`` pairs (see ``manifest_path``).

    The metadata index (paths, sizes, mtimes) is built with a single tree walk and
    can be cached next to the data, while decoding stays lazy unless
    ``in_memory=True``.

    Attributes:
        _path (Union[Path, List[Path]]): The configured source.
        _transforms (Optional[Compose]): Transformations applied to the images.
        _extensions (Tuple[str, ...]): Valid image file extensions.
        _loader (Callable): Function used to load images from disk.
        _return_path (bool): Whether to include the file path in the item.
        _manifest_path (Optional[Path]): Path to a CSV of ``path,label`` pairs.
        _classes (List[Any]): Class names, when labels are categorical.
        _class_to_idx (Optional[Dict[Any, int]]): Class name to index mapping.
        _samples (List[ImageRecord]): Indexed ``(path, label)`` records.
        _cache (Dict[int, Any]): Decoded images, used when ``in_memory=True``.
    '''

    def __init__(
        self,
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
        verify: bool = False,
    ) -> None:
        '''
        Initializes the ImageDataset.

        Args:
            path (Union[str, Path, Sequence[...]]): A directory, an image file, a
                glob pattern, or a sequence of any of those.
            transforms (Optional[Compose]): A composition of torchvision transforms
                applied to every decoded image.
            extensions (Optional[Tuple[str, ...]]): Valid image extensions.
                Defaults to ``('.jpg', '.jpeg', '.png', '.bmp', '.webp')``.
            loader (Optional[Callable]): Custom loader for filesystem paths.
            return_path (bool): If True, the file path is returned in the item.
            manifest_path (Optional[Union[str, Path]]): CSV of ``path,label`` pairs.
                Relative paths resolve against ``path`` when that is a directory,
                otherwise against the manifest's own directory.
            cache_metadata (bool): If True, the scanned index is cached to disk and
                revalidated against file sizes and mtimes, so later constructions
                skip the tree walk.
            classes (Optional[Sequence[Any]]): Explicit class list. Labels matching
                a class name are mapped to its index; other labels keep their raw
                value. Defaults to the sorted folder names.
            in_memory (bool): If True, every image is decoded, transformed, and kept
                in memory at initialization.
            num_threads (int): Threads used by :meth:`prefetch`, :meth:`verify`, and
                :meth:`get_statistics`. Defaults to 16.
            verify (bool): If True, every indexed file is opened once at
                initialization so unreadable files are reported by
                :meth:`verify`.

        Raises:
            FileNotFoundError: If nothing matches the configured source.
            ValueError: If a manifest row is malformed.
        '''
        super().__init__()
        self._path = path
        self._transforms = transforms
        self._extensions = tuple(extensions) if extensions else (
            '.jpg', '.jpeg', '.png', '.bmp', '.webp'
        )
        self._loader = loader or default_loader
        self._return_path = return_path
        self._manifest_path = Path(manifest_path) if manifest_path else None
        self._cache_metadata = cache_metadata
        self._in_memory = in_memory
        self._num_threads = max(1, int(num_threads))
        self._explicit_classes = [str(cls) for cls in classes] if classes is not None else None

        self._classes: List[Any] = []
        self._class_to_idx: Optional[Dict[Any, int]] = None
        self._samples: List[ImageRecord] = []
        self._cache: Dict[int, Any] = {}
        self._row_offset = 0
        self._row_count: Optional[int] = None

        if not self._restore_metadata_cache():
            self._build_index()

        if verify:
            self.verify()
        if self._in_memory:
            self.prefetch()

    # ------------------------------------------------------------------
    # index construction
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_sources(source: Any, extensions: Tuple[str, ...]) -> List[Path]:
        '''
        Expands the configured source into concrete image file paths.

        Args:
            source (Any): A path, a glob pattern, or a sequence of both.
            extensions (Tuple[str, ...]): Extensions accepted for directories.

        Returns:
            List[Path]: Matching image files.

        Raises:
            FileNotFoundError: If a concrete entry does not exist.
        '''
        entries: Iterable[Any]
        if isinstance(source, (list, tuple, set)):
            entries = list(source)
        else:
            entries = [source]

        found: List[Path] = []
        for entry in entries:
            text = str(entry)
            entry_path = Path(entry)
            if entry_path.is_dir():
                found.extend(
                    path for path in sorted(entry_path.rglob('*'))
                    if path.is_file() and path.suffix.lower() in extensions
                )
            elif entry_path.is_file():
                found.append(entry_path)
            elif any(char in text for char in '*?['):
                found.extend(
                    sorted(
                        Path(match) for match in glob.glob(text, recursive=True)
                        if Path(match).is_file()
                    )
                )
            else:
                raise FileNotFoundError(f'No such image file or directory: {entry_path}')
        return list(dict.fromkeys(found))

    def _build_index(self) -> None:
        '''
        Builds the sample index from the manifest or the filesystem.

        Raises:
            FileNotFoundError: If nothing matches the configured source.
            ValueError: If a manifest row is malformed.
        '''
        if self._manifest_path is not None and self._manifest_path.exists():
            samples, class_names = self._read_manifest()
        else:
            files = self._expand_sources(self._path, self._extensions)
            if not files:
                raise FileNotFoundError(
                    f'No images with extensions {self._extensions} found under: {self._path}'
                )
            samples, class_names = self._label_from_layout(files)

        self._samples = samples
        self._apply_classes(class_names)

        if self._cache_metadata:
            self.refresh_metadata_cache()

    def _label_from_layout(self, files: List[Path]) -> Tuple[List[ImageRecord], List[Any]]:
        '''
        Derives labels from the directory layout.

        A file is labelled by its path below its source root, which is the
        ``root/class/file.jpg`` convention; nested class folders keep their relative
        path as the class name. Source roots that are themselves class folders are
        notable: listing them (``['data/cats', 'data/dogs']``) labels those files
        ``cats`` and ``dogs``. Files sitting directly in a single root stay
        unlabeled, so a partially organised tree still indexes cleanly.

        Args:
            files (List[Path]): Image files to index.

        Returns:
            Tuple[List[ImageRecord], List[Any]]: Records and discovered classes.
        '''
        roots = self._source_roots()
        # Several sources are passed one class of image each; a single source is a
        # dataset root whose subdirectories are the classes.
        leaf_is_class = len(roots) > 1

        samples: List[ImageRecord] = []
        class_names: List[Any] = []

        for file_path in files:
            absolute = file_path.absolute()
            class_name = self._class_of(absolute, roots, leaf_is_class)
            label: Any = None
            if class_name is not None:
                if class_name not in class_names:
                    class_names.append(class_name)
                label = class_name
            samples.append(ImageRecord(path=absolute, label=label))

        return samples, sorted(class_names, key=str)

    def _source_roots(self) -> List[Path]:
        '''
        Returns the directory roots used to derive class labels.

        For a glob pattern the non-magic prefix is used, so ``'data/**/*.jpg'``
        still labels ``data/cats/a.jpg`` as ``cats``.

        Returns:
            List[Path]: Existing directories among the configured sources.
        '''
        entries: Iterable[Any]
        if isinstance(self._path, (list, tuple, set)):
            entries = list(self._path)
        else:
            entries = [self._path]

        roots: List[Path] = []
        for entry in entries:
            parts = Path(entry).absolute().parts
            kept: List[str] = []
            for part in parts:
                if any(char in part for char in '*?['):
                    break
                kept.append(part)
            if not kept:
                continue
            candidate = Path(*kept)
            if candidate.is_dir():
                roots.append(candidate)
        return roots

    @staticmethod
    def _class_of(absolute: Path, roots: List[Path], leaf_is_class: bool) -> Optional[str]:
        '''
        Returns the class name of an image, when the layout implies one.

        Args:
            absolute (Path): Absolute image path.
            roots (List[Path]): Resolved source roots.
            leaf_is_class (bool): Whether a file directly inside a root is itself a
                class sample (true when several roots were listed).

        Returns:
            Optional[str]: The class name, or None for an unlabeled image.
        '''
        for root in roots:
            try:
                relative = absolute.relative_to(root)
            except ValueError:
                continue
            if len(relative.parts) >= 2:
                return str(Path(*relative.parts[:-1]))
            if leaf_is_class:
                return root.name
        return None

    def _read_manifest(self) -> Tuple[List[ImageRecord], List[Any]]:
        '''
        Reads the ``path,label`` manifest.

        Blank lines and ``#`` comments are skipped. Numeric labels become ints;
        a row without a label is kept as unlabeled.

        Returns:
            Tuple[List[ImageRecord], List[Any]]: Records and discovered classes.

        Raises:
            ValueError: If a row has an empty path.
        '''
        base = Path(self._path) if isinstance(self._path, (str, Path)) and Path(self._path).is_dir() \
            else self._manifest_path.parent
        samples: List[ImageRecord] = []
        class_names: List[Any] = []

        with open(self._manifest_path, 'r', encoding='utf-8') as handle:
            for number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [part.strip() for part in line.split(',')]
                if not parts[0]:
                    raise ValueError(f'{self._manifest_path}:{number}: empty path')
                image_path = Path(parts[0])
                if not image_path.is_absolute():
                    image_path = base / image_path

                label: Any = None
                label_text = parts[1] if len(parts) > 1 else ''
                if label_text:
                    if _looks_like_number(label_text):
                        label = int(float(label_text))
                    else:
                        label = label_text
                        if label_text not in class_names:
                            class_names.append(label_text)
                samples.append(ImageRecord(path=image_path.absolute(), label=label))

        return samples, sorted(class_names, key=str)

    def _apply_classes(self, discovered: List[Any]) -> None:
        '''
        Resolves the class list and the label mapping.

        Integer labels (``0..9`` folders, numeric manifest labels) are already dense
        class ids and stay verbatim; string labels are mapped to their index once a
        class list is known.

        Args:
            discovered (List[Any]): Class names found while indexing.
        '''
        names = self._explicit_classes if self._explicit_classes is not None else discovered
        labels = [record.label for record in self._samples if record.label is not None]
        numeric_only = bool(labels) and all(
            isinstance(label, int) and not isinstance(label, bool) for label in labels
        )

        if not names or numeric_only:
            self._classes = []
            self._class_to_idx = None
            return

        self._classes = list(names)
        self._class_to_idx = {name: index for index, name in enumerate(self._classes)}

        for record in self._samples:
            if record.label is None:
                continue
            if record.label in self._class_to_idx:
                record.label = self._class_to_idx[record.label]
            elif str(record.label) in self._class_to_idx:
                record.label = self._class_to_idx[str(record.label)]

    # ------------------------------------------------------------------
    # metadata cache
    # ------------------------------------------------------------------

    def _cache_file(self) -> Path:
        '''
        Returns the metadata cache path.

        Preferred location is next to the data (or next to the manifest), so the
        cache travels with the dataset. Multi-source inputs fall back to the system
        temporary directory.

        Returns:
            Path: The cache file path.
        '''
        if isinstance(self._path, (str, Path)):
            base = Path(self._path)
            if base.is_dir():
                return base / '.codon_image_cache.pkl'
        if self._manifest_path is not None:
            return self._manifest_path.parent / '.codon_image_cache.pkl'
        digest = hashlib.sha1(
            f'{self._path}|{self._manifest_path}'.encode('utf-8')
        ).hexdigest()[:16]
        return Path(tempfile.gettempdir()) / f'codon_image_cache_{digest}.pkl'

    def _data_base(self) -> Optional[Path]:
        '''
        Returns the absolute directory that cache entries are relative to.

        Returns:
            Optional[Path]: The dataset root, or None when entries must be stored as
            absolute paths (a file/glob/multi-source input).
        '''
        if isinstance(self._path, (str, Path)):
            base = Path(self._path).absolute()
            if base.is_dir():
                return base
        return None

    def _relative_index(self) -> List[Tuple[str, ImageRecord, int, int]]:
        '''
        Builds the ``(relative_path, record, size, mtime_ns)`` cache signature.

        Returns:
            List[Tuple[str, ImageRecord, int, int]]: One entry per sample.
        '''
        base = self._data_base()
        entries: List[Tuple[str, ImageRecord, int, int]] = []
        for record in self._samples:
            stat = record.path.stat()
            relative = str(record.path.relative_to(base)) if base is not None else str(record.path)
            entries.append((relative, record, stat.st_size, stat.st_mtime_ns))
        return entries

    def _directory_signature(self) -> Tuple[Tuple[str, int, int], ...]:
        '''
        Captures the state of the directories the index was scanned from.

        Per-file ``stat`` data alone cannot notice an *added* image, so the parent
        directories are recorded too: creating or deleting a file bumps the parent
        mtime, and its file count changes as well.

        Returns:
            Tuple[Tuple[str, int, int], ...]: Sorted ``(path, mtime_ns, entries)``.
        '''
        directories = {record.path.parent for record in self._samples}
        directories.update(root for root in (self._data_base(),) if root is not None)
        signature: List[Tuple[str, int, int]] = []
        for directory in directories:
            try:
                stat = directory.stat()
            except OSError:
                signature.append((str(directory), -1, -1))
                continue
            try:
                entries = sum(1 for _ in os.scandir(directory))
            except OSError:
                entries = -1
            signature.append((str(directory), stat.st_mtime_ns, entries))
        return tuple(sorted(signature))

    def refresh_metadata_cache(self) -> Optional[Path]:
        '''
        Writes the current index to the metadata cache.

        Returns:
            Optional[Path]: The cache file written, or None when caching is disabled
            or the write failed (a read-only dataset must never break).
        '''
        if not self._cache_metadata:
            return None
        try:
            stored = [
                (relative, record.label, size, mtime)
                for relative, record, size, mtime in self._relative_index()
            ]
            payload = {
                'path': str(self._path),
                'extensions': self._extensions,
                'classes': list(self._classes),
                'class_to_idx': dict(self._class_to_idx) if self._class_to_idx else None,
                'samples': stored,
                'directories': self._directory_signature(),
            }
            cache_file = self._cache_file()
            with open(cache_file, 'wb') as handle:
                pickle.dump(payload, handle)
            return cache_file
        except Exception:
            return None

    def _restore_metadata_cache(self) -> bool:
        '''
        Loads the index from the on-disk cache when it is still valid.

        Every referenced file is re-``stat``-ed and every source directory
        re-checked, so added, removed, or rewritten images always invalidate the
        cache.

        Returns:
            bool: True when a valid cache replaced the filesystem scan.
        '''
        if not self._cache_metadata:
            return False

        cache_file = self._cache_file()
        if not cache_file.exists():
            return False
        try:
            with open(cache_file, 'rb') as handle:
                payload = pickle.load(handle)
        except Exception:
            return False

        stored = payload.get('samples') or []
        if not stored:
            return False
        if payload.get('path') != str(self._path):
            return False
        if tuple(payload.get('extensions') or ()) != self._extensions:
            return False

        base = self._data_base()
        records: List[ImageRecord] = []
        for relative, label, size, mtime in stored:
            candidate = Path(relative)
            path = candidate if candidate.is_absolute() or base is None else base / candidate
            try:
                stat = path.stat()
            except OSError:
                return False
            if stat.st_size != size or stat.st_mtime_ns != mtime:
                return False
            records.append(ImageRecord(path=path, label=label))

        self._samples = records
        self._classes = list(payload.get('classes') or [])
        mapping = payload.get('class_to_idx')
        self._class_to_idx = dict(mapping) if mapping else None

        # Only now can the directory signature be rebuilt, since it needs the
        # indexed sample paths. It catches files that were added since the write.
        stored_directories = tuple(
            tuple(entry) for entry in (payload.get('directories') or ())
        )
        if stored_directories and stored_directories != self._directory_signature():
            self._samples = []
            self._classes = []
            self._class_to_idx = None
            return False
        return True

    # ------------------------------------------------------------------
    # CodonDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        '''
        Returns the number of indexed images.

        Returns:
            int: Sample count (just this worker's share for a :meth:`split` view).
        '''
        if self._row_count is not None:
            return self._row_count
        return len(self._samples)

    def __getitem__(self, idx: int) -> ImageDatasetItem:
        '''
        Loads, decodes, and transforms one image.

        Args:
            idx (int): Index inside this instance (negative values wrap around).

        Returns:
            ImageDatasetItem: The image, its label, and the file path.

        Raises:
            IndexError: If the index is out of range.
            RuntimeError: If the image cannot be decoded.
        '''
        record, local = self._record_at(idx)
        return ImageDatasetItem(
            image=self._decoded(local, record),
            label=record.label,
            path=record.path if self._return_path else None,
            row=local + self._row_offset,
        )

    def get_image(self, idx: int) -> Image.Image:
        '''
        Loads and returns the raw PIL image of one sample, without transforms.

        Args:
            idx (int): Index inside this instance (negative values wrap around).

        Returns:
            Image.Image: The decoded image.

        Raises:
            IndexError: If the index is out of range.
            RuntimeError: If the image cannot be decoded.
        '''
        record, _ = self._record_at(idx)
        try:
            return self._loader(record.path)
        except Exception as error:
            raise RuntimeError(f'Failed to load image at {record.path}: {error}') from error

    def with_path(self, idx: int) -> ImageDatasetItem:
        '''
        Returns one item with its file path attached, whatever ``return_path`` is.

        Args:
            idx (int): Index inside this instance.

        Returns:
            ImageDatasetItem: The item, with ``path`` populated.
        '''
        item = self[idx]
        item.path = self._record_at(idx)[0].path
        return item

    def __iter__(self):
        '''
        Yields every item in index order.

        Returns:
            Iterator[ImageDatasetItem]: Item iterator.
        '''
        for idx in range(len(self)):
            yield self[idx]

    def _record_at(self, idx: int) -> Tuple[ImageRecord, int]:
        '''
        Resolves a local index into its record.

        Args:
            idx (int): Index inside this instance (negative values wrap around).

        Returns:
            Tuple[ImageRecord, int]: The record and the normalized local index.

        Raises:
            IndexError: If the index is out of range.
        '''
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f'Index {idx} out of range (0-{len(self) - 1})')
        return self._samples[idx + self._row_offset], idx

    def _decoded(self, local: int, record: ImageRecord) -> Any:
        '''
        Returns the decoded (and transformed) image of a record.

        Args:
            local (int): Local index, used as the in-memory cache key.
            record (ImageRecord): The record to decode.

        Returns:
            Any: The transformed image or tensor.

        Raises:
            RuntimeError: If the file cannot be decoded.
        '''
        if local in self._cache:
            return self._cache[local]

        try:
            image = self._loader(record.path)
        except Exception as error:
            raise RuntimeError(f'Failed to load image at {record.path}: {error}') from error

        if self._transforms is not None:
            image = self._transforms(image)
        if self._in_memory:
            self._cache[local] = image
        return image

    # ------------------------------------------------------------------
    # convenience accessors
    # ------------------------------------------------------------------

    @property
    def records(self) -> List[ImageRecord]:
        '''
        Returns the indexed records.

        Returns:
            List[ImageRecord]: Copies of the ``(path, label)`` records, so the
            caller cannot corrupt the index.
        '''
        return [ImageRecord(record.path, record.label) for record in self._samples]

    @property
    def files(self) -> List[Path]:
        '''
        Returns the indexed file paths.

        Returns:
            List[Path]: Absolute image paths in index order.
        '''
        return [record.path for record in self._samples]

    @property
    def labels(self) -> List[Any]:
        '''
        Returns the resolved label of every sample.

        Returns:
            List[Any]: Labels in index order (None where unlabeled).
        '''
        return [record.label for record in self._samples]

    @property
    def classes(self) -> List[Any]:
        '''
        Returns the class list.

        Returns:
            List[Any]: Class names, empty when labels are not categorical.
        '''
        return list(self._classes)

    @property
    def class_to_idx(self) -> Optional[Dict[Any, int]]:
        '''
        Returns the class name to index mapping.

        Returns:
            Optional[Dict[Any, int]]: A copy of the mapping, or None.
        '''
        return dict(self._class_to_idx) if self._class_to_idx else None

    @property
    def num_classes(self) -> Optional[int]:
        '''
        Returns how many classes the labels cover.

        Returns:
            Optional[int]: The class count, or None when labels are unavailable.
        '''
        if self._classes:
            return len(self._classes)
        integer_labels = [
            label for label in self.labels
            if isinstance(label, int) and not isinstance(label, bool)
        ]
        if not integer_labels or min(integer_labels) < 0:
            return None
        return max(integer_labels) + 1

    def split(self, rank: int, world_size: int) -> 'ImageDataset':
        '''
        Returns a non-overlapping slice of this dataset for one worker.

        The view shares the index and the in-memory cache, so it costs nothing
        extra. Intended for multi-worker / distributed training.

        Args:
            rank (int): Zero-based index of this worker.
            world_size (int): Total number of workers.

        Returns:
            ImageDataset: A view covering this worker's samples.

        Raises:
            ValueError: If ``rank`` or ``world_size`` is out of range.
        '''
        if world_size < 1:
            raise ValueError(f'world_size must be >= 1, got {world_size}')
        if not 0 <= rank < world_size:
            raise ValueError(f'rank must be in [0, {world_size}), got {rank}')

        view = object.__new__(ImageDataset)
        view.__dict__.update(self.__dict__)
        total = len(self)
        start = total * rank // world_size
        end = total * (rank + 1) // world_size
        view._row_offset = self._row_offset + start
        view._row_count = end - start
        return view

    def collate_dict(self) -> Callable[[ImageDatasetItem], Dict[str, Any]]:
        '''
        Returns a collate function that turns items into plain dicts.

        Useful when the training loop prefers a mapping over the dataclass; the
        default :meth:`compose` collation already produces batched dataclasses.
        Only populated keys are emitted, so ``default_collate`` never sees a
        ``None`` field::

            loader = dataset.compose(collate_fn=dataset.collate_dict()).loader(4)

        Returns:
            Callable[[ImageDatasetItem], Dict[str, Any]]: The collate function.
        '''
        def _collate(item: ImageDatasetItem) -> Dict[str, Any]:
            batch = {'image': item.image}
            if item.label is not None:
                batch['label'] = item.label
            if item.path is not None:
                batch['path'] = str(item.path)
            if item.row is not None:
                batch['row'] = item.row
            return batch

        return _collate

    def compose(self, collate_fn: Optional[Callable] = None, **kwargs: Any) -> Any:
        '''
        Wraps the dataset for PyTorch.

        Batches are collated by :func:`collate_image_items` unless ``collate_fn``
        is given, so a plain ``dataset.compose().loader(...)`` yields batches whose
        fields mirror :class:`ImageDatasetItem`.

        Args:
            collate_fn (Optional[Callable]): Per-item function, passed through to
                :meth:`CodonDataset.compose`.
            **kwargs (Any): Forwarded to :meth:`CodonDataset.compose`.

        Returns:
            TorchDatasetWrapper: The wrapped dataset.
        '''
        return super().compose(collate_fn=collate_fn, **kwargs)

    def prefetch(self, num_samples: Optional[int] = None) -> int:
        '''
        Decodes samples in parallel to warm the OS page cache.

        Nothing is retained unless ``in_memory=True``: the point is to pay the
        cold-read cost up front instead of inside the first training steps.

        Args:
            num_samples (Optional[int]): Stop after this many samples. Defaults to
                every sample.

        Returns:
            int: The number of samples decoded (failures are skipped).
        '''
        limit = len(self) if num_samples is None else min(int(num_samples), len(self))
        if limit <= 0:
            return 0

        def _load(local: int) -> bool:
            record = self._samples[local + self._row_offset]
            try:
                image = self._loader(record.path)
                if self._transforms is not None:
                    image = self._transforms(image)
                if self._in_memory:
                    self._cache[local] = image
                return True
            except Exception:
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._num_threads) as pool:
            return sum(1 for ok in pool.map(_load, range(limit)) if ok)

    def verify(self) -> List[Tuple[Path, str]]:
        '''
        Opens every indexed file once and reports the ones that fail.

        Returns:
            List[Tuple[Path, str]]: ``(path, error)`` pairs for unreadable files,
            empty when the dataset is clean.
        '''
        def _check(position: int) -> Optional[Tuple[Path, str]]:
            record = self._samples[position]
            try:
                with Image.open(record.path) as handle:
                    handle.verify()
                return None
            except Exception as error:
                return record.path, str(error)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._num_threads) as pool:
            failures = list(pool.map(_check, range(len(self._samples))))
        return [failure for failure in failures if failure is not None]

    def sample_weights(self) -> Optional[torch.Tensor]:
        '''
        Computes per-sample inverse-frequency weights for balanced sampling.

        Returns:
            Optional[torch.Tensor]: A float tensor of shape ``[len(self)]``, or None
            when labels are missing or not integer-valued.
        '''
        labels = [record.label for record in self._samples]
        if not labels or any(
            not isinstance(label, int) or isinstance(label, bool) for label in labels
        ):
            return None
        tensor = torch.as_tensor(labels, dtype=torch.long)
        counts = torch.bincount(tensor, minlength=int(tensor.max()) + 1).clamp(min=1)
        return (1.0 / counts.float())[tensor]

    def get_statistics(
        self, sample_size: Optional[int] = 1000, scale: str = 'unit'
    ) -> Dict[str, List[float]]:
        '''
        Estimates per-channel mean and standard deviation.

        Images are measured after ``ToTensor``, so the default statistics are on
        the 0-1 scale used by ``Normalize``; pass ``scale='byte'`` for the 0-255
        range shown by most dataset viewers.

        Args:
            sample_size (Optional[int]): Samples to use. Defaults to 1000; None or
                a value >= ``len(self)`` uses every sample.
            scale (str): ``'unit'`` (default, 0-1) or ``'byte'`` (0-255).

        Returns:
            Dict[str, List[float]]: ``{'mean': [...], 'std': [...]}``.
        '''
        loader_transform = Compose([ToTensor()])
        total = len(self)
        if total == 0:
            return {'mean': [0.0, 0.0, 0.0], 'std': [0.0, 0.0, 0.0]}

        if sample_size is None or sample_size >= total:
            indices = list(range(total))
        else:
            indices = torch.randperm(total)[:sample_size].tolist()

        def _process(local: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
            try:
                record = self._samples[local + self._row_offset]
                tensor = loader_transform(self._loader(record.path))
                return torch.mean(tensor, dim=(1, 2)), torch.std(tensor, dim=(1, 2))
            except Exception:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._num_threads) as pool:
            results = [result for result in pool.map(_process, indices) if result is not None]

        if not results:
            return {'mean': [0.0, 0.0, 0.0], 'std': [0.0, 0.0, 0.0]}

        factor = 255.0 if scale == 'byte' else 1.0
        means = torch.stack([result[0] for result in results]) * factor
        stds = torch.stack([result[1] for result in results]) * factor
        return {
            'mean': torch.mean(means, dim=0).tolist(),
            'std': torch.mean(stds, dim=0).tolist(),
        }

    def summary(self) -> Dict[str, Any]:
        '''
        Reports the resolved layout of the dataset.

        Returns:
            Dict[str, Any]: Source, sample count, label coverage, and class count.
        '''
        labeled = sum(1 for record in self._samples if record.label is not None)
        return {
            'path': str(self._path),
            'samples': len(self._samples),
            'labeled': labeled,
            'unlabeled': len(self._samples) - labeled,
            'classes': list(self._classes),
            'manifest': str(self._manifest_path) if self._manifest_path else None,
            'extensions': list(self._extensions),
            'in_memory': self._in_memory,
            'cached': len(self._cache),
        }

    # ------------------------------------------------------------------
    # Stateful protocol & pickling
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        '''
        Snapshots the index so a checkpoint can restore the exact sample order.

        Returns:
            Dict[str, Any]: Picklable state dictionary.
        '''
        return {
            'path': self._path,
            'manifest_path': str(self._manifest_path) if self._manifest_path else None,
            'extensions': self._extensions,
            'return_path': self._return_path,
            'classes': list(self._classes),
            'class_to_idx': dict(self._class_to_idx) if self._class_to_idx else None,
            'samples': [(str(record.path), record.label) for record in self._samples],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        '''
        Restores the dataset from a previously produced state dictionary.

        The stored index is installed as-is, so restoring never walks the tree
        again.

        Args:
            state (Dict[str, Any]): State from :meth:`state_dict`.
        '''
        stored = state.get('samples')
        self._cache.clear()
        if not stored:
            self._build_index()
            return
        self._samples = [ImageRecord(path=Path(path), label=label) for path, label in stored]
        self._classes = list(state.get('classes') or [])
        mapping = state.get('class_to_idx')
        self._class_to_idx = dict(mapping) if mapping else None

    def __getstate__(self) -> Dict[str, Any]:
        '''
        Produces picklable state, dropping the decoded-image cache.

        Returns:
            Dict[str, Any]: The instance dictionary without decoded images.
        '''
        state = self.__dict__.copy()
        state['_cache'] = {}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        '''
        Restores instance state.

        Args:
            state (Dict[str, Any]): State produced by :meth:`__getstate__`.
        '''
        self.__dict__.update(state)
        if self._cache is None:
            self._cache = {}

    def close(self) -> None:
        '''
        Drops cached decoded images, releasing their memory.

        Unlike :meth:`ParquetImageDataset.close` there are no persistent file
        handles to release, because every load opens and closes its own file.
        '''
        self._cache.clear()

    def __enter__(self) -> 'ImageDataset':
        '''
        Enters the runtime context.

        Returns:
            ImageDataset: self.
        '''
        return self

    def __exit__(self, *exc_info: Any) -> None:
        '''
        Releases cached images on context exit.

        Args:
            *exc_info (Any): Exception details, ignored.
        '''
        self.close()

class TarImageDataset(CodonDataset):
    '''
    A dataset class for loading image files directly from a TAR archive.

    This avoids high I/O overhead from many small image files on the filesystem.

    Attributes:
        _tar_path (Path): Path to the tar archive file.
        _transforms (Optional[Compose]): Transformations to apply to the images.
        _extensions (Tuple[str, ...]): Valid image file extensions.
        _return_path (bool): Whether to include the file path in the returned item.
        _classes (List[str]): List of class names (parsed from tar paths).
        _class_to_idx (Dict[str, int]): Mapping from class name to integer label.
        _samples (List[Tuple[str, int]]): List of (member_name, label) pairs.
    '''

    def __init__(
        self,
        tar_path: Union[str, Path],
        transforms: Optional[Compose] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        return_path: bool = False
    ) -> None:
        '''
        Initializes the TarImageDataset.

        Args:
            tar_path (Union[str, Path]): Path to the tar archive file.
            transforms (Optional[Compose]): A composition of torchvision transforms.
            extensions (Optional[Tuple[str, ...]): Valid image file extensions.
            return_path (bool): If True, returns the file path within the tar.
        '''
        super().__init__()
        self._tar_path = Path(tar_path)
        self._transforms = transforms
        self._extensions = extensions or ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        self._return_path = return_path
        self._tar_handle = None

        self._samples, self._classes, self._class_to_idx = self._build_index()

    def _build_index(self) -> Tuple[List[Tuple[str, int]], List[str], Dict[str, int]]:
        '''
        Scans the tar archive once to build an index and find classes.

        Returns:
            Tuple[List[Tuple[str, int]], List[str], Dict[str, int]]:
                Index of members, list of classes, and class mapping.
        '''
        samples = []
        class_names = set()

        if not self._tar_path.exists():
            return [], [], {}

        with tarfile.open(self._tar_path, 'r') as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue

                path_parts = Path(member.name).parts
                if member.name.lower().endswith(self._extensions):
                    # Assume first part of path is the class if nested
                    if len(path_parts) > 1:
                        cls_name = path_parts[-2] # Parent directory name
                        class_names.add(cls_name)
                        samples.append((member.name, cls_name))
                    else:
                        samples.append((member.name, 'default'))
                        class_names.add('default')

        classes = sorted(list(class_names))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        final_samples = [
            (name, class_to_idx[cls_name]) for name, cls_name in samples
        ]

        return final_samples, classes, class_to_idx

    def __len__(self) -> int:
        '''
        Returns the total number of images in the tar archive.

        Returns:
            int: Number of image samples.
        '''
        return len(self._samples)

    def __getitem__(self, idx: int) -> ImageDatasetItem:
        '''
        Retrieves the image item from the tar archive at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            ImageDatasetItem: Data class containing image, label, and optionally path.
        '''
        member_name, label = self._samples[idx]

        try:
            if self._tar_handle is None:
                self._tar_handle = tarfile.open(self._tar_path, 'r')

            member = self._tar_handle.getmember(member_name)
            f = self._tar_handle.extractfile(member)
            if f is None:
                raise RuntimeError(f'Could not extract {member_name}')
            
            image_data = f.read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            f.close()
        except Exception as error:
            raise RuntimeError(f'Failed to load {member_name} from {self._tar_path}: {error}') from error

        if self._transforms is not None:
            image = self._transforms(image)

        return ImageDatasetItem(
            image=image,
            label=label,
            path=Path(member_name) if self._return_path else None
        )

    def __getstate__(self) -> Dict[str, Any]:
        '''
        Prepares the state for pickling, ensuring the file handle is excluded.
        
        Returns:
            Dict[str, Any]: The object's state dictionary without the tar handle.
        '''
        state = self.__dict__.copy()
        state['_tar_handle'] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        '''
        Restores the object state after unpickling.
        
        Args:
            state (Dict[str, Any]): The unpickled state dictionary.
        '''
        self.__dict__.update(state)

    def __del__(self) -> None:
        '''
        Ensures the tar file handle is closed upon object destruction.
        '''
        if getattr(self, '_tar_handle', None) is not None:
            self._tar_handle.close()


class ParquetImageDataset(CodonDataset):
    '''
    A map-style dataset that streams images out of one or more Parquet shards.

    Rows are addressed globally across every shard, but only the row group that
    contains the requested row is ever materialized, so arbitrarily large shards
    stay usable without preloading them. Materialized row groups are kept in a
    bounded LRU cache, which makes sequential access (the usual DataLoader case)
    hit the cache almost every time.

    The image cell may be stored in any of the formats commonly produced by
    WebDataset-style pipelines:

    * raw encoded bytes (``binary``) containing PNG/JPEG/WebP payload,
    * a ``struct`` with ``bytes`` / ``path`` fields (the HuggingFace
      ``datasets.Image`` layout),
    * a plain string/bytes path pointing at an image file on disk.

    Typical schemas::

        # 1) raw bytes + integer label
        pa.schema([('image', pa.binary()), ('label', pa.int64())])

        # 2) HF-style nested image struct + string label
        pa.schema([
            ('image', pa.struct([('bytes', pa.binary()), ('path', pa.string())])),
            ('label', pa.string()),
        ])

    Attributes:
        _path (Union[Path, List[Path]]): The configured source (file/dir/list).
        _image_key (str): Column holding the encoded image.
        _label_key (Optional[Union[str, List[str]]]): Column(s) holding labels.
        _path_key (Optional[str]): Column holding a per-row identifier/path.
        _file_paths (List[Path]): Resolved parquet shards, in read order.
        _file_rows (List[int]): Row count of each shard.
        _file_offsets (List[int]): Global first-row index of each shard.
        _group_meta (List[Tuple[int, int, int, int]]): Per row group tuple of
            ``(file_index, group_index, first_global_row, num_rows)``.
        _group_starts (List[int]): First global row of every row group, used
            for bisect lookups.
        _classes (Optional[List[Any]]): Sorted class names, if labels are
            categorical.
        _class_to_idx (Optional[Dict[Any, int]]): Class name to index mapping.
        _samples (Optional[List[Any]]): Lazily materialized raw label per row.
    '''

    def __init__(
        self,
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
        max_images: Optional[int] = None,
    ) -> None:
        '''
        Initializes the ParquetImageDataset.

        Args:
            path (Union[str, Path, List[str], List[Path]]): A ``.parquet`` file,
                a directory containing ``*.parquet`` shards, or an explicit
                sequence of files/directories.
            image_key (Optional[Union[str, Sequence[str]]]): Column holding the
                encoded image bytes, the image ``struct`` (``bytes``/``path``), an
                image file path, or a list of any of those. Several names read
                several image columns per row. When None, the column is detected
                from the schema and from a sample of the data; see
                :meth:`detect_image_keys`.
            label_key (Optional[Union[str, List[str]]]): Column(s) holding the
                label. A list reads several columns into a label vector. If
                None, the returned label is None.
            transforms (Optional[Compose]): A composition of torchvision
                transforms applied to each decoded PIL image.
            return_path (bool): If True, ``path_key`` values are returned in
                ``ImageDatasetItem.path``.
            path_key (Optional[str]): Column holding a per-row identifier (file
                name, URL, relative path). Used by ``return_path``.
            classes (Optional[Sequence[Any]]): Explicit class list. Any label
                whose string form is in the list is mapped to its index,
                regardless of ``index_mode``.
            index_mode (bool): If True (default), already-categorical integer
                labels (e.g. ``0..9``) are returned verbatim and no class
                mapping is built. If False, a sorted class list is derived from
                the label column and integer labels are remapped to indices.
            mode (Optional[str]): PIL mode to convert decoded images to
                (e.g. ``'RGB'``). Defaults to None, which keeps the decoded
                mode untouched.
            loader (Optional[Callable]): Custom loader used when the payload of
                ``image_key`` is a string/bytes path rather than encoded image
                data. Defaults to :func:`default_loader`.
            cache_size (int): Number of materialized row groups kept in memory.
                Defaults to 2. Use 0 to disable caching.
            columns (Optional[Sequence[str]]): Extra columns to read alongside
                the image and label columns, exposed through ``get_row``.
            num_threads (int): Threads used by :meth:`prefetch` for parallel
                row-group warm-up. Defaults to 4.
            cache_metadata (bool): If True, caches the resolved shard list,
                schema, and class mapping to disk so re-initialization is
                instant for large sharded datasets.
            multi_image (str): What to do when a row carries several images (a
                list column, or several ``image_key`` columns): ``'auto'``
                (default) stacks them into a ``[N, C, H, W]`` tensor when the
                transforms already yield equally shaped tensors and otherwise
                returns a list, ``'stack'`` always stacks (converting to tensors
                when needed), and ``'list'`` always returns a list. Single-image
                rows are unaffected and keep their plain shape.
            max_images (Optional[int]): Cap on the number of images taken from
                one row. Defaults to None (no cap).

        Raises:
            FileNotFoundError: If no parquet shard can be resolved.
            ValueError: If the dataset is empty, a key is unusable, ``label_key``
                has a bad shape, ``multi_image`` is unknown, or no image column
                can be detected.
            KeyError: If ``image_key``, ``label_key``, or ``path_key`` is absent
                from the parquet schema.
        '''
        super().__init__()
        self._path = path
        self._label_key = label_key
        self._path_key = path_key
        self._transforms = transforms
        self._return_path = return_path
        self._index_mode = index_mode
        self._mode = mode
        self._loader = loader or default_loader
        self._cache_size = max(0, int(cache_size))
        self._num_threads = max(1, int(num_threads))
        self._cache_metadata = cache_metadata
        self._extra_columns = list(columns) if columns else []

        if label_key is not None and not isinstance(label_key, (str, list, tuple)):
            raise ValueError(
                f'label_key must be a column name or a list of column names, '
                f'got {type(label_key).__name__}.'
            )
        if multi_image not in ('auto', 'stack', 'list'):
            raise ValueError(
                f"multi_image must be one of 'auto', 'stack', or 'list', got {multi_image!r}."
            )
        if max_images is not None and int(max_images) < 1:
            raise ValueError(f'max_images must be >= 1 or None, got {max_images}.')
        self._multi_image = multi_image
        self._max_images = int(max_images) if max_images is not None else None

        # An explicit key may name one column or several image columns.
        self._auto_image_key = image_key is None
        self._image_keys: List[str] = self._normalize_image_keys(image_key)

        # Column selection: keep each materialized row group as small as possible.
        # An auto-detected image column is appended by `_resolve_files`.
        keys = list(self._image_keys)
        if isinstance(self._label_key, str):
            keys.append(self._label_key)
        elif isinstance(self._label_key, (list, tuple)):
            if not self._label_key:
                raise ValueError('label_key was given as an empty list.')
            keys.extend(self._label_key)
        if self._path_key:
            keys.append(self._path_key)
        keys.extend(self._extra_columns)
        self._read_columns = list(dict.fromkeys(keys))

        # Process-local runtime state (never pickled).
        self._cache: 'OrderedDict[Tuple[int, int], Any]' = OrderedDict()
        self._handles: Optional[Dict[int, pq.ParquetFile]] = None
        self._handle_pid: Optional[int] = None
        self._thread_handles: Dict[Tuple[int, int], pq.ParquetFile] = {}
        self._io_lock = threading.Lock()

        # Global row addressing.
        self._file_paths: List[Path] = []
        self._file_rows: List[int] = []
        self._file_offsets: List[int] = []
        self._file_row_groups: List[int] = []
        self._group_meta: List[Tuple[int, int, int, int]] = []
        self._group_starts: List[int] = []
        # View addressing: None means "the whole source".
        self._row_offset: int = 0
        self._row_count: Optional[int] = None

        # Label bookkeeping.
        self._classes: Optional[List[Any]] = None
        self._class_to_idx: Optional[Dict[Any, int]] = None
        self._samples: Optional[List[Any]] = None
        self._label_min: Optional[int] = None
        self._label_max: Optional[int] = None

        if not self._restore_metadata_cache(classes):
            self._resolve_files()
            self._scan_index(classes)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_entries(path: Any) -> List[Path]:
        '''
        Expands a configured path into an ordered list of parquet shards.

        Args:
            path (Any): A single file/directory or a sequence of them.

        Returns:
            List[Path]: Candidate shard paths (not yet validated).

        Raises:
            FileNotFoundError: If an entry does not exist.
        '''
        entries: Iterable[Any]
        if isinstance(path, (list, tuple, set)):
            entries = list(path)
        else:
            entries = [path]

        candidates: List[Path] = []
        for entry in entries:
            entry_path = Path(entry)
            if entry_path.is_dir():
                candidates.extend(sorted(p for p in entry_path.glob('*.parquet') if p.is_file()))
            elif entry_path.is_file():
                candidates.append(entry_path)
            else:
                raise FileNotFoundError(f'No such parquet file or directory: {entry_path}')
        return candidates

    def _normalize_image_keys(self, image_key: Optional[Union[str, Sequence[str]]]) -> List[str]:
        '''
        Normalizes the configured image key into a list of column names.

        Args:
            image_key (Optional[Union[str, Sequence[str]]]): One column name, a
                sequence of column names, or None for auto-detection.

        Returns:
            List[str]: The requested column names, empty when auto-detecting.

        Raises:
            ValueError: If the value is neither a string nor a sequence.
        '''
        if image_key is None:
            return []
        if isinstance(image_key, str):
            return [image_key]
        if isinstance(image_key, (list, tuple)):
            if not image_key:
                raise ValueError('image_key was given as an empty sequence.')
            keys: List[str] = []
            for key in image_key:
                if not isinstance(key, str):
                    raise ValueError(
                        f'image_key entries must be column names, got {type(key).__name__}.'
                    )
                if key not in keys:
                    keys.append(key)
            return keys
        raise ValueError(
            f'image_key must be a column name, a sequence of column names, or None, '
            f'got {type(image_key).__name__}.'
        )

    @property
    def image_keys(self) -> List[str]:
        '''
        Returns the image columns in use.

        Returns:
            List[str]: One or more column names, detected ones included.
        '''
        return list(self._image_keys)

    def detect_image_keys(
        self,
        candidates: Optional[Sequence[str]] = None,
        probe_rows: int = 64,
        strict: bool = True,
    ) -> List[str]:
        '''
        Finds the image column(s) of the dataset.

        Detection is schema-driven: only ``binary``, ``struct``, and ``string``
        columns are considered. Each candidate is then scored from a small sample
        of real values plus its name, so a ``caption`` string column or a
        ``thumbnail_hash`` binary column is not mistaken for image data. Binary
        columns are validated by decoding a sample payload, and string columns by
        checking for an image suffix or URL prefix.

        Args:
            candidates (Optional[Sequence[str]]): Columns to consider. Defaults to
                the schema.
            probe_rows (int): Number of values read per column while scoring.
                Defaults to 64.
            strict (bool): If True, raise when nothing looks like an image. If
                False, return an empty list instead.

        Returns:
            List[str]: Detected column names, best first. Several names are
            returned when the data carries the same image in more than one
            (equally plausible) column.

        Raises:
            FileNotFoundError: If no shard can be resolved.
            ValueError: If no image column is found and ``strict`` is True.
        '''
        if not self._file_paths:
            self._file_paths = self._expand_entries(self._path)
            if not self._file_paths:
                raise FileNotFoundError(f'No .parquet shards found under: {self._path}')

        ranked: List[Tuple[float, str]] = []
        with pq.ParquetFile(self._file_paths[0]) as parquet_file:
            schema = parquet_file.schema_arrow
            if not candidates:
                wanted = [
                    field.name for field in schema
                    if pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type)
                    or pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
                    or pa.types.is_struct(field.type)
                    or pa.types.is_list(field.type) or pa.types.is_large_list(field.type)
                ]
            else:
                wanted = list(candidates)
            sample = self._read_candidate_sample(parquet_file, wanted, probe_rows)

            for name in wanted:
                if name not in schema.names:
                    continue
                field_type = schema.field(name).type
                values = sample.get(name) or []

                kind = 'unknown'
                detected = 0
                for value in values:
                    cell_kind = image_cell_kind(value, probe=True)
                    if cell_kind in ('image', 'struct'):
                        kind = 'struct' if kind == 'unknown' else 'image'
                        detected += 1
                    elif cell_kind == 'path':
                        kind = 'path' if kind == 'unknown' else kind
                        detected += 1

                score = float(detected)
                if pa.types.is_struct(field_type):
                    score += 4.0
                elif pa.types.is_binary(field_type) or pa.types.is_large_binary(field_type):
                    score += 2.0
                elif pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
                    score += 0.5
                elif pa.types.is_list(field_type) or pa.types.is_large_list(field_type):
                    score += 2.0
                elif pa.types.is_null(field_type):
                    continue
                else:
                    continue

                if kind in ('image', 'struct', 'path'):
                    score += 2.0
                score += self._image_key_name_bonus(name)
                ranked.append((score, name))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        best = ranked[0][0] if ranked else 0.0
        detected = [name for score, name in ranked if score >= max(3.0, best - 0.5)]

        if not detected and strict:
            raise ValueError(
                f'Could not detect an image column in {self._file_paths[0]}. '
                f'Available columns: {sorted(wanted)}. Pass image_key=... explicitly.'
            )
        return detected

    @staticmethod
    def _image_key_name_bonus(name: str) -> float:
        '''
        Scores how much a column name suggests image data.

        Args:
            name (str): The column name.

        Returns:
            float: A positive bonus for image-like names, negative for metadata.
        '''
        lowered = name.lower()
        bonus = 0.0
        for token in ('image', 'img', 'pixel', 'photo', 'picture', 'thumbnail', 'jpeg', 'jpg', 'png'):
            if token in lowered:
                bonus += 2.0
                break
        for token in ('label', 'target', 'class', 'caption', 'text', 'hash', 'size', 'id', 'shape', 'meta'):
            if token in lowered:
                bonus -= 1.5
                break
        return bonus

    @staticmethod
    def _read_candidate_sample(
        parquet_file: pq.ParquetFile,
        candidates: Sequence[str],
        probe_rows: int,
    ) -> Dict[str, List[Any]]:
        '''
        Reads a small sample of the candidate columns from the first row group.

        Args:
            parquet_file (pq.ParquetFile): The shard to sample.
            candidates (Sequence[str]): Column names to read.
            probe_rows (int): Maximum values taken per column.

        Returns:
            Dict[str, List[Any]]: Sampled values per column.
        '''
        sample: Dict[str, List[Any]] = {}
        if parquet_file.num_row_groups == 0 or not candidates:
            return sample
        try:
            batch = parquet_file.read_row_group(0, columns=list(candidates))
        except Exception:
            return sample
        limit = max(1, int(probe_rows))
        for name in candidates:
            if name in batch.column_names:
                sample[name] = batch.column(name).slice(0, limit).to_pylist()
        return sample

    def _resolve_files(self) -> None:
        '''
        Resolves the configured path into an ordered list of parquet shards,
        detects the image column when none was given, and validates that every
        requested column is present in every schema.

        Raises:
            FileNotFoundError: If nothing usable is found.
            ValueError: If the resolved shards contain no rows, or no image
                column can be detected.
            KeyError: If a requested column is missing from a schema.
        '''
        candidates = self._expand_entries(self._path)
        if not candidates:
            raise FileNotFoundError(f'No .parquet shards found under: {self._path}')

        self._file_paths = candidates
        self._file_rows = []
        self._file_row_groups = []

        if self._auto_image_key:
            self._image_keys = self.detect_image_keys()

        ordered: List[str] = list(self._image_keys)
        if isinstance(self._label_key, str):
            ordered.append(self._label_key)
        elif self._label_key is not None:
            ordered.extend(self._label_key)
        if self._path_key:
            ordered.append(self._path_key)
        ordered.extend(self._extra_columns)
        self._read_columns = list(dict.fromkeys(ordered))

        for file_path in self._file_paths:
            parquet_file = pq.ParquetFile(file_path)
            try:
                schema_names = set(parquet_file.schema_arrow.names)
                missing = [key for key in self._read_columns if key not in schema_names]
                if missing:
                    raise KeyError(
                        f'Column(s) {missing} not found in {file_path}. '
                        f'Available columns: {sorted(schema_names)}'
                    )
                self._file_rows.append(parquet_file.metadata.num_rows)
                self._file_row_groups.append(parquet_file.num_row_groups)
            finally:
                parquet_file.close()

        if sum(self._file_rows) == 0:
            raise ValueError(f'The resolved parquet shards contain no rows: {self._path}')

    def _scan_index(self, classes: Optional[Sequence[Any]] = None) -> None:
        '''
        Builds the global row-group index and resolves the class mapping.

        Row groups from every shard are indexed into one global address space so
        a row index maps to exactly one (file, row group, offset) triple. Only
        the label column is read, and only when a class mapping is required.

        Args:
            classes (Optional[Sequence[Any]]): Explicit class list, if any.
        '''
        self._file_offsets = []
        self._group_meta = []
        self._group_starts = []

        running_row = 0
        for file_index, row_count in enumerate(self._file_rows):
            self._file_offsets.append(running_row)
            if row_count == 0:
                continue
            with pq.ParquetFile(self._file_paths[file_index]) as parquet_file:
                local_row = 0
                for group_index in range(parquet_file.num_row_groups):
                    group_rows = parquet_file.metadata.row_group(group_index).num_rows
                    self._group_meta.append((file_index, group_index, running_row, group_rows))
                    self._group_starts.append(running_row)
                    running_row += group_rows
                    local_row += group_rows
                if local_row != row_count:
                    raise ValueError(
                        f'Row group metadata of {self._file_paths[file_index]} is '
                        f'inconsistent with its row count.'
                    )

        self._build_classes(classes)

    def _build_classes(self, classes: Optional[Sequence[Any]] = None) -> None:
        '''
        Determines whether labels are categorical and prepares the mapping.

        Args:
            classes (Optional[Sequence[Any]]): Explicit class list, if any.
        '''
        self._classes = None
        self._class_to_idx = None
        self._samples = None
        self._label_min = None
        self._label_max = None

        if self._label_key is None:
            return

        if isinstance(self._label_key, (list, tuple)):
            if classes is not None:
                self._classes = [str(cls) for cls in classes]
            return  # Multi-column label vectors stay numeric.

        if classes is not None:
            self._classes = [str(cls) for cls in classes]
            self._class_to_idx = {name: idx for idx, name in enumerate(self._classes)}
            return

        if self._index_mode:
            # Integer labels are already dense class ids; remember their range so
            # `num_classes` and `sample_weights()` work without a second pass.
            samples = self._read_label_column()
            if samples is not None:
                self._samples = samples
            return

        unique_labels = self._unique_labels(self._label_key)
        if not unique_labels:
            return
        if all(isinstance(value, int) and not isinstance(value, bool) for value in unique_labels):
            # Numeric labels in non-index mode are passed through unchanged.
            return

        self._classes = sorted(unique_labels, key=lambda value: str(value))
        self._class_to_idx = {name: idx for idx, name in enumerate(self._classes)}

    def _unique_labels(self, column: str) -> List[Any]:
        '''
        Collects the distinct values of one label column across every shard.

        Args:
            column (str): The label column name.

        Returns:
            List[Any]: Distinct label values (unsorted).
        '''
        unique: Dict[Any, None] = {}
        for file_index, file_path in enumerate(self._file_paths):
            if self._file_rows[file_index] == 0:
                continue
            with pq.ParquetFile(file_path) as parquet_file:
                for group_index in range(parquet_file.num_row_groups):
                    batch = parquet_file.read_row_group(group_index, columns=[column])
                    for value in batch.column(column).to_pylist():
                        if isinstance(value, (list, tuple, dict)):
                            value = _freeze(value)
                        try:
                            unique.setdefault(value, None)
                        except TypeError:
                            continue
        return list(unique.keys())

    # ------------------------------------------------------------------
    # metadata cache
    # ------------------------------------------------------------------

    @staticmethod
    def _metadata_cache_path_for(files: List[Path]) -> Path:
        '''
        Computes the on-disk location of the metadata cache.

        Args:
            files (List[Path]): Resolved parquet shards.

        Returns:
            Path: A path next to the single shard, or a hashed name inside the
                system temporary directory for sharded datasets.
        '''
        if len(files) == 1:
            return files[0].with_suffix('.codon_pqmeta.pkl')
        digest = hashlib.sha1(
            '|'.join(str(p) for p in files).encode('utf-8')
        ).hexdigest()[:16]
        return Path(tempfile.gettempdir()) / f'codon_pqimage_{digest}.pkl'

    def _metadata_cache_path(self) -> Path:
        '''
        Returns the metadata cache location of the resolved shards.

        Returns:
            Path: The cache file path.
        '''
        return self._metadata_cache_path_for(self._file_paths)

    @staticmethod
    def _cache_key(files: List[Path]) -> Tuple[str, float]:
        '''
        Builds a validity key covering the shard paths, sizes, and mtimes.

        Args:
            files (List[Path]): Resolved parquet shards.

        Returns:
            Tuple[str, float]: A hash of the shard list plus the newest mtime.
        '''
        joined = '|'.join(f'{p}:{p.stat().st_size}' for p in files)
        return hashlib.sha1(joined.encode('utf-8')).hexdigest(), max(
            p.stat().st_mtime for p in files
        )

    def _restore_metadata_cache(self, classes: Optional[Sequence[Any]] = None) -> bool:
        '''
        Resolves files and the class mapping from the on-disk cache when valid.

        Args:
            classes (Optional[Sequence[Any]]): Explicit class list, if any.

        Returns:
            bool: True if the cache satisfied the whole index build.
        '''
        if not self._cache_metadata:
            return False

        try:
            files = self._expand_entries(self._path)
        except FileNotFoundError:
            return False

        if not files:
            # Let the regular path raise the descriptive error.
            return False

        cache_path = self._metadata_cache_path_for(files)
        if not cache_path.exists():
            return False

        try:
            with open(cache_path, 'rb') as handle:
                payload = pickle.load(handle)
        except Exception:
            return False

        try:
            if payload.get('key') != self._cache_key(files):
                return False
            if payload.get('keys') != self._read_columns:
                return False
            if classes is not None and [str(cls) for cls in classes] != payload.get('classes'):
                return False
        except (OSError, KeyError, TypeError):
            return False

        self._file_paths = files
        self._apply_cached_index(payload)
        return True

    def _apply_cached_index(self, payload: Dict[str, Any]) -> None:
        '''
        Installs a cached index payload into this instance.

        Args:
            payload (Dict[str, Any]): Payload produced by :meth:`refresh_metadata_cache`.
        '''
        self._file_rows = list(payload['file_rows'])
        self._file_row_groups = list(payload['file_row_groups'])
        self._file_offsets = list(payload['file_offsets'])
        self._group_meta = [tuple(meta) for meta in payload['group_meta']]
        self._group_starts = list(payload['group_starts'])
        self._classes = payload.get('classes')
        self._class_to_idx = (
            {name: idx for idx, name in enumerate(self._classes)}
            if self._classes is not None else None
        )
        # Per-row labels are not part of the cache; they are re-derived lazily.
        self._samples = None
        self._label_min = None
        self._label_max = None

    def refresh_metadata_cache(self) -> Optional[Path]:
        '''
        Persists the resolved index so later constructions skip the full scan.

        Reads only parquet footers and the label column, so it is cheap next to
        decoding an image. Failures are non-fatal: a read-only dataset directory
        must never break dataset construction.

        Returns:
            Optional[Path]: The cache file written, or None when caching is
            disabled or the write failed.
        '''
        if not self._cache_metadata:
            return None
        try:
            payload = {
                'key': self._cache_key(self._file_paths),
                'keys': self._read_columns,
                'file_rows': self._file_rows,
                'file_row_groups': self._file_row_groups,
                'file_offsets': self._file_offsets,
                'group_meta': self._group_meta,
                'group_starts': self._group_starts,
                'classes': self._classes,
            }
            cache_path = self._metadata_cache_path()
            with open(cache_path, 'wb') as handle:
                pickle.dump(payload, handle)
            return cache_path
        except Exception:
            return None

    # ------------------------------------------------------------------
    # parquet access
    # ------------------------------------------------------------------

    def _get_handle(self, file_index: int) -> pq.ParquetFile:
        '''
        Returns the ParquetFile handle of one shard, valid for this process.

        DataLoader workers inherit the dataset through pickling, so handles are
        opened lazily and rebuilt whenever the owning process changes. Views
        created by :meth:`split` share the handle table, so an existing table is
        never invalidated in place — a new process simply installs a fresh one.

        Args:
            file_index (int): Index into ``_file_paths``.

        Returns:
            pq.ParquetFile: An open handle over that shard.
        '''
        pid = os.getpid()
        handles = self._handles
        if handles is None or self._handle_pid != pid:
            # First use in this process: the table may be missing (fresh state)
            # or stale (inherited through a fork), never a live local one.
            handles = {}
            self._handles = handles
            self._handle_pid = pid

        handle = handles.get(file_index)
        if handle is None:
            handle = pq.ParquetFile(self._file_paths[file_index])
            handles[file_index] = handle
        return handle

    def _thread_handle(self, file_index: int) -> pq.ParquetFile:
        '''
        Returns a handle private to the calling thread, used by :meth:`prefetch`.

        A single ``ParquetFile`` reader is not safe for concurrent reads, so the
        prefetch pool gives every worker thread its own reader for a shard. The
        pool is instance-wide (not thread-local) so :meth:`close` can release
        these readers even when it runs on another thread.

        Args:
            file_index (int): Index into ``_file_paths``.

        Returns:
            pq.ParquetFile: A handle private to the calling thread.
        '''
        pool = getattr(self, '_thread_handles', None)
        if pool is None:
            pool = {}
            self._thread_handles = pool

        key = (threading.get_ident(), file_index)
        handle = pool.get(key)
        if handle is None:
            handle = pq.ParquetFile(self._file_paths[file_index])
            with self._io_lock:
                pool[key] = handle
        return handle

    @staticmethod
    def _locate(index: int, starts: List[int]) -> int:
        '''
        Finds the entry whose range contains a global row index.

        Args:
            index (int): The global row index.
            starts (List[int]): Sorted first-row index of every entry.

        Returns:
            int: The entry position, or -1 when the index is out of range.
        '''
        position = bisect.bisect_right(starts, index) - 1
        return position

    def _row_group_table(self, file_index: int, group_index: int) -> Any:
        '''
        Reads (or reuses) the materialized table of one row group.

        Args:
            file_index (int): Index into ``_file_paths``.
            group_index (int): Row group position inside that shard.

        Returns:
            pa.Table: The row group restricted to the requested columns.
        '''
        if self._cache_size == 0:
            return self._get_handle(file_index).read_row_group(
                group_index, columns=self._read_columns, use_threads=False
            )

        cache_key = (file_index, group_index)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached

        table = self._get_handle(file_index).read_row_group(
            group_index, columns=self._read_columns, use_threads=False
        )
        with self._io_lock:
            self._cache[cache_key] = table
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return table

    def _locate_row(self, global_row: int) -> Tuple[int, int, int, int]:
        '''
        Maps an absolute row index onto its shard, row group, and offset.

        Args:
            global_row (int): The absolute row index inside the source data.

        Returns:
            Tuple[int, int, int, int]: ``(file_index, group_index, offset_in_group,
            rows_in_group)``.

        Raises:
            IndexError: If the index is out of range.
        '''
        if global_row < 0 or global_row >= self._total_rows():
            raise IndexError(
                f'Index {global_row} out of range (0-{self._total_rows() - 1})'
            )
        position = self._locate(global_row, self._group_starts)
        if position < 0:
            raise IndexError(f'Index {global_row} is not covered by this view')
        file_index, group_index, first_row, group_rows = self._group_meta[position]
        return file_index, group_index, global_row - first_row, group_rows

    def _total_rows(self) -> int:
        '''
        Returns the row count of the underlying source data.

        Returns:
            int: Total rows across all shards, regardless of any view subset.
        '''
        return sum(self._file_rows)

    def get_row(self, idx: int) -> Dict[str, Any]:
        '''
        Returns the raw column values of one row without decoding the image.

        Args:
            idx (int): The row index (negative values wrap around).

        Returns:
            Dict[str, Any]: Mapping of the configured image/label/path/extra
            columns to their Python values.
        '''
        return self._fetch_row(idx)[0]

    def _fetch_row(self, idx: int) -> Tuple[Dict[str, Any], Tuple[int, int, int, int]]:
        '''
        Reads one row and reports where it came from.

        Args:
            idx (int): The row index inside this instance (negative values wrap
                around).

        Returns:
            Tuple[Dict[str, Any], Tuple[int, int, int, int]]: The row values and
            the ``(file_index, group_index, offset, rows_in_group)`` location.

        Raises:
            IndexError: If the index is out of range.
        '''
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f'Index {idx} out of range (0-{len(self) - 1})')
        location = self._locate_row(idx + self._row_offset)
        file_index, group_index, offset, _ = location
        table = self._row_group_table(file_index, group_index)
        row: Dict[str, Any] = {}
        for column in self._read_columns:
            if column in table.column_names:
                row[column] = table.column(column)[offset].as_py()
        return row, location

    # ------------------------------------------------------------------
    # decoding helpers
    # ------------------------------------------------------------------

    def resolve_image_payload(self, cell: Any) -> Tuple[Optional[bytes], Optional[str]]:
        '''
        Normalizes an image cell into bytes or a filesystem path.

        Args:
            cell (Any): Raw value of the ``image_key`` column. Supports raw
                bytes, a ``{'bytes', 'path'}`` struct, a bare path, or a
                pre-decoded PIL image.

        Returns:
            Tuple[Optional[bytes], Optional[str]]: Encoded image bytes and/or the
            referenced path.

        Raises:
            TypeError: If the cell cannot be interpreted.
        '''
        if isinstance(cell, dict):
            payload = cell.get('bytes')
            if payload is None:
                payload = cell.get('data')
            reference = cell.get('path')
            if payload is None and reference is None:
                raise TypeError(
                    f'Image struct is missing both "bytes" and "path": '
                    f'keys={sorted(cell.keys())}'
                )
            return (
                bytes(payload) if isinstance(payload, (bytes, bytearray, memoryview)) else None,
                str(reference) if reference is not None else None,
            )

        if isinstance(cell, (bytes, bytearray, memoryview)):
            return bytes(cell), None

        if isinstance(cell, Image.Image):
            return None, None

        if cell is None:
            raise TypeError('The image cell is NULL in this row.')

        if isinstance(cell, str):
            return None, cell

        if isinstance(cell, (list, tuple)) and cell and all(isinstance(b, int) for b in cell):
            return bytes(cell), None

        if isinstance(cell, (torch.Tensor,)):
            return None, None

        try:
            import numpy as np
            if isinstance(cell, np.ndarray):
                return None, None
        except ImportError:
            pass

        raise TypeError(
            f'Unsupported image cell of type {type(cell).__name__}; expected '
            f'bytes, an image struct, or a path.'
        )

    def decode_image(self, cell: Any) -> Image.Image:
        '''
        Decodes one image cell into a PIL image.

        Args:
            cell (Any): Raw value of the ``image_key`` column.

        Returns:
            Image.Image: The decoded image, converted to ``mode`` when set.

        Raises:
            RuntimeError: If the payload is neither decodable bytes nor a
                readable path.
        '''
        if isinstance(cell, Image.Image):
            image = cell
        else:
            payload, reference = self.resolve_image_payload(cell)

            if payload is None and reference is not None:
                candidate = Path(reference)
                if not candidate.exists():
                    raise RuntimeError(f'Referenced image file does not exist: {reference}')
                image = self._loader(candidate)
            elif payload is None:
                raise RuntimeError(
                    f'Image cell of type {type(cell).__name__} carries neither '
                    f'encoded bytes nor a usable path.'
                )
            else:
                try:
                    with Image.open(io.BytesIO(payload)) as handle:
                        handle.load()
                        image = handle.copy()
                except Exception:
                    decoded = _opencv_decode(payload)
                    if decoded is None:
                        raise
                    image = decoded

        if self._mode is not None and image.mode != self._mode:
            image = image.convert(self._mode)
        return image

    def _decode_label(self, row: Dict[str, Any]) -> Any:
        '''
        Extracts and maps the label of one row.

        Args:
            row (Dict[str, Any]): Raw row values from :meth:`get_row`.

        Returns:
            Any: An int index, a label vector, the raw label value, or None.
        '''
        if self._label_key is None:
            return None

        if isinstance(self._label_key, (list, tuple)):
            return [row.get(key) for key in self._label_key]

        value = row.get(self._label_key)
        if self._class_to_idx is None:
            return value
        if not isinstance(value, (str, bytes, int, float, bool)) and value is not None:
            return value  # Unhashable/nested label: pass it through untouched.
        if value in self._class_to_idx:
            return self._class_to_idx[value]
        return self._class_to_idx.get(str(value), value)

    # ------------------------------------------------------------------
    # CodonDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        '''
        Returns the total number of rows across every shard.

        Returns:
            int: The number of image samples (just this worker's rows for a
            :meth:`split` view).
        '''
        if self._row_count is not None:
            return self._row_count
        return sum(self._file_rows)

    def __getitem__(self, idx: int) -> ImageDatasetItem:
        '''
        Retrieves one decoded (and optionally transformed) image item.

        A row that carries several images is assembled according to
        ``multi_image``: stacked into a ``[N, C, H, W]`` tensor under ``'auto'``
        (when the transforms already produced tensors) or ``'stack'``, and
        returned as a list under ``'list'``. Single-image rows always keep their
        plain shape.

        Args:
            idx (int): The row index inside this instance (negative values wrap
                around).

        Returns:
            ImageDatasetItem: Dataclass with the decoded image(s), label, and the
            absolute origin of the row (``row`` is the row number in the source
            data, even for a :meth:`split` view).

        Raises:
            IndexError: If the index is out of range.
            RuntimeError: If an image cannot be decoded.
        '''
        if idx < 0:
            idx += len(self)

        row, (file_index, group_index, offset, _) = self._fetch_row(idx)
        images, keys, sub_index = self.decode_row(
            row,
            error_context=(
                f'row {idx} ({self._file_paths[file_index]}, row group '
                f'{group_index}, offset {offset})'
            ),
        )

        label = self._decode_label(row)
        return ImageDatasetItem(
            image=self._combine(images),
            label=label,
            path=(row.get(self._path_key) if self._return_path and self._path_key else None),
            path_key=self._path_key,
            row=idx + self._row_offset,
            parquet_path=self._file_paths[file_index],
            key=keys[0] if len(keys) == 1 else None,
            sub_index=sub_index,
            image_keys=tuple(keys),
        )

    def decode_row(
        self,
        row: Dict[str, Any],
        error_context: str = 'row',
    ) -> Tuple[List[Any], List[str], Optional[int]]:
        '''
        Decodes, transforms, and combines every image carried by one raw row.

        Args:
            row (Dict[str, Any]): Raw column values from :meth:`get_row`.
            error_context (str): Description used in decode errors, e.g.
                ``'row 12 (shard.parquet, row group 0, offset 3)'``.

        Returns:
            Tuple[List[Any], List[str], Optional[int]]: The decoded images (a
            single-element list is collapsed to a plain image by the caller's
            policy), the contributing column names, and the sub-index when one
            column carried several images.

        Raises:
            RuntimeError: If an image cannot be decoded.
        '''
        decoded: List[Any] = []
        keys: List[str] = []
        sub_index: Optional[int] = None

        for key in self._image_keys:
            payloads = image_cell_values(row.get(key))
            if not payloads:
                continue
            keys.append(key)
            if len(payloads) > 1:
                sub_index = 0
            for position, payload in enumerate(payloads):
                try:
                    image = self.decode_image(payload)
                except Exception as error:
                    raise RuntimeError(
                        f'Failed to decode image at {error_context} '
                        f'(column {key!r}, index {position}): {error}'
                    ) from error
                decoded.append(self._apply_transforms(image))

        return decoded, keys, sub_index

    def _apply_transforms(self, image: Any) -> Any:
        '''
        Applies the configured transforms to one decoded image.

        Args:
            image (Image.Image): The decoded image.

        Returns:
            Any: The transformed image or tensor.
        '''
        if self._transforms is not None:
            return self._transforms(image)
        return image

    def _combine(self, images: List[Any]) -> Any:
        '''
        Applies the ``multi_image`` policy to the images of one row.

        Args:
            images (List[Any]): Decoded (and transformed) images, in column order.

        Returns:
            Any: A single image, a stacked ``[N, C, H, W]`` tensor, or a list.

        Raises:
            RuntimeError: If ``multi_image='stack'`` receives images that cannot
            be stacked into one tensor.
        '''
        if self._max_images is not None:
            images = images[:self._max_images]

        if not images:
            raise RuntimeError('The row carries no images; check image_key and the data.')
        if len(images) == 1:
            return images[0]
        if self._multi_image == 'list':
            return images
        if self._multi_image == 'auto' and not all(
            isinstance(image, torch.Tensor) for image in images
        ):
            return images
        return self.stack(images)

    @staticmethod
    def stack(images: Sequence[Any]) -> torch.Tensor:
        '''
        Stacks decoded images into one ``[N, C, H, W]`` tensor.

        PIL images and arrays are converted with ``ToTensor`` and
        ``torch.as_tensor`` first, so stacking works with or without transforms.

        Args:
            images (Sequence[Any]): Images to stack.

        Returns:
            torch.Tensor: The stacked batch.

        Raises:
            RuntimeError: If the images do not share a shape.
        '''
        tensors: List[torch.Tensor] = []
        for image in images:
            if isinstance(image, torch.Tensor):
                tensors.append(image)
            elif isinstance(image, Image.Image):
                tensors.append(ToTensor()(image))
            else:
                tensors.append(torch.as_tensor(image))

        shapes = {tuple(tensor.shape) for tensor in tensors}
        if len(shapes) != 1:
            raise RuntimeError(
                f'Cannot stack images of differing shapes {sorted(shapes)}; resize '
                f'them in `transforms`, or use multi_image="list".'
            )
        return torch.stack(tensors)

    # ------------------------------------------------------------------
    # convenience accessors
    # ------------------------------------------------------------------

    @property
    def classes(self) -> Optional[List[Any]]:
        '''
        Returns the resolved class list, if labels are categorical.

        Returns:
            Optional[List[Any]]: Class names, or None for free-form labels.
        '''
        return self._classes

    @property
    def class_to_idx(self) -> Optional[Dict[Any, int]]:
        '''
        Returns the class name to index mapping, if any.

        Returns:
            Optional[Dict[Any, int]]: The mapping, or None.
        '''
        return self._class_to_idx

    @property
    def label_key(self) -> Optional[Union[str, List[str]]]:
        '''
        Returns the configured label column(s).

        Returns:
            Optional[Union[str, List[str]]]: The label key configuration.
        '''
        return self._label_key

    @property
    def file_paths(self) -> List[Path]:
        '''
        Returns the resolved parquet shards in read order.

        Returns:
            List[Path]: Shard paths.
        '''
        return list(self._file_paths)

    @property
    def num_classes(self) -> Optional[int]:
        '''
        Returns the number of classes when the dataset is categorical.

        Returns:
            Optional[int]: Class count, or None.
        '''
        if self._classes is not None:
            return len(self._classes)
        if self._index_mode and self._samples is not None and self._label_min is not None:
            if 0 <= self._label_min <= self._label_max:
                return int(self._label_max) + 1
        return None

    def __iter__(self):
        '''
        Yields every item in global row order.

        Returns:
            Iterator[ImageDatasetItem]: Item iterator.
        '''
        for idx in range(len(self)):
            yield self[idx]

    def sample_weights(self) -> Optional[torch.Tensor]:
        '''
        Computes per-row inverse-frequency weights for balanced sampling.

        Requires a categorical label (or ``index_mode`` integer labels in
        ``0..k``). The raw label column is materialized on first use and kept.

        Returns:
            Optional[torch.Tensor]: A float tensor of shape ``[len(self)]``, or
            None when labels are unavailable or not categorical.
        '''
        if self._label_key is None or isinstance(self._label_key, (list, tuple)):
            return None
        if self._samples is None:
            self._samples = self._read_label_column()
        if self._samples is None:
            return None
        if self._class_to_idx is None:
            if self._label_min is None or self._label_min < 0:
                return None
            counts = torch.bincount(torch.as_tensor(self._samples, dtype=torch.long))
        else:
            if any(sample not in self._class_to_idx for sample in self._samples):
                return None
            counts = torch.bincount(
                torch.as_tensor(
                    [self._class_to_idx[sample] for sample in self._samples], dtype=torch.long
                )
            )
        counts = counts.clamp(min=1)
        per_sample = (1.0 / counts.float())
        if self._class_to_idx is not None:
            labels = torch.as_tensor(
                [self._class_to_idx[sample] for sample in self._samples], dtype=torch.long
            )
        else:
            labels = torch.as_tensor(self._samples, dtype=torch.long)
        return per_sample[labels]

    def _read_label_column(self) -> Optional[List[Any]]:
        '''
        Materializes the raw label column of every row.

        In ``index_mode`` the labels are also validated as dense integers and
        their range is recorded for :attr:`num_classes` and
        :meth:`sample_weights`; a non-integer label discards the result so the
        values simply pass through.

        Returns:
            Optional[List[Any]]: Raw labels, or None when unavailable or (in
            index mode) not integer-valued.
        '''
        if self._label_key is None or isinstance(self._label_key, (list, tuple)):
            return None
        labels: List[Any] = []
        minimum: Optional[int] = None
        maximum: Optional[int] = None
        for file_index, file_path in enumerate(self._file_paths):
            if self._file_rows[file_index] == 0:
                continue
            with pq.ParquetFile(file_path) as parquet_file:
                for group_index in range(parquet_file.num_row_groups):
                    batch = parquet_file.read_row_group(group_index, columns=[self._label_key])
                    labels.extend(batch.column(self._label_key).to_pylist())

        if not self._index_mode:
            self._samples = labels
            return labels

        for value in labels:
            if isinstance(value, bool) or not isinstance(value, int):
                self._samples = None
                self._label_min = None
                self._label_max = None
                return None
            minimum = value if minimum is None else min(minimum, value)
            maximum = value if maximum is None else max(maximum, value)

        self._label_min = minimum
        self._label_max = maximum
        return labels

    def split(self, rank: int, world_size: int) -> 'ParquetImageDataset':
        '''
        Returns a non-overlapping slice of this dataset for one worker.

        The view shares the underlying parquet handles and cache, so it costs no
        extra memory. Intended for multi-worker / distributed training.

        Args:
            rank (int): Zero-based index of this worker.
            world_size (int): Total number of workers.

        Returns:
            ParquetImageDataset: A view covering this worker's rows.

        Raises:
            ValueError: If ``rank`` or ``world_size`` is out of range.
        '''
        if world_size < 1:
            raise ValueError(f'world_size must be >= 1, got {world_size}')
        if not 0 <= rank < world_size:
            raise ValueError(f'rank must be in [0, {world_size}), got {rank}')

        view = object.__new__(ParquetImageDataset)
        view.__dict__.update(self.__dict__)
        total = len(self)
        start = total * rank // world_size
        end = total * (rank + 1) // world_size
        view._row_offset = self._row_offset + start
        view._row_count = end - start
        return view

    def collate_dict(self) -> Callable[[ImageDatasetItem], Dict[str, Any]]:
        '''
        Returns a collate function that turns items into plain dicts.

        Useful when the training loop prefers a mapping over the dataclass; the
        default :meth:`compose` collation already produces batched dataclasses.
        Only populated keys are emitted, so ``default_collate`` never sees a
        ``None`` field::

            loader = dataset.compose(collate_fn=dataset.collate_dict()).loader(4)

        Returns:
            Callable[[ImageDatasetItem], Dict[str, Any]]: The collate function.
        '''
        def _collate(item: ImageDatasetItem) -> Dict[str, Any]:
            batch = {'image': item.image}
            if item.label is not None:
                batch['label'] = item.label
            if item.path is not None:
                batch['path'] = str(item.path)
            if item.row is not None:
                batch['row'] = item.row
            if item.parquet_path is not None:
                batch['parquet_path'] = str(item.parquet_path)
            return batch

        return _collate

    def compose(self, collate_fn: Optional[Callable] = None, **kwargs: Any) -> Any:
        '''
        Wraps the dataset for PyTorch.

        Batches are collated by :func:`collate_image_items` unless ``collate_fn``
        is given, so a plain ``dataset.compose().loader(...)`` yields batches whose
        fields mirror :class:`ImageDatasetItem`.

        Args:
            collate_fn (Optional[Callable]): Per-item function, passed through to
                :meth:`CodonDataset.compose`.
            **kwargs (Any): Forwarded to :meth:`CodonDataset.compose`.

        Returns:
            TorchDatasetWrapper: The wrapped dataset.
        '''
        return super().compose(collate_fn=collate_fn, **kwargs)

    def prefetch(self, num_rows: Optional[int] = None) -> int:
        '''
        Warms the row-group cache in parallel using per-thread file handles.

        Useful when the first epoch would otherwise pay for cold reads on every
        worker at once. Only complete row groups are read; already cached groups
        are skipped.

        Args:
            num_rows (Optional[int]): Stop after covering this many rows.
                Defaults to every row.

        Returns:
            int: The number of row groups read (0 means the cache was warm).
        '''
        limit = len(self) if num_rows is None else min(int(num_rows), len(self))
        read_count = 0
        targets: List[Tuple[int, Any]] = []
        for meta in self._group_meta:
            # Row groups carry absolute positions; a split view shifts them.
            if meta[2] - self._row_offset >= limit:
                break
            if self._cache_size and (meta[0], meta[1]) in self._cache:
                continue
            targets.append((meta[0], meta[1]))

        if not targets:
            return 0

        def _read(target: Tuple[int, Any]) -> Any:
            file_index, group_index = target
            handle = self._thread_handle(file_index)
            return target, handle.read_row_group(
                group_index, columns=self._read_columns, use_threads=False
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._num_threads) as pool:
            for target, table in pool.map(_read, targets):
                read_count += 1
                if self._cache_size == 0:
                    continue
                with self._io_lock:
                    self._cache[target] = table
                    self._cache.move_to_end(target)
                    while len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
        return read_count

    def get_statistics(
        self, sample_size: Optional[int] = 1000, mode: Optional[str] = 'RGB'
    ) -> Dict[str, List[float]]:
        '''
        Estimates per-channel mean and standard deviation of the dataset.

        The already-decoded parquet rows are reused, so this is cheaper than
        re-reading the shards. Multi-image rows contribute their first image only,
        which keeps the statistics comparable across rows.

        Args:
            sample_size (Optional[int]): Rows to sample. Defaults to 1000; None
                or a value >= ``len(self)`` uses every row.
            mode (Optional[str]): PIL mode used for the statistics pass. Defaults
                to ``'RGB'``; None keeps each image's own mode.

        Returns:
            Dict[str, List[float]]: ``{'mean': [...], 'std': [...]}``.
        '''
        total = len(self)
        if total == 0:
            return {'mean': [0.0, 0.0, 0.0], 'std': [0.0, 0.0, 0.0]}

        if sample_size is None or sample_size >= total:
            indices = list(range(total))
        else:
            indices = torch.randperm(total)[:sample_size].tolist()

        to_tensor = ToTensor()
        means: List[torch.Tensor] = []
        stds: List[torch.Tensor] = []

        for idx in indices:
            try:
                row, _ = self._fetch_row(idx)
                images, _, _ = self.decode_row(row, error_context=f'row {idx}')
                if not images:
                    continue
                # Statistics use the first image of each row; mixing frames of
                # different counts into one mean would not describe anything.
                image = images[0]
                if mode is not None and image.mode != mode:
                    image = image.convert(mode)
                tensor = to_tensor(image)
            except Exception:
                continue
            means.append(torch.mean(tensor, dim=(1, 2)))
            stds.append(torch.std(tensor, dim=(1, 2)))

        if not means:
            return {'mean': [0.0, 0.0, 0.0], 'std': [0.0, 0.0, 0.0]}

        return {
            'mean': torch.mean(torch.stack(means), dim=0).tolist(),
            'std': torch.mean(torch.stack(stds), dim=0).tolist(),
        }

    def summary(self) -> Dict[str, Any]:
        '''
        Reports the resolved layout of the dataset.

        Returns:
            Dict[str, Any]: Shard count, row count, class count, cache state,
            and the decoded column names.
        '''
        return {
            'path': str(self._path),
            'files': [str(p) for p in self._file_paths],
            'rows': len(self),
            'row_groups': len(self._group_meta),
            'columns': list(self._read_columns),
            'image_key': list(self._image_keys),
            'image_key_detected': self._auto_image_key,
            'label_key': self._label_key,
            'path_key': self._path_key,
            'classes': len(self._classes) if self._classes is not None else None,
            'cached_row_groups': len(self._cache),
            'cache_size': self._cache_size,
            'multi_image': self._multi_image,
            'max_images': self._max_images,
        }

    # ------------------------------------------------------------------
    # Stateful protocol & pickling
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        '''
        Snapshots the resumable state of the dataset.

        Only the addressing metadata is stored; row groups are re-read on
        restore so checkpoints stay small.

        Returns:
            Dict[str, Any]: Picklable state dictionary.
        '''
        return {
            'path': self._path,
            'image_key': list(self._image_keys),
            'image_key_detected': self._auto_image_key,
            'label_key': self._label_key,
            'path_key': self._path_key,
            'return_path': self._return_path,
            'index_mode': self._index_mode,
            'mode': self._mode,
            'cache_size': self._cache_size,
            'num_threads': self._num_threads,
            'multi_image': self._multi_image,
            'max_images': self._max_images,
            'classes': self._classes,
            'file_paths': [str(p) for p in self._file_paths],
            'file_rows': list(self._file_rows),
            'group_meta': list(self._group_meta),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        '''
        Restores the dataset from a previously produced state dictionary.

        The stored shard index is installed as-is, which avoids re-reading the
        parquet footers of every shard.

        Args:
            state (Dict[str, Any]): State from :meth:`state_dict`. When the
                stored shards are missing, the dataset is rebuilt from the
                filesystem instead.

        Raises:
            FileNotFoundError: If the shards are gone and the configured path no
                longer resolves.
        '''
        stored_files = [Path(p) for p in state.get('file_paths', [])]
        stored_rows = list(state.get('file_rows', []))
        stored_meta = [tuple(meta) for meta in state.get('group_meta', [])]
        if not stored_files or len(stored_rows) != len(stored_files) or not stored_meta:
            return self._rebuild_index(state.get('classes'))
        if not all(path.exists() for path in stored_files):
            return self._rebuild_index(state.get('classes'))

        self._file_paths = stored_files
        self._file_rows = stored_rows
        self._group_meta = stored_meta
        self._group_starts = [meta[2] for meta in stored_meta]
        self._file_offsets = []
        running = 0
        for rows in self._file_rows:
            self._file_offsets.append(running)
            running += rows
        self._cache.clear()

    def _rebuild_index(self, classes: Optional[Sequence[Any]] = None) -> None:
        '''
        Re-resolves the shards from the filesystem and rebuilds the index.

        Args:
            classes (Optional[Sequence[Any]]): Class list stored in the state, if
                any.
        '''
        self._cache.clear()
        self._file_rows = []
        self._file_row_groups = []
        self._group_meta = []
        self._group_starts = []
        self._resolve_files()
        self._scan_index(classes)

    def __getstate__(self) -> Dict[str, Any]:
        '''
        Produces picklable state, discarding process-local file handles.

        Returns:
            Dict[str, Any]: The instance dictionary without handles, caches, or
            materialized row groups.
        '''
        state = self.__dict__.copy()
        for key in ('_handles', '_thread_handles', '_cache', '_io_lock'):
            state.pop(key, None)
        state['_handle_pid'] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        '''
        Restores instance state and recreates per-process runtime objects.

        Args:
            state (Dict[str, Any]): State produced by :meth:`__getstate__`.
        '''
        self.__dict__.update(state)
        self._handles = None
        self._handle_pid = None
        self._cache = OrderedDict()
        self._thread_handles = {}
        self._io_lock = threading.Lock()

    def close(self) -> None:
        '''
        Closes every parquet handle this instance or its views own.

        Handles are otherwise released when the object is garbage collected;
        calling this is only needed to let a filesystem (Windows) delete or
        replace a shard while the dataset is still referenced. Reading afterwards
        simply reopens the shards, so a closed dataset stays usable.
        '''
        handles = self._handles
        self._handles = None
        self._handle_pid = None
        if handles:
            for handle in list(handles.values()):
                try:
                    handle.close()
                except Exception:
                    pass
            handles.clear()

        thread_handles = getattr(self, '_thread_handles', None)
        if thread_handles:
            for handle in list(thread_handles.values()):
                try:
                    handle.close()
                except Exception:
                    pass
            thread_handles.clear()

        self._cache.clear()

    def __enter__(self) -> 'ParquetImageDataset':
        '''
        Enters the runtime context.

        Returns:
            ParquetImageDataset: self.
        '''
        return self

    def __exit__(self, *exc_info: Any) -> None:
        '''
        Closes the handles owned by this instance on context exit.

        Args:
            *exc_info (Any): Exception details, ignored.
        '''
        self.close()