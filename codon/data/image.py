import torch
import concurrent.futures
import os, io
import pickle
import tarfile

from dataclasses import dataclass

from pathlib import Path
from typing  import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
from torchvision.transforms import Compose, ToTensor

from .base import CodonDataset

@dataclass
class ImageDatasetItem:
    '''
    A data class representing a single item from the ImageDataset or TarImageDataset.

    Attributes:
        image (Any): The loaded and potentially transformed image (e.g., torch.Tensor).
        label (int): The integer label corresponding to the image's class.
        path (Optional[Path]): The original file path or path within the tar.
    '''
    image: Any
    label: int
    path: Optional[Path] = None

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

class ImageDataset(CodonDataset):
    '''
    A dataset class optimized for large-scale image loading with caching and manifest support.

    Attributes:
        _path (Path): The root directory containing image files or subdirectories.
        _transforms (Optional[Compose]): Transformations to apply to the images.
        _extensions (Tuple[str, ...]): Valid image file extensions.
        _loader (Callable): Function to load images from disk.
        _return_path (bool): Whether to include the file path in the returned item.
        _manifest_path (Optional[Path]): Path to a CSV file containing (path, label) pairs.
        _cache_metadata (bool): Whether to cache/load scanned metadata from disk.
        _classes (List[str]): List of class names (subdirectory names).
        _class_to_idx (Dict[str, int]): Mapping from class name to integer label.
        _samples (List[Tuple[Path, int]]): List of (image_path, label) pairs.
    '''

    def __init__(
        self,
        path: Union[str, Path],
        transforms: Optional[Compose] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        loader: Optional[Callable] = None,
        return_path: bool = False,
        manifest_path: Optional[Union[str, Path]] = None,
        cache_metadata: bool = False
    ) -> None:
        '''
        Initializes the ImageDataset.

        Args:
            path (Union[str, Path]): Path to the directory containing images.
            transforms (Optional[Compose]): A composition of torchvision transforms.
            extensions (Optional[Tuple[str, ...]): Valid image file extensions.
                Defaults to ('.jpg', '.jpeg', '.png', '.bmp', '.webp').
            loader (Optional[Callable]): Custom image loader function.
            return_path (bool): If True, returns the file path in ImageDatasetItem.
            manifest_path (Optional[Union[str, Path]]): Path to a pre-generated manifest CSV.
            cache_metadata (bool): If True, caches the scanned file list to disk for faster re-initialization.
        '''
        super().__init__()
        self._path = Path(path)
        self._transforms = transforms
        self._extensions = extensions or ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        self._loader = loader or default_loader
        self._return_path = return_path
        self._manifest_path = Path(manifest_path) if manifest_path else None
        self._cache_metadata = cache_metadata

        # Try to find classes first (may be overridden by manifest or cache)
        self._classes, self._class_to_idx = self._find_classes()
        self._samples = self._build_dataset()

    def _find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        '''
        Finds the class names based on subdirectories.

        Returns:
            Tuple[List[str], Dict[str, int]]: List of classes and mapping to indices.
        '''
        if not self._path.is_dir():
            return [], {}

        classes = [d.name for d in self._path.iterdir() if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _build_dataset(self) -> List[Tuple[Path, int]]:
        '''
        Scans or loads the dataset samples (path, label).

        Prioritizes manifest_path, then cache_file (if enabled), then rglob scan.

        Returns:
            List[Tuple[Path, int]]: A list of (absolute_path, label) tuples.
        '''
        # 1. Load from manifest if provided
        if self._manifest_path and self._manifest_path.exists():
            return self._load_manifest()

        cache_file = self._path / '.codon_image_cache.pkl'

        # 2. Load from cache if enabled and exists
        if self._cache_metadata and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self._classes = cache_data.get('classes', self._classes)
                    self._class_to_idx = cache_data.get('class_to_idx', self._class_to_idx)
                    return cache_data['samples']
            except Exception:
                pass # Fallback to scanning

        # 3. Standard scan
        samples = []
        if not self._path.is_dir():
            return samples

        def _scan_dir(dir_path: Path, label_idx: int) -> None:
            for root, _, files in os.walk(dir_path):
                root_path = Path(root)
                for file_name in files:
                    if file_name.lower().endswith(self._extensions):
                        samples.append(((root_path / file_name).absolute(), label_idx))

        if self._classes:
            for target_class in self._classes:
                class_idx = self._class_to_idx[target_class]
                target_dir = self._path / target_class
                _scan_dir(target_dir, class_idx)
        else:
            _scan_dir(self._path, 0)

        samples.sort(key=lambda x: x[0])

        # 4. Save to cache if enabled
        if self._cache_metadata:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'classes': self._classes,
                        'class_to_idx': self._class_to_idx,
                        'samples': samples
                    }, f)
            except Exception:
                pass

        return samples

    def _load_manifest(self) -> List[Tuple[Path, int]]:
        '''
        Loads samples from a CSV manifest file.

        The CSV should have format: relative_or_absolute_path,label_index

        Returns:
            List[Tuple[Path, int]]: Loaded samples.
        '''
        samples = []
        try:
            with open(self._manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        img_path_str, label_idx = parts[0], int(parts[1])
                        img_path = Path(img_path_str)
                        if not img_path.is_absolute():
                            img_path = self._path / img_path
                        samples.append((img_path, label_idx))
        except Exception as error:
            print(f'Error loading manifest: {error}')
        return samples

    def __len__(self) -> int:
        '''
        Returns the total number of images in the dataset.

        Returns:
            int: Number of image samples.
        '''
        return len(self._samples)

    def __getitem__(self, idx: int) -> ImageDatasetItem:
        '''
        Retrieves the image item at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            ImageDatasetItem: Data class containing image, label, and optionally path.
        '''
        img_path, label = self._samples[idx]

        try:
            image = self._loader(img_path)
        except Exception as error:
            raise RuntimeError(f'Failed to load image at {img_path}: {error}') from error

        if self._transforms is not None:
            image = self._transforms(image)

        return ImageDatasetItem(
            image=image,
            label=label,
            path=img_path if self._return_path else None
        )

    def get_statistics(self, sample_size: Optional[int] = 1000) -> Dict[str, List[float]]:
        '''
        Calculates the mean and standard deviation of the dataset for RGB channels.

        For large datasets, it defaults to sampling 1000 images for speed.

        Args:
            sample_size (Optional[int]): Number of images to sample. Defaults to 1000.

        Returns:
            Dict[str, List[float]]: Dictionary with 'mean' and 'std' keys.
        '''
        loader_transform = Compose([ToTensor()])
        means = []
        stds = []

        n_total = len(self)
        if n_total == 0:
            return {'mean': [0.0, 0.0, 0.0], 'std': [0.0, 0.0, 0.0]}

        if sample_size is None or sample_size >= n_total:
            indices = list(range(n_total))
        else:
            indices = torch.randperm(n_total)[:sample_size].tolist()

        def _process_single_image(idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
            img_path, _ = self._samples[idx]
            try:
                img = self._loader(img_path)
                tensor = loader_transform(img)
                return torch.mean(tensor, dim=(1, 2)), torch.std(tensor, dim=(1, 2))
            except Exception:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            results = executor.map(_process_single_image, indices)

        for res in results:
            if res is not None:
                means.append(res[0])
                stds.append(res[1])

        if not means:
            return {'mean': [0.0, 0.0, 0.0], 'std': [0.0, 0.0, 0.0]}

        final_mean = torch.mean(torch.stack(means), dim=0).tolist()
        final_std = torch.mean(torch.stack(stds), dim=0).tolist()

        return {'mean': final_mean, 'std': final_std}

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
