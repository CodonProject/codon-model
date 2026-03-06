import torch
import numpy as np
import random
import os

from typing import Any, Optional

def seed_everything(seed: int = 42, strict: bool = False, warn_only: bool = True) -> None:
    '''
    Sets all random seeds to ensure reproducibility of PyTorch experiments.

    Args:
        seed (int): The random seed value. Defaults to 42.
        strict (bool): Whether to enable strict deterministic mode.
            If True, calls torch.use_deterministic_algorithms(True),
            which may raise errors for certain operations that do not support
            deterministic algorithms and may reduce training speed.
        warn_only (bool): If True, only warns when deterministic algorithms
            are not available. Defaults to True.
    '''
    import codon
    codon.__seed__ = seed
    random.seed(seed)
    np.random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False 
        torch.backends.cudnn.deterministic = True 
    if strict:
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            print(f'[Info] Strict deterministic mode enabled. (seed={seed})')
        except AttributeError:
            print('[Warning] torch.use_deterministic_algorithms is not available in your PyTorch version.')
    else:
        print(f'[Info] Random seed set as {seed}')

def get_seed() -> Optional[int]:
    '''
    Retrieves the global random seed.

    Returns:
        Optional[int]: The current seed value, or None if not set.
    '''
    import codon
    return codon.__seed__

def worker_init_fn(worker_id: Any) -> None:
    '''
    Worker initialization function for DataLoader to ensure each worker has a unique seed.

    Args:
        worker_id (Any): The worker ID.
    '''
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_generator() -> torch.Generator:
    '''
    Creates a random number generator with the global seed.

    Returns:
        torch.Generator: The configured random number generator.
    '''
    import codon
    seed = codon.__seed__ if hasattr(codon, '__seed__') and codon.__seed__ is not None else 42
    return torch.Generator().manual_seed(seed)
