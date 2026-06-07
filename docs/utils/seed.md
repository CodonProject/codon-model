# Seed Management Documentation

## Overview

Utilities for managing random seeds to ensure reproducibility in experiments.

## Functions

### seed_everything()

Sets all random seeds to ensure reproducibility.

```python
def seed_everything(
    seed: int = 42,
    strict: bool = False,
    warn_only: bool = True,
    verbose: bool = True
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| seed | int | 42 | Random seed value |
| strict | bool | False | Enable strict deterministic mode |
| warn_only | bool | True | Only warn when deterministic algorithms unavailable |
| verbose | bool | True | Print seed information |

**Example:**

```python
from codon.utils.seed import seed_everything

seed_everything(seed=42, strict=False)
# Sets seed for: random, numpy, torch, CUDA, CUBLAS
```

### get_seed()

Retrieves the global random seed.

```python
def get_seed() -> Optional[int]
```

**Returns:** Current seed value or None if not set.

### worker_init_fn()

Worker initialization function for DataLoader.

```python
def worker_init_fn(worker_id: Any) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| worker_id | Any | Worker ID from DataLoader |

**Example:**

```python
from torch.utils.data import DataLoader
from codon.utils.seed import seed_everything, worker_init_fn

seed_everything(42)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    worker_init_fn=worker_init_fn  # Ensures each worker has unique seed
)
```

### create_generator()

Creates a random number generator with the global seed.

```python
def create_generator() -> torch.Generator
```

**Returns:** Configured PyTorch Generator.

**Example:**

```python
from codon.utils.seed import create_generator

generator = create_generator()
# Use generator for reproducible random operations
indices = torch.randperm(100, generator=generator)
```

---

## Usage Patterns

### Reproducible Training

```python
import torch
from codon.utils.seed import seed_everything, create_generator

# Set seed
seed_everything(42, strict=True)

# Create generator for data sampling
generator = create_generator()

# Training loop with reproducibility
for epoch in range(10):
    # Shuffle dataset with fixed seed
    dataset.shuffle(generator=generator)
    
    # Training code...
```

### Strict Deterministic Mode

```python
# Enables strict deterministic mode (slower but reproducible)
seed_everything(42, strict=True)

# This will enable torch.use_deterministic_algorithms(True)
# and set CUBLAS_WORKSPACE_CONFIG environment variable
```

---

## Notes

1. **CUDA Determinism**: Sets `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.
2. **Strict Mode**: Uses `torch.use_deterministic_algorithms()` which may reduce performance.
3. **Worker Seeds**: Each DataLoader worker gets a unique seed based on the global seed.
4. **Global State**: The seed is stored in `codon.__seed__` for later retrieval.