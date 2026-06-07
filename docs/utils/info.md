# System Information Documentation

## Overview

Utilities for collecting and storing system environment information for experiment reproducibility.

## Data Classes

### SystemEnvironment

Comprehensive snapshot of the experimental environment.

```python
@dataclass(frozen=True)
class SystemEnvironment:
    timestamp: str
    hostname: str
    os_info: Dict[str, str]
    cpu_info: Dict[str, Any]
    ram_total_gb: float
    gpu_info: Dict[str, Any]
    libraries: Dict[str, str]
    seeds: Dict[str, Any]
    
    def to_json(self, indent: int = 4) -> str: ...
    def save(self, path: str): ...
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| timestamp | str | Experiment start time |
| hostname | str | Machine hostname |
| os_info | Dict | OS version information |
| cpu_info | Dict | CPU model and cores |
| ram_total_gb | float | Total RAM in GB |
| gpu_info | Dict | GPU details and CUDA versions |
| libraries | Dict | Versions of key libraries |
| seeds | Dict | Random seed state |

---

## Functions

### get_system_info()

Collects current system environment information.

```python
def get_system_info(manual_seed: Optional[int] = None) -> SystemEnvironment
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| manual_seed | int | None | Manually provided seed for recording |

**Returns:** `SystemEnvironment` object with environment details.

**Example:**

```python
from codon.utils.info import get_system_info

env = get_system_info()
print(f"Hostname: {env.hostname}")
print(f"CPU: {env.cpu_info['model']}")
print(f"GPU Count: {env.gpu_info['count']}")
print(f"PyTorch Version: {env.libraries['torch']}")
```

### Saving Environment Information

```python
# Save to JSON file
env = get_system_info()
env.save('experiment_environment.json')

# Or get JSON string
json_str = env.to_json(indent=4)
print(json_str)
```

---

## Usage Patterns

### Experiment Logging

```python
import logging
from codon.utils.info import get_system_info
from codon.utils.seed import seed_everything

# Set seed
seed_everything(42)

# Log environment
env = get_system_info()
logging.info(f"Experiment started at: {env.timestamp}")
logging.info(f"Running on: {env.hostname}")
logging.info(f"GPU: {env.gpu_info['devices']}")

# Save for reproducibility
env.save('env_info.json')
```

### Checking Library Versions

```python
env = get_system_info()

# Check if required libraries are installed
required_libs = ['torch', 'transformers', 'accelerate']
for lib in required_libs:
    version = env.libraries.get(lib, 'not_installed')
    print(f"{lib}: {version}")
```

---

## Notes

1. **Frozen Dataclass**: `SystemEnvironment` is frozen (immutable) to prevent accidental modification.
2. **GPU Detection**: Automatically detects all CUDA devices and their properties.
3. **Seed Tracking**: Records the manual seed if provided, along with torch/cuda initial seeds.
4. **Library Versions**: Checks for common ML libraries: numpy, torch, transformers, accelerate, etc.