# Developer CLI Documentation

## Overview

Command-line utilities for development workflows.

## Commands

### codon hash

Calculate MD5 hash of files or directories.

```bash
codon hash <file_or_directory>
```

**Example:**

```bash
# Hash single file
codon hash model.safetensors

# Hash all files in directory
codon hash ./checkpoints/
```

**Output format:**
```
a1b2c3d4e5f6...  path/to/file
```

---

### codon clear

Recursively clean `__pycache__` directories.

```bash
codon clear [target_path]
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| target_path | . | Directory to clean |

**Example:**

```bash
# Clean current directory
codon clear

# Clean specific directory
codon clear ./src/
```

---

## Python API

### hash_target()

Calculate MD5 hashes for a file or directory.

```python
from codon.dev.hash import hash_target

results = hash_target('./model_weights/')
for path, md5 in results.items():
    print(f'{md5}  {path}')
```

### clear_pycache()

Clean `__pycache__` directories.

```python
from codon.dev.clear import clear_pycache

clear_pycache('./src/')
```

---

## Notes

1. **Hash Progress**: Shows progress bar for large files.
2. **Clear Safety**: Only removes `__pycache__` directories, not other files.