# BasicModel Documentation

## Overview

The `BasicModel` class is the base class for all models in Codon, providing common functionality like gradient checkpointing, parameter counting, model loading/saving, and remote resource support.

## Class Definition

```python
class BasicModel(nn.Module, RemoteResourceMixin):
    '''
    Base class for all models, providing common functionality like gradient checkpointing and parameter counting.
    '''
```

## Properties

### device

Get the device of the model.

```python
@property
def device(self) -> torch.device:
```

**Returns:** The device where model parameters are located, or 'cpu' if no parameters exist.

#### Example Usage

```python
model = BasicModel()
print(f"Model device: {model.device}")  # 'cpu'

model = model.to('cuda')
print(f"Model device: {model.device}")  # 'cuda:0'
```

### trainable_params

Get an iterator over trainable parameters.

```python
@property
def trainable_params(self) -> Iterator[torch.nn.Parameter]:
```

**Returns:** Iterator over parameters with `requires_grad=True`.

---

## Methods

### set_checkpoint()

Enable or disable gradient checkpointing for the model and its sub-modules.

```python
def set_checkpoint(self, value: bool) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| value | bool | True to enable gradient checkpointing |

**Example Usage**

```python
model = MyModel()
model.set_checkpoint(True)  # Enable gradient checkpointing
model.train()
```

### checkpoint()

Apply gradient checkpointing to a function if enabled and in training mode.

```python
def checkpoint(self, function: Callable, *args, **kwargs) -> Any
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| function | Callable | Function to be checkpointed |
| *args | - | Positional arguments |
| **kwargs | - | Keyword arguments |

**Returns:** Output of the function.

**Example Usage**

```python
def forward(self, x):
    return self.checkpoint(self._compute, x)

def _compute(self, x):
    # Expensive computation
    return x @ self.weight
```

### get_params()

Get an iterator over model parameters.

```python
def get_params(self, trainable_only: bool = False) -> Iterator[torch.nn.Parameter]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| trainable_only | bool | False | If True, only yield parameters requiring gradients |

**Example Usage**

```python
# Get all parameters
all_params = list(model.get_params())

# Get only trainable parameters
trainable_params = list(model.get_params(trainable_only=True))
```

### count_params()

Count the number of parameters in the model.

```python
def count_params(
    self,
    trainable_only: bool = False,
    active_only: bool = False,
    human_readable: bool = False,
    seen: set = None
) -> Union[int, str]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| trainable_only | bool | False | Count only trainable parameters |
| active_only | bool | False | Count only active parameters (for MoE) |
| human_readable | bool | False | Return string with units (B, M, K) |
| seen | set | None | Set of already counted parameters |

**Returns:** Total parameter count (int or human-readable string).

**Example Usage**

```python
model = MyModel()

# Count all parameters
total = model.count_params()
print(f"Total params: {total}")

# Count only trainable parameters
trainable = model.count_params(trainable_only=True)
print(f"Trainable params: {trainable}")

# Human-readable format
print(f"Total params: {model.count_params(human_readable=True)}")  # '125.5M'
```

### load_pretrained()

Load a pretrained model from a file.

```python
def load_pretrained(self, path: str, strict: bool = False) -> TBasicModel
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| path | str | - | Path to model file (.safetensors or .pth) |
| strict | bool | False | Whether to strictly enforce key matching |

**Returns:** The model itself for method chaining.

**Supported Formats:**
- `.safetensors` (recommended)
- `.pth` / `.pt` (PyTorch format)

**Example Usage**

```python
model = MyModel()

# Load from safetensors
model.load_pretrained('model_weights.safetensors')

# Load from PyTorch checkpoint
model.load_pretrained('model_weights.pth')

# Chain loading with model creation
model = MyModel().load_pretrained('weights.safetensors')
```

### save_pretrained()

Save the model to a file.

```python
def save_pretrained(
    self,
    path: str,
    trainable_only: bool = False,
    include_buffer: bool = True,
    exclude_modules: list[Union[type, nn.Module]] = None,
    only: list[str] = None,
    exclude: list[str] = None
) -> TBasicModel
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| path | str | - | Path to save file |
| trainable_only | bool | False | Save only trainable parameters |
| include_buffer | bool | True | Include registered buffers |
| exclude_modules | list | None | Module types or instances to exclude |
| only | list | None | Only save keys containing these strings |
| exclude | list | None | Exclude keys containing these strings |

**Returns:** The model itself for method chaining.

**Example Usage**

```python
model = MyModel()

# Save entire model
model.save_pretrained('model.safetensors')

# Save only trainable parameters
model.save_pretrained('trainable_only.safetensors', trainable_only=True)

# Exclude specific modules
model.save_pretrained('model.safetensors', exclude_modules=[nn.LayerNorm])

# Save only LoRA parameters
model.save_pretrained('lora_only.safetensors', only=['lora'])

# Exclude optimizer states
model.save_pretrained('model.safetensors', exclude=['optimizer'])
```

### freeze()

Freeze all parameters by setting `requires_grad=False`.

```python
def freeze(self) -> TBasicModel
```

**Returns:** The model itself for method chaining.

**Example Usage**

```python
model = MyModel()
model.freeze()  # Freeze all parameters
```

### unfreeze()

Unfreeze all parameters by setting `requires_grad=True`.

```python
def unfreeze(self) -> TBasicModel
```

**Returns:** The model itself for method chaining.

**Example Usage**

```python
model = MyModel()
model.freeze()
# ... do something ...
model.unfreeze()  # Unfreeze all parameters
```

### compile()

Compile the model using PyTorch 2.0 compiler.

```python
def compile(self) -> TBasicModel
```

**Returns:** The compiled model.

**Example Usage**

```python
model = MyModel()
model = model.compile()  # Compile for faster inference
```

---

## RemoteResourceMixin

The `BasicModel` inherits from `RemoteResourceMixin`, providing methods to download and load pre-trained models from remote repositories.

### from_remote()

Load model weights from remote sources (ModelScope or Hugging Face).

```python
def from_remote(
    self,
    platform: Optional[Literal['modelscope', 'huggingface']] = None,
    url: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> TRemoteResource
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| platform | str | None | Platform to download from ('modelscope' or 'huggingface') |
| url | str | None | Direct URL to download |
| cache_dir | str | None | Local cache directory |

**Returns:** The model itself with loaded weights.

**Example Usage**

```python
from codon.motif import MotifA1

# Auto-detect optimal platform
model = MotifA1().from_remote()

# Specify platform
model = MotifA1().from_remote(platform='huggingface')

# Download from custom URL
model = MotifA1().from_remote(url='https://example.com/model.safetensors')
```

### Configuration

Models must define class attributes for remote resources:

```python
class MyModel(BasicModel):
    __modelscope__ = {
        'repo': 'username/model-name',
        'files': ['model.safetensors'],
        'branch': 'master'
    }
    
    __huggingface__ = {
        'repo': 'username/model-name',
        'files': ['model.safetensors'],
        'branch': 'main'
    }
```

---

## Usage Patterns

### Full Training Workflow

```python
from codon.base import BasicModel

class MyModel(BasicModel):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.layer(x)

# Create and configure
model = MyModel()
model.set_checkpoint(True)  # Enable gradient checkpointing

# Load pretrained
model.load_pretrained('pretrained.safetensors')

# Freeze base, unfreeze head
model.freeze()
model.head.unfreeze()

# Train...

# Save final model
model.save_pretrained('final_model.safetensors')

# Count parameters
print(f"Params: {model.count_params(human_readable=True)}")
```

### Model Compilation for Inference

```python
model = MyModel().load_pretrained('model.safetensors')
model = model.compile()  # Optimize for speed
model.eval()

# Fast inference
with torch.no_grad():
    output = model(input_tensor)
```

### Remote Loading with Auto-Platform Selection

```python
# Auto-select best platform based on network conditions
model = MotifA1().from_remote()

# Use cached weights if available
model = MotifA1().from_remote(cache_dir='./cache')
```

---

## Notes

1. **Gradient Checkpointing**: Reduces memory usage during training but increases computation time.
2. **Parameter Counting**: Handles shared parameters correctly using a `seen` set to avoid double-counting.
3. **Safetensors**: Recommended for safer model loading (no arbitrary code execution).
4. **Remote Caching**: Downloads are cached locally to avoid repeated downloads.
5. **Platform Fallback**: If download fails from one platform, automatically tries the other.