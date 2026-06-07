# Fourier Operations Documentation

## Overview

Frequency-domain operations for efficient sequence mixing, based on 'Caracal: Causal Architecture via Spectral Mixing' [arXiv:2605.00292 cs.LG].

## Functions

### apply_fourier_mixing()

Perform causal mixing in the frequency domain via FFT.

```python
def apply_fourier_mixing(
    x_v: torch.Tensor,
    x_g: torch.Tensor,
    seq_len: int
) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| x_v | torch.Tensor | Content stream tensor [batch_size, n_heads, seq_len, d_head] |
| x_g | torch.Tensor | Gate stream tensor [batch_size, n_heads, seq_len, d_head] |
| seq_len | int | Original sequence length L |

**Returns:** Mixed sequence tensor truncated to original length.

**Algorithm:**

1. **Padding**: Pad sequences to length 2*L for causal FFT
2. **FFT**: Apply real FFT to both content and gate streams
3. **Frequency Mixing**: Multiply V_fft * G_fft in frequency domain
4. **Inverse FFT**: Apply inverse real FFT
5. **Truncation**: Truncate result to original sequence length

**Complexity**: O(L log L) compared to O(L²) for standard attention.

#### Example Usage

```python
import torch
from codon.ops.fourier import apply_fourier_mixing

# Create content and gate streams
x_v = torch.randn(2, 12, 64, 64)  # [batch, heads, seq_len, d_head]
x_g = torch.randn(2, 12, 64, 64)

# Apply Fourier mixing
result = apply_fourier_mixing(x_v, x_g, seq_len=64)
print(f"Result shape: {result.shape}")  # [2, 12, 64, 64]
```

---

## Usage Notes

### Causality

The padding to 2*L ensures causality by avoiding circular convolution artifacts. The final truncation to L ensures only valid causal outputs are used.

### Numerical Precision

The function converts inputs to float32 for FFT operations and converts back to the original dtype for output.

### Performance

Fourier mixing is particularly efficient for long sequences:
- Standard attention: O(L²) complexity
- Fourier mixing: O(L log L) complexity

### Memory Efficiency

Requires storing FFT results which is O(L) per head, similar to attention but with different constant factors.

---

## Example: Replacing Attention with Fourier Mixing

```python
import torch
import torch.nn as nn
from codon.ops.fourier import apply_fourier_mixing

class FourierAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.d_model = hidden_size
        self.n_heads = num_heads
        self.d_head = hidden_size // num_heads
        
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.W_G = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Project to content and gate streams
        x_v = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        x_g = self.W_G(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Apply Fourier mixing
        x_mixed = apply_fourier_mixing(x_v, x_g, seq_len)
        
        # Reshape and project
        x_mixed = x_mixed.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(x_mixed)
        
        return output

# Usage
model = FourierAttention(hidden_size=768, num_heads=12)
x = torch.randn(2, 1024, 768)
output = model(x)
print(f"Output shape: {output.shape}")  # [2, 1024, 768]
```

---

## Notes

1. **Causality**: The algorithm ensures causal outputs by padding to 2*L and truncating.
2. **Numerical Stability**: Uses float32 for FFT operations regardless of input dtype.
3. **Sequence Length**: Works best with power-of-two sequence lengths for optimal FFT performance.
4. **Head Dimensions**: The d_head dimension should be compatible with FFT operations (typically divisible by small primes).