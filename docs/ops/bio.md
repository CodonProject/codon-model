# Bio-inspired Operations Documentation

## Overview

Collection of bio-inspired learning rules and synaptic plasticity operations.

## Functions

### hebbian_update()

Classic Hebbian learning rule: `dw = learning_rate * pre * post`

```python
def hebbian_update(
    weight: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    learning_rate: float = 0.01
) -> torch.Tensor
```

**Example:**

```python
import torch
from codon.ops import hebbian_update

weight = torch.randn(10, 20)
pre = torch.randn(10)
post = torch.randn(20)

updated_weight = hebbian_update(weight, pre, post, learning_rate=0.01)
```

---

### anti_hebbian_update()

Anti-Hebbian learning rule: `dw = -learning_rate * pre * post`

```python
def anti_hebbian_update(
    weight: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    learning_rate: float = 0.01
) -> torch.Tensor
```

---

### oja_update()

Oja's rule for principal component analysis: `dw = learning_rate * (pre * post - weight * post^2)`

```python
def oja_update(
    weight: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    learning_rate: float = 0.01
) -> torch.Tensor
```

---

### bcm_update()

Bienenstock-Cooper-Munro rule with sliding threshold:

```python
def bcm_update(
    weight: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    theta: torch.Tensor,
    learning_rate: float = 0.01
) -> torch.Tensor
```

**Parameters:**
- `theta`: Sliding threshold for plasticity

---

### covariance_update()

Covariance-based learning:

```python
def covariance_update(
    weight: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    learning_rate: float = 0.01,
    decay: float = 0.99
) -> torch.Tensor
```

---

### instar_update()

Instar learning rule for pattern recognition:

```python
def instar_update(
    weight: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    learning_rate: float = 0.01
) -> torch.Tensor
```

---

### eligibility_trace_update()

Temporal difference learning with eligibility traces:

```python
def eligibility_trace_update(
    weight: torch.Tensor,
    eligibility_trace: torch.Tensor,
    error: torch.Tensor,
    learning_rate: float = 0.01,
    trace_decay: float = 0.9
) -> torch.Tensor
```

---

### reward_modulated_hebbian_update()

Reward-modulated Hebbian learning:

```python
def reward_modulated_hebbian_update(
    weight: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    reward: torch.Tensor,
    learning_rate: float = 0.01
) -> torch.Tensor
```

---

### rate_based_stdp_update()

Rate-based spike-timing-dependent plasticity:

```python
def rate_based_stdp_update(
    weight: torch.Tensor,
    pre_rate: torch.Tensor,
    post_rate: torch.Tensor,
    learning_rate: float = 0.01,
    tau_plus: float = 20.0,
    tau_minus: float = 20.0,
    A_plus: float = 0.01,
    A_minus: float = 0.012
) -> torch.Tensor
```

---

### vogels_sprekeler_update()

Vogels-Sprekeler balanced network learning rule:

```python
def vogels_sprekeler_update(
    weight: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    target_rate: float = 40.0,
    learning_rate: float = 0.001
) -> torch.Tensor
```

---

### local_error_driven_update()

Local error-driven learning rule:

```python
def local_error_driven_update(
    weight: torch.Tensor,
    pre: torch.Tensor,
    error: torch.Tensor,
    learning_rate: float = 0.01
) -> torch.Tensor
```

---

### synaptic_scaling_update()

Synaptic scaling for maintaining firing rate homeostasis:

```python
def synaptic_scaling_update(
    weight: torch.Tensor,
    avg_activity: torch.Tensor,
    target_activity: float = 1.0,
    scaling_factor: float = 0.1
) -> torch.Tensor
```

---

## Usage Example

### Building a Spiking Neural Network

```python
import torch
from codon.ops import hebbian_update, synaptic_scaling_update

# Initialize weights
weights = torch.randn(100, 100)
pre_activity = torch.randn(100)
post_activity = torch.randn(100)

# Apply Hebbian learning
weights = hebbian_update(weights, pre_activity, post_activity)

# Apply synaptic scaling
avg_activity = post_activity.mean()
weights = synaptic_scaling_update(weights, avg_activity)
```

### Reinforcement Learning with Eligibility Traces

```python
import torch
from codon.ops import eligibility_trace_update

weights = torch.randn(50, 50)
eligibility_trace = torch.zeros(50, 50)
error = torch.tensor(0.5)  # Reward prediction error

# Update eligibility trace
eligibility_trace = 0.9 * eligibility_trace + pre_activity * post_activity

# Update weights
weights = eligibility_trace_update(weights, eligibility_trace, error)
```

---

## Notes

1. **Batch Processing**: All functions support batch operations where applicable.
2. **Autograd**: Functions are compatible with PyTorch autograd.
3. **Stability**: Learning rates may need adjustment based on application.
4. **Biological Plausibility**: These rules are inspired by neuroscientific findings.