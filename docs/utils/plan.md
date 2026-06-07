# Training Plan Documentation

## Overview

Utilities for planning and managing context-length training schedules.

## Functions

### calculate_training_steps()

Calculate training steps based on model size.

```python
def calculate_training_steps(
    model_size: int,
    tokens_per_sample: int,
    batch_size: int,
    min_tpp: float = 20.0,
    rec_tpp: float = 80.0,
    overfit_tpp: float = 200.0
) -> TrainingStepsConfig
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_size | int | - | Number of model parameters |
| tokens_per_sample | int | - | Tokens per sample |
| batch_size | int | - | Batch size |
| min_tpp | float | 20.0 | Minimum tokens per parameter |
| rec_tpp | float | 80.0 | Recommended tokens per parameter |
| overfit_tpp | float | 200.0 | Overfit tokens per parameter |

**Returns:** `TrainingStepsConfig` with min/recommended/overfit steps.

---

### calculate_training_schedule()

Calculate training schedule for context expansion.

```python
def calculate_training_schedule(
    params: ContextTrainingParams,
    target_len: int
) -> Dict[str, Any]
```

---

## Classes

### ContextTrainingPlanner

Planner for context-length training.

#### Constructor

```python
ContextTrainingPlanner(
    model,
    step_mode: str = 'recommended',
    base_context: int = 512,
    target_context: int = 8192,
    global_batch_tokens: int = 8192 * 2
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | - | - | Model to train |
| step_mode | str | 'recommended' | 'min', 'recommended', or 'overfit' |
| base_context | int | 512 | Starting context length |
| target_context | int | 8192 | Target context length |
| global_batch_tokens | int | 16384 | Global batch tokens |

#### generate_plan()

Generate the training plan.

```python
def generate_plan() -> TrainingPlan
```

#### Example Usage

```python
from codon.motif import MotifA1
from codon.utils.plan import ContextTrainingPlanner

model = MotifA1()

planner = ContextTrainingPlanner(
    model=model,
    step_mode='recommended',
    base_context=512,
    target_context=8192
)

plan = planner.generate_plan()
plan.print_report()
```

**Output:**
```
======================================================================
LLM Context Training Plan | Strategy: [RECOMMENDED]
======================================================================
Total Budget: 8.000 B Tokens | Total Steps: 12,345
----------------------------------------------------------------------
➜ [1] Foundation        | Seq_512   | BS=32 | Tokens: 3.200B | Steps: 4,000
➜ [2] Expansion_1024    | Seq_1024  | BS=16 | Tokens: 1.600B | Steps: 2,000
➜ [3] Expansion_2048    | Seq_2048  | BS=8  | Tokens: 1.600B | Steps: 2,000
➜ [4] Stabilization     | Seq_8192  | BS=2  | Tokens: 1.600B | Steps: 4,345
======================================================================
```

---

### TrainingPlan

Training plan data class.

```python
@dataclass
class TrainingPlan:
    total_tokens: int
    total_steps: int
    stages: List[Stage]
    step_mode: str
    
    def print_report(self) -> TrainingPlan
```

---

### Stage

Single training stage.

```python
@dataclass
class Stage:
    name: str
    seq_len: int
    chunk_len: int
    batch_size: int
    tokens: int
    steps: int
    
    def build_stream(self, data, eos_token_id) -> ChunkedTokenStream
```

---

## Notes

1. **Chinchilla Scaling**: Uses tokens-per-parameter heuristics.
2. **Context Expansion**: Gradually increases context length.
3. **Stage Types**: Foundation → Expansion → Stabilization.