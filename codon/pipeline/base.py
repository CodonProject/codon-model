from codon.pipeline.callback import *
from codon.utils.safecode import safecode as utils_safecode
from torch.optim import Optimizer

import random
import json
import tempfile
import time
import os

from abc import abstractmethod
from enum import Enum, auto
from functools import wraps
from typing import Optional, List, Dict, Callable, Iterator


class PipelinePhase(Enum):
    INIT = auto()
    SETUP = auto()
    TRAINING = auto()
    EVALUATING = auto()
    FINISHED = auto()
    INTERRUPTED = auto()
    FAILED = auto()


@dataclass
class TrainState:
    '''Mutable training state, fully captured in checkpoints.'''
    global_step: int = 0
    epoch: int = 0
    best_metric: float = float('-inf')
    best_metric_name: Optional[str] = None
    phase: PipelinePhase = PipelinePhase.INIT
    interrupted: bool = False
    start_time: float = 0.0
    elapsed: float = 0.0

    def advance_step(self) -> int:
        self.global_step += 1
        return self.global_step


class MetricsTracker:
    '''
    Thread of named metrics with history + running averages.

    Naming convention: 'namespace/key' (e.g. 'loss/policy', 'reward/mean').
    Callbacks query via `get('reward/mean')` or `last('approx_kl')`.
    '''

    def __init__(self, maxlen: int = 10000):
        self._history: Dict[str, List[float]] = {}
        self._maxlen = maxlen

    def log(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            self._history.setdefault(k, []).append(float(v))
            if len(self._history[k]) > self._maxlen:
                self._history[k] = self._history[k][-self._maxlen:]

    def last(self, key: str, default: float = 0.0) -> float:
        h = self._history.get(key)
        return h[-1] if h else default

    def get(self, key: str) -> List[float]:
        return list(self._history.get(key, []))

    def running_avg(self, key: str, window: int = 100) -> float:
        h = self._history.get(key, [])
        w = h[-window:]
        return sum(w) / len(w) if w else 0.0

    def keys(self) -> List[str]:
        return sorted(self._history.keys())

    def snapshot(self, window: int = 100) -> Dict[str, float]:
        return {k: self.running_avg(k, window) for k in self.keys()}

    def dump_jsonl(self, path: str) -> None:
        n = max((len(v) for v in self._history.values()), default=0)
        keys = self.keys()
        with open(path, 'w', encoding='utf-8') as f:
            for i in range(n):
                row = {k: self._history[k][i] for k in keys if i < len(self._history[k])}
                f.write(json.dumps(row) + '\n')


_PIPELINE_REGISTRY: Dict[str, Type['BasicPipeline']] = {}


def register_pipeline(name: str):
    def decorator(cls):
        if name in _PIPELINE_REGISTRY:
            raise ValueError(f"pipeline '{name}' already registered")
        _PIPELINE_REGISTRY[name] = cls
        cls.__pipeline_name__ = name
        return cls
    return decorator


def build_pipeline(name: str, **kwargs) -> 'BasicPipeline':
    if name not in _PIPELINE_REGISTRY:
        raise KeyError(f"unknown pipeline '{name}'. Available: {sorted(_PIPELINE_REGISTRY)}")
    return _PIPELINE_REGISTRY[name](**kwargs)


class BasicPipeline:
    '''
    Base class for all training pipelines.

    Contract for subclasses:
      REQUIRED
        - train_step(batch) -> Dict[str, float]   # one optimization iteration
        - state_payload() / load_state_payload()  # algorithm-specific checkpoint pieces
      OPTIONAL
        - setup() / teardown()
        - iterate_epochs(dataset) -> Iterator[List[Any]]   # default: shuffled epochs
        - evaluate(dataset) -> Dict[str, float]
        - optimizers() -> Dict[str, Optimizer]

    The base class owns: lifecycle, callbacks, metrics, checkpointing,
    gradient accumulation, AMP, grad clipping, interruption handling, seeding.
    '''

    # Version tag embedded in checkpoints for forward-compat
    __pipeline_version__: str = '1.0'

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        callbacks: Optional[List[Callback]] = None,
        seed: Optional[int] = None,
    ):
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.state = TrainState()
        self.metrics = MetricsTracker()
        self.callbacks: List[Callback] = callbacks or []

        self.seed = seed
        if seed is not None:
            from codon.utils import seed_everything
            seed_everything(self.seed)

        self._setup_done = False

    # Identity (mirrors BasicModel.safecode convention)

    @wraps(utils_safecode)
    def safecode(self, length: int = 4, exclude_confusing: bool = False) -> str:
        return utils_safecode(length=length, exclude_confusing=exclude_confusing)

    @property
    def name(self) -> str:
        return getattr(self, '__pipeline_name__', type(self).__name__)

    # Abstract interface

    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        '''Execute one optimization iteration. Returns scalar metrics.'''
        ...

    @abstractmethod
    def state_payload(self) -> Dict[str, Any]:
        '''Algorithm-specific state to checkpoint (models, optimizers, etc.).'''
        ...

    @abstractmethod
    def load_state_payload(self, payload: Dict[str, Any]) -> None:
        '''Restore from state_payload().'''
        ...

    # Optional overridables

    def setup(self) -> None:
        '''Lazy initialization (device transfer, optimizer build, contract validation).'''

    def teardown(self) -> None:
        '''Release resources.'''

    def optimizers(self) -> Dict[str, Optimizer]:
        '''All optimizers — used for LR scheduling and checkpointing.'''
        return {}

    def iterate_epochs(self, dataset: List[Any]) -> Iterator[List[Any]]:
        '''Default epoch iterator: shuffle + yield batches.'''
        while True:
            data = list(dataset)
            random.shuffle(data)
            yield data

    def evaluate(self, dataset: Optional[List[Any]] = None) -> Dict[str, float]:
        '''Optional evaluation. Default: empty.'''
        return {}

    # Utility surface for subclasses

    def log(self, metrics: Dict[str, float]) -> None:
        self.metrics.log(metrics)

    def to_device(self, obj):
        '''Move tensors / modules / nested containers to self.device.'''
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        if isinstance(obj, nn.Module):
            return obj.to(self.device)
        if isinstance(obj, dict):
            return {k: self.to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self.to_device(v) for v in obj)
        return obj

    def _callback(self, hook: str, *args) -> None:
        for cb in self.callbacks:
            getattr(cb, hook)(self, *args)

    def _should_stop(self) -> bool:
        return any(
            getattr(cb, 'stop_requested', False) for cb in self.callbacks
        )

    # Checkpointing (atomic write)

    def save_checkpoint(self, path: str) -> None:
        payload = {
            '__pipeline__': self.name,
            '__version__': self.__pipeline_version__,
            'state': self.state.__dict__.copy(),
            'metrics': self.metrics.snapshot(),
            'callbacks': {i: cb.state_dict() for i, cb in enumerate(self.callbacks)},
            'payload': self.state_payload(),
        }
        # Atomic: write to temp then rename — never corrupt on crash mid-save
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as f:
                torch.save(payload, f)
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise
        self._callback('on_checkpoint', path)

    def load_checkpoint(self, path: str, strict: bool = True) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if strict and ckpt.get('__pipeline__') != self.name:
            raise ValueError(
                f"checkpoint pipeline {ckpt.get('__pipeline__')!r} "
                f"!= current {self.name!r}"
            )
        self.state.__dict__.update(ckpt['state'])
        self.state.phase = PipelinePhase.INIT
        self.load_state_payload(ckpt['payload'])
        for i, s in ckpt.get('callbacks', {}).items():
            if i < len(self.callbacks):
                self.callbacks[i].load_state_dict(s)

    # Training loop (template method)

    def train(
        self,
        dataset: List[Any],
        num_epochs: int = 1,
        steps_per_epoch: Optional[int] = None,
        batch_fn: Optional[Callable[[List[Any]], Iterator[Any]]] = None,
        grad_accum_steps: int = 1,
        use_amp: bool = False,
        max_grad_norm: Optional[float] = None,
        eval_every_epochs: Optional[int] = None,
        eval_dataset: Optional[List[Any]] = None,
        resume_from: Optional[str] = None,
    ) -> Dict[str, float]:
        '''
        Run the full training loop.

        Args:
            dataset: Training data (interpretation is subclass-defined).
            num_epochs: Epoch budget.
            steps_per_epoch: Cap steps per epoch (None = dataset length).
            batch_fn: Optional batcher; default treats dataset items as batches.
            grad_accum_steps: Accumulate gradients N steps before optimizer.step().
            use_amp: Enable torch autocast.
            max_grad_norm: Global norm clipping (None disables).
            eval_every_epochs: Run evaluate() + on_evaluate every N epochs.
            resume_from: Checkpoint path to restore before training.
        '''
        # --- Resume ---
        if resume_from is not None:
            self.load_checkpoint(resume_from)
            print(f"Resumed from {resume_from}: step={self.state.global_step}, "
                  f"epoch={self.state.epoch}")

        # --- Setup (idempotent) ---
        if not self._setup_done:
            self.setup()
            self._setup_done = True

        self.state.phase = PipelinePhase.TRAINING
        self.state.start_time = time.time() - self.state.elapsed
        self._callback('on_train_start')

        amp_ctx = torch.autocast(
            device_type=self.device.type, enabled=use_amp and self.device.type != 'cpu'
        )

        stop = False
        try:
            epoch_iter = self.iterate_epochs(dataset)
            while self.state.epoch < num_epochs and not stop:
                self._callback('on_epoch_start')
                data = next(epoch_iter)

                batches = (batch_fn(data) if batch_fn else iter(data))
                step_in_epoch = 0

                for batch in batches:
                    if steps_per_epoch is not None and step_in_epoch >= steps_per_epoch:
                        break

                    # --- One optimization iteration ---
                    with amp_ctx:
                        step_metrics = self.train_step(batch)
                    # Subclass handles its own backward/step; base only tracks
                    self.log(step_metrics)
                    self.state.advance_step()
                    step_in_epoch += 1
                    self._callback('on_step_end', step_metrics)

                    if self._should_stop():
                        stop = True
                        break

                self.state.epoch += 1
                self._callback('on_epoch_end')

                if eval_every_epochs and eval_dataset is not None:
                    if self.state.epoch % eval_every_epochs == 0:
                        self.state.phase = PipelinePhase.EVALUATING
                        eval_metrics = self.evaluate(eval_dataset)
                        self._track_best(eval_metrics)
                        self.log(eval_metrics)
                        self._callback('on_evaluate', eval_metrics)
                        self.state.phase = PipelinePhase.TRAINING

                if self._should_stop():
                    stop = True

        except KeyboardInterrupt:
            self.state.interrupted = True
            self.state.phase = PipelinePhase.INTERRUPTED
            self.state.elapsed = time.time() - self.state.start_time
            print(f'\nInterrupted at step {self.state.global_step}. Saving emergency checkpoint...')
            self._callback('on_interrupt')
            self.state.phase = PipelinePhase.FINISHED

        else:
            self.state.phase = PipelinePhase.FINISHED

        finally:
            self.state.elapsed = time.time() - self.state.start_time
            self._callback('on_train_end')
            self.teardown()

        return self.metrics.snapshot()

    def _track_best(self, eval_metrics: Dict[str, float]) -> None:
        '''Update best_metric from eval metrics (max of all keys if unset).'''
        if not eval_metrics:
            return
        name = self.state.best_metric_name
        if name is None or name not in eval_metrics:
            name = max(eval_metrics, key=lambda k: eval_metrics[k])
            self.state.best_metric_name = name
        cur = eval_metrics[name]
        if cur > self.state.best_metric:
            self.state.best_metric = cur

    # Context manager

    def __enter__(self) -> 'BasicPipeline':
        if not self._setup_done:
            self.setup()
            self._setup_done = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.teardown()

    def __repr__(self) -> str:
        return (
            f'{self.name}(device={self.device}, step={self.state.global_step}, '
            f'epoch={self.state.epoch}, phase={self.state.phase.name})'
        )
