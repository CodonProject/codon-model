from codon import *

if TYPE_CHECKING:
    from codon.pipeline.base import BasicPipeline


class Callback:
    '''Base callback. All hooks are no-ops; override what you need.'''

    def on_train_start(self, pipeline: 'BasicPipeline') -> None: ...
    def on_epoch_start(self, pipeline: 'BasicPipeline') -> None: ...
    def on_step_end(self, pipeline: 'BasicPipeline', step_metrics: Dict[str, float]) -> None: ...
    def on_epoch_end(self, pipeline: 'BasicPipeline') -> None: ...
    def on_evaluate(self, pipeline: 'BasicPipeline', eval_metrics: Dict[str, float]) -> None: ...
    def on_train_end(self, pipeline: 'BasicPipeline') -> None: ...
    def on_checkpoint(self, pipeline: 'BasicPipeline', path: str) -> None: ...
    def on_interrupt(self, pipeline: 'BasicPipeline') -> None: ...
    def state_dict(self) -> Dict[str, Any]: return {}
    def load_state_dict(self, state: Dict[str, Any]) -> None: pass


class MetricLogger(Callback):
    '''Console logging on interval.'''

    def __init__(self, interval: int = 10, keys: Optional[Sequence[str]] = None):
        self.interval = interval
        self.keys = keys

    def on_step_end(self, pipeline, metrics):
        s = pipeline.state
        if s.global_step % self.interval != 0:
            return
        keys = self.keys or pipeline.metrics.keys()
        parts = [f'{k}={pipeline.metrics.running_avg(k):.4f}' for k in keys]
        print(f'[E{s.epoch+1} S{s.global_step}] ' + ' '.join(parts))


class EarlyStopping(Callback):
    '''Stop when a metric stops improving. Checks at epoch end or every `check_every` steps.'''

    def __init__(
        self,
        monitor: str,
        mode: Literal['min', 'max'] = 'max',
        patience: int = 3,
        check_every: Optional[int] = None,   # None → epoch-level check
    ):
        self.monitor, self.mode = monitor, mode
        self.patience, self.check_every = patience, check_every
        self._best = float('inf') if mode == 'min' else float('-inf')
        self._bad = 0
        self.stop_requested = False

    def _check(self, pipeline):
        cur = pipeline.metrics.running_avg(self.monitor)
        improved = (cur < self._best) if self.mode == 'min' else (cur > self._best)
        if improved:
            self._best, self._bad = cur, 0
        else:
            self._bad += 1
            if self._bad >= self.patience:
                self.stop_requested = True

    def on_step_end(self, pipeline, metrics):
        if self.check_every and pipeline.state.global_step % self.check_every == 0:
            self._check(pipeline)

    def on_epoch_end(self, pipeline):
        if not self.check_every:
            self._check(pipeline)

    def state_dict(self):
        return {'best': self._best, 'bad': self._bad}

    def load_state_dict(self, state):
        self._best, self._bad = state['best'], state['bad']


class CheckpointCallback(Callback):
    '''Periodic + best-metric checkpoints with atomic writes.'''

    def __init__(
        self,
        directory: str,
        every_steps: Optional[int] = None,
        every_epochs: Optional[int] = 1,
        monitor: Optional[str] = None,       # Save 'best_{monitor}.pt'
        keep_last: int = 3,
    ):
        self.dir = directory
        self.every_steps = every_steps
        self.every_epochs = every_epochs
        self.monitor = monitor
        self.keep_last = keep_last
        self._ckpts: List[str] = []
        os.makedirs(directory, exist_ok=True)

    def _maybe_save(self, pipeline, tag: str) -> None:
        path = os.path.join(self.dir, f'{tag}.pt')
        pipeline.save_checkpoint(path)
        self._ckpts.append(path)
        self._prune()

    def _prune(self) -> None:
        while len(self._ckpts) > self.keep_last:
            victim = self._ckpts.pop(0)
            try:
                os.remove(victim)
            except OSError:
                pass

    def on_step_end(self, pipeline, metrics):
        if self.every_steps and pipeline.state.global_step % self.every_steps == 0:
            self._maybe_save(pipeline, f'step_{pipeline.state.global_step}')

    def on_epoch_end(self, pipeline):
        if self.every_epochs and (pipeline.state.epoch + 1) % self.every_epochs == 0:
            self._maybe_save(pipeline, f'epoch_{pipeline.state.epoch + 1}')
        if self.monitor:
            cur = pipeline.metrics.running_avg(self.monitor)
            if cur >= pipeline.state.best_metric:
                self._maybe_save(pipeline, 'best')

    def on_interrupt(self, pipeline):
        self._maybe_save(pipeline, 'interrupted')