import torch
import dataclasses

@dataclasses.dataclass
class ModelIssues:
    model_class_name: str = 'Model'
    nan_params: list[str] = dataclasses.field(default_factory=list)
    inf_params: list[str] = dataclasses.field(default_factory=list)
    nan_grads:  list[str] = dataclasses.field(default_factory=list)
    inf_grads:  list[str] = dataclasses.field(default_factory=list)
    unused_params: list[str] = dataclasses.field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(self.nan_params or self.inf_params or self.nan_grads or self.inf_params or self.unused_params)

    def __repr__(self) -> str:
        if not self.has_issues: return f'{self.model_class_name}(Status: HEALTHY)'
        details = []
        for field in dataclasses.fields(self):
            if field.name == 'model_class_name': continue
            val = getattr(self, field.name)
            if val and isinstance(val, list):
                details.append(f'  {field.name} ({len(val)}): {val[:3]}... (total {len(val)})' if len(val) > 3 else f'  {field.name}: {val}')
        return f'{self.model_class_name}(Status: UNHEALTHY):\n' + '\n'.join(details)

@dataclasses.dataclass
class MemoryFootprint:
    parameters_bytes: int = 0
    trainable_parameters_bytes: int = 0
    buffers_bytes: int = 0
    trainable_parameters_count: int = 0
    gradients_bytes: int = 0 
    optimizer_state_bytes: int = 0
    temporary_cache_bytes: int = 0

    @property
    def total_bytes(self) -> int: return self.parameters_bytes + self.buffers_bytes

    @property
    def training_static_bytes(self) -> int:
        return (self.parameters_bytes + self.buffers_bytes + self.gradients_bytes + self.optimizer_state_bytes + self.temporary_cache_bytes)
    
    @staticmethod
    def _format_bytes(size_bytes: int) -> str:
        if size_bytes == 0: return '0 B'
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f'{int(size_bytes)} B' if unit == 'B' else f'{size_bytes:.2f} {unit}'
            size_bytes /= 1024.0
        return f'{size_bytes:.2f} PB'

    @property
    def human_readable_parameters(self) -> str: return self._format_bytes(self.parameters_bytes)
    @property
    def human_readable_trainable(self) -> str: return self._format_bytes(self.trainable_parameters_bytes)
    @property
    def human_readable_buffers(self) -> str: return self._format_bytes(self.buffers_bytes)
    @property
    def human_readable_total(self) -> str: return self._format_bytes(self.total_bytes)
    @property
    def human_readable_gradients(self) -> str: return self._format_bytes(self.gradients_bytes)
    @property
    def human_readable_optimizer_state(self) -> str: return self._format_bytes(self.optimizer_state_bytes)
    @property
    def human_readable_temporary_cache(self) -> str: return self._format_bytes(self.temporary_cache_bytes)
    @property
    def human_readable_training_static(self) -> str: return self._format_bytes(self.training_static_bytes)

    def __repr__(self) -> str:
        lines = [
            'MemoryFootprint:',
            f'  Parameters:  {self.human_readable_parameters} (Trainable: {self.human_readable_trainable})',
            f'  Buffers:     {self.human_readable_buffers}',
            f'  Total:       {self.human_readable_total}',
        ]
        if self.trainable_parameters_count > 0:
            lines += [
                '  <Training static estimate (Adam[W], fp32, cache=15%)>',
                f'  Gradients:   {self.human_readable_gradients}',
                f'  Optimizer:   {self.human_readable_optimizer_state}',
                f'  Temp Cache:  {self.human_readable_temporary_cache}',
                f'  Static Peak: {self.human_readable_training_static}',
            ]
        return '\n'.join(lines)

class DiagnosticsMixin:
    def __init__(self):
        super().__init__()
        self._cached_issues: ModelIssues | None = None
        self._cached_state_signature: tuple | None = None

    @property
    def state_signature(self) -> tuple:
        signature = []
        for _, param in self.named_parameters():
            grad = param.grad
            grad_sig = (id(grad), grad._version) if grad is not None else (None, 0)
            signature.append((param._version, grad_sig))
        return tuple(signature)

    @property
    def unused_params(self) -> list[str]: return self.issue.unused_params

    @property
    def memory_footprint(self) -> MemoryFootprint:
        param_bytes = trainable_bytes = buffer_bytes = 0
        trainable_count = 0
        for p in self.parameters():
            numel, elem_size = p.numel(), p.element_size()
            param_bytes += numel * elem_size
            if p.requires_grad:
                trainable_bytes += numel * elem_size
                trainable_count += numel
        for b in self.buffers(): buffer_bytes += b.numel() * b.element_size()
        
        return MemoryFootprint(
            parameters_bytes=param_bytes, trainable_parameters_bytes=trainable_bytes,
            buffers_bytes=buffer_bytes, trainable_parameters_count=trainable_count,
            gradients_bytes=trainable_bytes, optimizer_state_bytes=trainable_count * 8,
            temporary_cache_bytes=int((param_bytes + trainable_count * 8) * 0.15),
        )
    
    @property
    def grad_norm(self) -> float:
        total_norm = 0.0
        for p in self.parameters():
            if p.requires_grad and p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    @property
    def has_grad(self) -> bool:
        return any(p.grad is not None for p in self.parameters() if p.requires_grad)
    
    @property
    def issue(self) -> ModelIssues:
        current_signature = self.state_signature
        if self._cached_issues is not None and self._cached_state_signature == current_signature:
            return self._cached_issues
        
        nan_params, inf_params, nan_grads, inf_grads, unused_params = [], [], [], [], []
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is None: unused_params.append(name)
            if param.numel() > 0:
                if not torch.isfinite(param).all():
                    if torch.isnan(param).any(): nan_params.append(name)
                    if torch.isinf(param).any(): inf_params.append(name)
            if param.grad is not None and param.grad.numel() > 0:
                if not torch.isfinite(param.grad).all():
                    if torch.isnan(param.grad).any(): nan_grads.append(name)
                    if torch.isinf(param.grad).any(): inf_grads.append(name)
        
        issues = ModelIssues(
            model_class_name=self.__class__.__name__, nan_params=nan_params, inf_params=inf_params,
            nan_grads=nan_grads, inf_grads=inf_grads, unused_params=unused_params if self.has_grad else []
        )
        self._cached_issues, self._cached_state_signature = issues, current_signature
        return issues