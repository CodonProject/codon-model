import torch
import torch.nn as nn


def _constrain(raw: torch.Tensor, low: torch.Tensor, high: torch.Tensor):
    s = torch.sigmoid(raw)
    out = low + (high - low) * s
    return out.clamp(min=low, max=high)

def _check_bound(bound):
    if not isinstance(bound, (tuple, list)) or len(bound) != 2:
        raise TypeError("bound 必须是 (float, float) 形式的元组或列表")
    low, high = float(bound[0]), float(bound[1])
    if not (low < high):
        raise ValueError("low 必须严格小于 high")
    return low, high

class BoundedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, raw, low, high):
        if low >= high:
            raise ValueError('low 必须严格小于 high')
        return _constrain(raw, low, high).as_subclass(cls)

    def __init__(self, raw, low, high):
        self._raw = raw
        self._low = low
        self._high = high

    def fresh(self):
        return _constrain(self._raw, self._low, self._high)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else dict(kwargs)
        args = [
            a.fresh() if isinstance(a, BoundedTensor) and hasattr(a, 'raw') else a
            for a in args
        ]
        kwargs = {
            k: v.fresh() if isinstance(v, BoundedTensor) and hasattr(v, 'raw') else v
            for k, v in kwargs.items()
        }
        return func(*args, **kwargs)

    @classmethod
    def from_value(cls, value, low, high):
        value = torch.as_tensor(value, dtype=torch.float32)
        t = ((value - low) / (high - low)).clamp(min=1e-6, max=1.0 - 1e-6)
        raw = nn.Parameter(torch.log(t / (1.0 - t)))
        return cls(raw, low, high)

    def sum(self, *args, **kwargs):
        return self.fresh().sum(*args, **kwargs)

    def mean(self, *args, **kwargs):
        return self.fresh().mean(*args, **kwargs)

    def __repr__(self):
        if hasattr(self, 'raw'):
            return f'BoundedTensor(value={self.fresh().detach().tolist()}, low={self._low}, high={self._high})'
        return f'BoundedTensor(data={self.detach().tolist()})'
    
    @staticmethod
    def randn(*shape, bound=(-1.0, 1.0), dtype=None, device=None, requires_grad=True):
        low, high = _check_bound(bound)
        raw = torch.randn(*shape, dtype=dtype, device=device)
        if requires_grad: raw = nn.Parameter(raw)
        return BoundedTensor(raw, low, high)

    @staticmethod
    def single(bound=(-1.0, 1.0), dtype=None, device=None, requires_grad=True):
        low, high = _check_bound(bound)
        raw = torch.tensor(0.0, dtype=dtype, device=device)
        if requires_grad: raw = nn.Parameter(raw)
        return BoundedTensor(raw, low, high)

    @staticmethod
    def bounded(n, bound=(-1.0, 1.0), dtype=None, device=None, requires_grad=True):
        low, high = _check_bound(bound)
        raw = torch.zeros(n, dtype=dtype, device=device)
        if requires_grad: raw = nn.Parameter(raw)
        return BoundedTensor(raw, low, high)

