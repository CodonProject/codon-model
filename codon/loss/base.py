import inspect
import torch
from typing import Dict, Type
from dataclasses import dataclass, field

import torch.nn as nn
from functools import wraps

from codon.mixins import ParameterMixin, DeviceDtypeMixin, SnapshotMixin, TraversalMixin
from codon.utils.safecode import safecode as utils_safecode

@dataclass
class LossOutput:
    """Standardized loss result for all BasicLoss modules."""
    loss: torch.Tensor
    metrics: Dict[str, float] = field(default_factory=dict)


class BasicLoss(
    nn.Module,
    DeviceDtypeMixin,
    ParameterMixin,
    TraversalMixin,
    SnapshotMixin
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        @wraps(utils_safecode)
        def safecode(self, length: int = 4, exclude_confusing: bool = False) -> str:
            return utils_safecode(length=length, exclude_confusing=exclude_confusing)

    def __call__(self, *args, **kwds) -> LossOutput:
        return super().__call__(*args, **kwds)


_LOSS_REGISTRY: Dict[str, Type['BasicLoss']] = {}


def register_loss(name: str):
    def decorator(cls):
        if name in _LOSS_REGISTRY:
            raise ValueError(f"loss '{name}' already registered")
        cls.__loss_name__ = name
        _LOSS_REGISTRY[name] = cls
        return cls
    return decorator


def build_loss(name: str, **kwargs) -> 'BasicLoss':
    if name not in _LOSS_REGISTRY:
        raise KeyError(
            f"unknown loss '{name}'. Available: {sorted(_LOSS_REGISTRY)}"
        )
    return _LOSS_REGISTRY[name](**kwargs)


def loss_from_config(loss_cls: Type['BasicLoss'], config) -> 'BasicLoss':
    """
    Build a loss from a configclass/dataclass by matching field names to
    the loss __init__ signature. Missing non-defaulted params raise.
    """
    sig = inspect.signature(loss_cls.__init__)
    params = [
        p.name for p in sig.parameters.values()
        if p.name != 'self'
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    kwargs = {p: getattr(config, p) for p in params if hasattr(config, p)}
    missing = [
        p for p in params
        if p not in kwargs
        and sig.parameters[p].default is inspect.Parameter.empty
    ]
    if missing:
        raise ValueError(f'config missing required loss params: {missing}')
    return loss_cls(**kwargs)