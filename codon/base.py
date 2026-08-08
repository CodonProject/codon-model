import torch.nn as nn
from functools import wraps
from typing import TypeVar

from codon.mixins import (
    DiagnosticsMixin, ParameterMixin, ExecutionContextMixin, 
    TrainingUtilsMixin, SerializationMixin, FreezeMixin, 
    DeviceDtypeMixin, SnapshotMixin, TraversalMixin,
    RemoteResourceMixin, BuildMixin
)
from codon.utils.safecode import safecode as utils_safecode

TBasicModel = TypeVar('TBasicModel', bound='BasicModel')


class BasicModel(
    nn.Module,
    DeviceDtypeMixin,
    ParameterMixin,
    DiagnosticsMixin,
    ExecutionContextMixin,
    TrainingUtilsMixin,
    SerializationMixin,
    FreezeMixin,
    SnapshotMixin,
    TraversalMixin,
    BuildMixin,
    RemoteResourceMixin
):
    '''
    Base class for all models. 
    '''
    def __init__(self):
        super().__init__()
        self.gradient_checkpointing: bool = False
        
    @wraps(utils_safecode)
    def safecode(self, length: int = 4, exclude_confusing: bool = False) -> str:
        return utils_safecode(length=length, exclude_confusing=exclude_confusing)