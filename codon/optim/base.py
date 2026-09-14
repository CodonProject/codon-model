from torch.optim import Optimizer

from codon.mixins import (
    DeviceDtypeMixin, ParameterMixin, TraversalMixin, SnapshotMixin
)


class BasicOptimizer(
    Optimizer,
    DeviceDtypeMixin,
    ParameterMixin,
    TraversalMixin,
    SnapshotMixin
):
    '''
    Base class for all optimizers.
    '''
    def __init__(self, params, defaults):
        super().__init__(params, defaults)
