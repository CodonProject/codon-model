import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from .base import BasicModel


__version__ = '0.0.6a3'

__seed__: int | None = None
