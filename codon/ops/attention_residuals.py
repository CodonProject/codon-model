'''
From Kimi 'Attention Residuals' [arXiv:2603.15031 cs.CL]
'''
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttnResState:
    blocks: List[torch.Tensor] = field(default_factory=list)
    partial_block: Optional[torch.Tensor] = None


@dataclass
class AttnResOutput:
    hidden_states: torch.Tensor
    state: AttnResState
    aux_outputs: Any = None


def apply_block_attn_res(
    blocks: List[torch.Tensor],
    partial_block: torch.Tensor,
    weight: Union[nn.Parameter, nn.Linear],
    norm: nn.Module
) -> torch.Tensor:
    if partial_block is not None:
        all_sources = blocks + [partial_block]
    else:
        all_sources = blocks
    
    # [B, T, D] -> [N+1, B, T, D]
    V = torch.stack(all_sources, dim=0)
    
    K = norm(V)  # [N+1, B, T, D]
    
    if isinstance(weight, nn.Linear):
        w = weight.weight.squeeze() # [1, D] -> [D]
    elif isinstance(weight, (nn.Parameter, torch.Tensor)):
        w = weight.squeeze()        # [D] or [1, D] -> [D]
    else:
        raise TypeError(f'unsop weight type: {type(weight)}')

    # [N+1, B, T, D] @ [D] -> [N+1, B, T]
    logits = K @ w
    
    weights = F.softmax(logits, dim=0)  # [N+1, B, T]
    
    # [N+1, B, T, 1] -> [N+1, B, T, D] -> [B, T, D]
    h = (weights.unsqueeze(-1) * V).sum(dim=0)
    
    return h