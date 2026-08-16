import torch
import torch.nn as nn
from typing import Callable, Any, Optional, List, Dict, Union
from dataclasses import dataclass
from codon.mixins._types import TModule


@dataclass
class OptimizerGroups:
    standard: List[Dict[str, Any]]
    muon: List[Dict[str, Any]]
    adamw: List[Dict[str, Any]]


class TrainingUtilsMixin:
    _grad_norm_ema: Optional[float] = None
    _ema_decay: float = 0.99

    def clip_grad_norm(
        self,
        max_norm: Union[float, str],
        norm_type: float = 2.0,
        auto_multiplier: float = 2.0
    ) -> float:
        if isinstance(max_norm, str) and max_norm.lower() == 'auto':
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.trainable_params, float('inf'), norm_type=norm_type
            )
            ema = getattr(self, '_grad_norm_ema', None)
            if ema is None:
                ema = total_norm.item()
            if total_norm < 1000:  
                ema = self._ema_decay * ema + (1 - self._ema_decay) * total_norm.item()
            self._grad_norm_ema = ema

            adaptive_threshold = ema * auto_multiplier
            if adaptive_threshold < 1e-8:
                adaptive_threshold = 1.0
            max_norm = adaptive_threshold

        return torch.nn.utils.clip_grad_norm_(
            self.trainable_params, max_norm, norm_type=norm_type
        )

    def tie_weights(self: TModule, source_module_path: str, target_module_path: str) -> TModule:
        modules = dict(self.named_modules())
        src, tgt = modules.get(source_module_path), modules.get(target_module_path)
        if not src or not tgt: raise ValueError(f'Modules not found.')
        
        from codon.block.lora import BasicLoRA
        if isinstance(src, BasicLoRA): src = src.original_layer
        if isinstance(tgt, BasicLoRA): tgt = tgt.original_layer

        if hasattr(src, 'weight') and hasattr(tgt, 'weight'):
            tgt.weight = src.weight
            if hasattr(src, 'bias') and hasattr(tgt, 'bias') and tgt.bias is not None: 
                tgt.bias = src.bias
        else: 
            raise AttributeError('Missing \'weight\' attribute.')
        return self
    
    def set_checkpoint(self: TModule, value: bool) -> TModule:
        self.gradient_checkpointing = value
        for model in self.modules():
            if hasattr(model, 'gradient_checkpointing') and model is not self: 
                model.gradient_checkpointing = value
        return self

    def checkpoint(self, function: Callable, *args, **kwargs) -> Any:
        if getattr(self, 'gradient_checkpointing', False) and self.training:
            return torch.utils.checkpoint.checkpoint(function, *args, use_reentrant=False, **kwargs)
        return function(*args, **kwargs)