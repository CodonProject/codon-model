import torch
import torch.nn as nn
from typing import Callable, Any

from codon.mixins._types import TModule


class TrainingUtilsMixin:
    def clip_grad_norm(self, max_norm: float, norm_type: float = 2.0) -> float:
        return torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm, norm_type=norm_type)

    def optimizer_groups(self, weight_decay: float = 1e-2) -> list[dict[str, Any]]:
        decay, no_decay = set(), set()
        whitelist = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
        blacklist = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters(recurse=False):
                fpn = f'{mn}.{pn}' if mn else pn
                if pn.endswith('bias'): no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist): decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist): no_decay.add(fpn)
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        uncategorized = param_dict.keys() - (decay | no_decay)
        for fpn in uncategorized:
            (no_decay if any(x in fpn.lower() for x in ['bias', 'norm', 'ln_', 'embed', 'scale', 'logit_scale']) else decay).add(fpn)
        
        assert not (decay & no_decay), f'Parameters {str(decay & no_decay)} in both sets!'
        return [
            {'params': [param_dict[pn] for pn in sorted(decay) if param_dict[pn].requires_grad], 'weight_decay': weight_decay},
            {'params': [param_dict[pn] for pn in sorted(no_decay) if param_dict[pn].requires_grad], 'weight_decay': 0.0},
        ]

    def tie_weights(self: TModule, source_module_path: str, target_module_path: str) -> TModule:
        modules = dict(self.named_modules())
        src, tgt = modules.get(source_module_path), modules.get(target_module_path)
        if not src or not tgt: raise ValueError(f"Modules not found.")
        if hasattr(src, 'weight') and hasattr(tgt, 'weight'):
            tgt.weight = src.weight
            if hasattr(src, 'bias') and hasattr(tgt, 'bias') and tgt.bias is not None: tgt.bias = src.bias
        else: raise AttributeError("Missing 'weight' attribute.")
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