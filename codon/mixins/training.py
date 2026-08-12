import torch
import torch.nn as nn
from typing import Callable, Any, Optional, List, Dict
from dataclasses import dataclass
from codon.mixins._types import TModule


@dataclass
class OptimizerGroups:
    standard: List[Dict[str, Any]]
    muon: List[Dict[str, Any]]
    adamw: List[Dict[str, Any]]


class TrainingUtilsMixin:
    def clip_grad_norm(self, max_norm: float, norm_type: float = 2.0) -> float:
        return torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm, norm_type=norm_type)

    def optimizer_groups(
        self, 
        weight_decay: float = 1e-2, 
        lora_lr: Optional[float] = None, 
        base_lr: Optional[float] = None,
        use_muon_for_lora: bool = False
    ) -> OptimizerGroups:
        param_to_module = {}
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                param_to_module[p] = (mn, m, pn)

        std_decay_base, std_nodecay_base = [], []
        std_decay_lora, std_nodecay_lora = [], []

        muon_base, muon_lora = [], []
        adamw_decay_base, adamw_nodecay_base = [], []
        adamw_decay_lora, adamw_nodecay_lora = [], []

        for fpn, p in self.named_parameters():
            if not p.requires_grad:
                continue

            mn, m, pn = param_to_module.get(p, ('', None, fpn.split('.')[-1]))

            is_lora = any(k in fpn for k in ['lora_', 'dora_']) and 'original_layer' not in fpn
            
            is_embed = 'embed' in fpn.lower() or (m is not None and isinstance(m, nn.Embedding))
            
            is_bias_or_norm = (
                p.ndim < 2
                or pn.endswith('bias')
                or any(k in pn for k in ['lora_gate', 'dora_m'])
                or any(x in fpn.lower() for x in ['norm', 'ln_', 'scale', 'gate'])
                or (m is not None and isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)))
            )

            is_decay = not is_bias_or_norm and not is_embed
            
            is_muon_candidate = p.ndim >= 2 and not is_embed and not is_bias_or_norm

            if is_lora:
                (std_decay_lora if is_decay else std_nodecay_lora).append(p)
            else:
                (std_decay_base if is_decay else std_nodecay_base).append(p)

            if is_muon_candidate and (use_muon_for_lora or not is_lora):
                (muon_lora if is_lora else muon_base).append(p)
            else:
                if is_lora:
                    (adamw_decay_lora if is_decay else adamw_nodecay_lora).append(p)
                else:
                    (adamw_decay_base if is_decay else adamw_nodecay_base).append(p)

        def _add_group(target_list: list, params: list, wd: float, lr: Optional[float]):
            if params:
                group = {'params': params, 'weight_decay': wd}
                if lr is not None:
                    group['lr'] = lr
                target_list.append(group)

        standard_groups = []
        _add_group(standard_groups, std_decay_base, weight_decay, base_lr)
        _add_group(standard_groups, std_nodecay_base, 0.0, base_lr)
        _add_group(standard_groups, std_decay_lora, weight_decay, lora_lr or base_lr)
        _add_group(standard_groups, std_nodecay_lora, 0.0, lora_lr or base_lr)

        muon_groups = []
        _add_group(muon_groups, muon_base, 0.0, base_lr)
        _add_group(muon_groups, muon_lora, 0.0, lora_lr or base_lr)

        adamw_groups = []
        _add_group(adamw_groups, adamw_decay_base, weight_decay, base_lr)
        _add_group(adamw_groups, adamw_nodecay_base, 0.0, base_lr)
        _add_group(adamw_groups, adamw_decay_lora, weight_decay, lora_lr or base_lr)
        _add_group(adamw_groups, adamw_nodecay_lora, 0.0, lora_lr or base_lr)

        return OptimizerGroups(
            standard=standard_groups,
            muon=muon_groups,
            adamw=adamw_groups
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