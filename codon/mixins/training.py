import torch
import torch.nn as nn
from typing import Callable, Any, Optional
from codon.mixins._types import TModule


class TrainingUtilsMixin:
    def clip_grad_norm(self, max_norm: float, norm_type: float = 2.0) -> float:
        return torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm, norm_type=norm_type)

    def optimizer_groups(
        self, 
        weight_decay: float = 1e-2, 
        lora_lr: Optional[float] = None, 
        base_lr: Optional[float] = None
    ) -> list[dict[str, Any]]:
        decay, no_decay = set(), set()
        lora_decay, lora_no_decay = set(), set()
        
        whitelist = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
        blacklist = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters(recurse=False):
                fpn = f'{mn}.{pn}' if mn else pn
                is_lora = any(k in fpn for k in ['lora_', 'dora_']) and 'original_layer' not in fpn
                
                if pn.endswith('bias') or any(k in pn for k in ['lora_gate', 'dora_m']):
                    (lora_no_decay if is_lora else no_decay).add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist):
                    (lora_decay if is_lora else decay).add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist):
                    (lora_no_decay if is_lora else no_decay).add(fpn)
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        categorized = decay | no_decay | lora_decay | lora_no_decay
        uncategorized = param_dict.keys() - categorized
        
        for fpn in uncategorized:
            is_lora = any(k in fpn for k in ['lora_', 'dora_']) and 'original_layer' not in fpn
            if any(x in fpn.lower() for x in ['bias', 'norm', 'ln_', 'embed', 'scale', 'gate', 'dora_m']):
                (lora_no_decay if is_lora else no_decay).add(fpn)
            else:
                (lora_decay if is_lora else decay).add(fpn)

        groups = []
        def _add_group(param_names, wd, lr):
            params = [param_dict[name] for name in sorted(param_names) if param_dict[name].requires_grad]
            if params:
                group = {'params': params, 'weight_decay': wd}
                if lr is not None: group['lr'] = lr
                groups.append(group)

        _add_group(decay, weight_decay, base_lr)
        _add_group(no_decay, 0.0, base_lr)
        _add_group(lora_decay, weight_decay, lora_lr or base_lr)
        _add_group(lora_no_decay, 0.0, lora_lr or base_lr)

        return groups

    def tie_weights(self: TModule, source_module_path: str, target_module_path: str) -> TModule:
        modules = dict(self.named_modules())
        src, tgt = modules.get(source_module_path), modules.get(target_module_path)
        if not src or not tgt: raise ValueError(f"Modules not found.")
        
        from codon.block.lora import BasicLoRA
        if isinstance(src, BasicLoRA): src = src.original_layer
        if isinstance(tgt, BasicLoRA): tgt = tgt.original_layer

        if hasattr(src, 'weight') and hasattr(tgt, 'weight'):
            tgt.weight = src.weight
            if hasattr(src, 'bias') and hasattr(tgt, 'bias') and tgt.bias is not None: 
                tgt.bias = src.bias
        else: 
            raise AttributeError("Missing 'weight' attribute.")
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