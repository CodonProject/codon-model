import torch
import torch.nn as nn
from typing import Iterator, Union


class ParameterMixin:
    @property
    def trainable_params(self) -> Iterator[torch.nn.Parameter]:
        return self.get_params(trainable_only=True)
    
    @property
    def all_params(self) -> Iterator[torch.nn.Parameter]:
        return self.get_params()

    @property
    def lora_params(self) -> Iterator[torch.nn.Parameter]:
        return self.get_params(lora_only=True)

    @property
    def backbone_params(self) -> Iterator[torch.nn.Parameter]:
        for name, param in self.named_parameters():
            if not any(kw in name for kw in ['lora_', 'dora_']) or 'original_layer' in name:
                yield param

    def get_params(self, trainable_only: bool = False, lora_only: bool = False) -> Iterator[torch.nn.Parameter]:
        for name, p in self.named_parameters():
            if trainable_only and not p.requires_grad:
                continue
            if lora_only:
                is_lora = any(kw in name for kw in ['lora_', 'dora_']) and 'original_layer' not in name
                if not is_lora:
                    continue
            yield p
    
    def count_params(
        self, 
        trainable_only: bool = False, 
        active_only: bool = False, 
        lora_only: bool = False, 
        human_readable: bool = False, 
        seen: set = None
    ) -> Union[int, str]:
        if seen is None: seen = set()
        
        if not active_only:
            total = sum(p.numel() for p in self.get_params(trainable_only, lora_only) if p not in seen and not seen.add(p))
        else:
            total = self._count_params_recursive(self, trainable_only, active_only, lora_only, seen)
        
        if human_readable:
            if total >= 1e9: return f'{total / 1e9:.2f}B'
            elif total >= 1e6: return f'{total / 1e6:.2f}M'
            elif total >= 1e3: return f'{total / 1e3:.2f}K'
            return str(total)
        return total

    @staticmethod
    def _count_params_recursive(module: nn.Module, trainable_only: bool, active_only: bool, lora_only: bool, seen: set) -> int:
        total = 0
        for name, p in module.named_parameters(recurse=False):
            if p not in seen:
                if trainable_only and not p.requires_grad:
                    continue
                if lora_only:
                    is_lora = any(kw in name for kw in ['lora_', 'dora_']) and 'original_layer' not in name
                    if not is_lora:
                        continue
                        
                seen.add(p)
                total += p.numel()
                
        for child in module.children():
            if hasattr(child, 'count_params'):
                total += child.count_params(
                    trainable_only=trainable_only, 
                    active_only=active_only, 
                    lora_only=lora_only, 
                    seen=seen
                )
            else:
                total += ParameterMixin._count_params_recursive(child, trainable_only, active_only, lora_only, seen)
        return total