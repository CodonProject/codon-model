import torch
import torch.nn as nn
from typing import Iterator, Union


class ParameterMixin:
    @property
    def _orig(self) -> nn.Module:
        return getattr(self, 'original_model', self)

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
        for name, param in self._orig.named_parameters():
            if not any(kw in name for kw in ['lora_', 'dora_']) or 'original_layer' in name:
                yield param

    def get_params(self, trainable_only: bool = False, lora_only: bool = False) -> Iterator[torch.nn.Parameter]:
        for name, p in self._orig.named_parameters():
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
            total = self._count_params_recursive(self._orig, trainable_only, active_only, lora_only, seen)
        
        if human_readable:
            if total >= 1e9: return f'{total / 1e9:.2f}B'
            elif total >= 1e6: return f'{total / 1e6:.2f}M'
            elif total >= 1e3: return f'{total / 1e3:.2f}K'
            return str(total)
        return total

    @staticmethod
    def _unwrap_model(module: nn.Module) -> nn.Module:
        m = module
        while True:
            if hasattr(m, '_orig_mod'):
                m = getattr(m, '_orig_mod')
            elif hasattr(m, 'module') and isinstance(m, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)):
                m = getattr(m, 'module')
            else:
                break
        return m

    @staticmethod
    def _count_params_recursive(module: nn.Module, trainable_only: bool, active_only: bool, lora_only: bool, seen: set) -> int:
        total = 0
        unwrapped = ParameterMixin._unwrap_model(module)
        
        for name, p in unwrapped.named_parameters(recurse=False):
            if p not in seen:
                if trainable_only and not p.requires_grad:
                    continue
                if lora_only:
                    is_lora = any(kw in name for kw in ['lora_', 'dora_']) and 'original_layer' not in name
                    if not is_lora:
                        continue
                        
                seen.add(p)
                total += p.numel()
                
        for child in unwrapped.children():
            child_unwrapped = ParameterMixin._unwrap_model(child)
            if hasattr(child_unwrapped, 'count_params'):
                total += child_unwrapped.count_params(
                    trainable_only=trainable_only, 
                    active_only=active_only, 
                    lora_only=lora_only, 
                    seen=seen
                )
            else:
                total += ParameterMixin._count_params_recursive(child_unwrapped, trainable_only, active_only, lora_only, seen)
        return total