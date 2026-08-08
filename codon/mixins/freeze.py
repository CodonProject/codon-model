import contextlib
from typing import Iterator

from codon.mixins._types import TModule


class FreezeMixin:
    def _apply_freeze(self, freeze: bool, exclude_modules=None, only=None, exclude=None, keep_lora: bool = False):
        exclude_params = set()
        if exclude_modules:
            exclude_types = tuple(t for t in exclude_modules if isinstance(t, type))
            exclude_instances = set(m for m in exclude_modules if not isinstance(m, type))
            for _, module in self.named_modules():
                if module in exclude_instances or (exclude_types and isinstance(module, exclude_types)):
                    exclude_params.update(module.parameters())

        lora_params = set()
        if keep_lora:
            from codon.block.lora import BasicLoRA
            for module in self.modules():
                if isinstance(module, BasicLoRA):
                    for name, param in module.named_parameters(recurse=False):
                        if any(kw in name for kw in ['lora_', 'dora_']):
                            lora_params.add(param)

        for name, param in self.named_parameters():
            if param in exclude_params: continue
            if exclude and any(kw in name for kw in exclude): continue
            if only and not any(kw in name for kw in only): continue
            
            if keep_lora and not freeze and param in lora_params:
                param.requires_grad = True
                continue
                
            param.requires_grad = freeze

    def freeze(self: TModule, exclude_modules=None, only=None, exclude=None, keep_lora: bool = False) -> TModule:
        self._apply_freeze(False, exclude_modules, only, exclude, keep_lora)
        return self

    def unfreeze(self: TModule, exclude_modules=None, only=None, exclude=None, keep_lora: bool = False) -> TModule:
        self._apply_freeze(True, exclude_modules, only, exclude, keep_lora)
        return self

    def freeze_backbone(self: TModule) -> TModule:
        from codon.utils.lora import freeze_backbone as _freeze_backbone
        _freeze_backbone(self)
        return self

    def _context_freeze(self, freeze: bool, exclude_modules=None, only=None, exclude=None, keep_lora: bool = False):
        target_params, exclude_params = [], set()
        if exclude_modules:
            exclude_types = tuple(t for t in exclude_modules if isinstance(t, type))
            exclude_instances = set(m for m in exclude_modules if not isinstance(m, type))
            for _, module in self.named_modules():
                if module in exclude_instances or (exclude_types and isinstance(module, exclude_types)):
                    exclude_params.update(module.parameters())

        lora_params = set()
        if keep_lora:
            from codon.block.lora import BasicLoRA
            for module in self.modules():
                if isinstance(module, BasicLoRA):
                    for name, param in module.named_parameters(recurse=False):
                        if any(kw in name for kw in ['lora_', 'dora_']):
                            lora_params.add(param)

        for name, param in self.named_parameters():
            if param in exclude_params: continue
            if exclude and any(kw in name for kw in exclude): continue
            if only and not any(kw in name for kw in only): continue
            
            target_params.append((param, param.requires_grad))
            
            if keep_lora and not freeze and param in lora_params:
                param.requires_grad = True
            else:
                param.requires_grad = freeze
        
        try: yield
        finally:
            for param, original_state in target_params: param.requires_grad = original_state

    @contextlib.contextmanager
    def frozen_context(self, exclude_modules=None, only=None, exclude=None, keep_lora: bool = False) -> Iterator[None]:
        yield from self._context_freeze(False, exclude_modules, only, exclude, keep_lora)

    @contextlib.contextmanager
    def unfrozen_context(self, exclude_modules=None, only=None, exclude=None, keep_lora: bool = False) -> Iterator[None]:
        yield from self._context_freeze(True, exclude_modules, only, exclude, keep_lora)
