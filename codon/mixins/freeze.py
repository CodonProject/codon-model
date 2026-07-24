import contextlib
from typing import Iterator

from codon.mixins._types import TModule


class FreezeMixin:
    def _apply_freeze(self, freeze: bool, exclude_modules=None, only=None, exclude=None):
        exclude_params = set()
        if exclude_modules:
            exclude_types = tuple(t for t in exclude_modules if isinstance(t, type))
            exclude_instances = set(m for m in exclude_modules if not isinstance(m, type))
            for _, module in self.named_modules():
                if module in exclude_instances or (exclude_types and isinstance(module, exclude_types)):
                    exclude_params.update(module.parameters())

        for name, param in self.named_parameters():
            if param in exclude_params: continue
            if exclude and any(kw in name for kw in exclude): continue
            if only and not any(kw in name for kw in only): continue
            param.requires_grad = freeze

    def freeze(self: TModule, exclude_modules=None, only=None, exclude=None) -> TModule:
        self._apply_freeze(False, exclude_modules, only, exclude); return self

    def unfreeze(self: TModule, exclude_modules=None, only=None, exclude=None) -> TModule:
        self._apply_freeze(True, exclude_modules, only, exclude); return self

    def _context_freeze(self, freeze: bool, exclude_modules=None, only=None, exclude=None):
        target_params, exclude_params = [], set()
        if exclude_modules:
            exclude_types = tuple(t for t in exclude_modules if isinstance(t, type))
            exclude_instances = set(m for m in exclude_modules if not isinstance(m, type))
            for _, module in self.named_modules():
                if module in exclude_instances or (exclude_types and isinstance(module, exclude_types)):
                    exclude_params.update(module.parameters())

        for name, param in self.named_parameters():
            if param in exclude_params: continue
            if exclude and any(kw in name for kw in exclude): continue
            if only and not any(kw in name for kw in only): continue
            target_params.append((param, param.requires_grad))
            param.requires_grad = freeze
        
        try: yield
        finally:
            for param, original_state in target_params: param.requires_grad = original_state

    @contextlib.contextmanager
    def frozen_context(self, exclude_modules=None, only=None, exclude=None) -> Iterator[None]:
        yield from self._context_freeze(False, exclude_modules, only, exclude)

    @contextlib.contextmanager
    def unfrozen_context(self, exclude_modules=None, only=None, exclude=None) -> Iterator[None]:
        yield from self._context_freeze(True, exclude_modules, only, exclude)