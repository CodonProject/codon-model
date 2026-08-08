import torch.nn as nn
from typing import List, Type, Union

from codon.mixins._types import TModule


class BuildMixin:
    def inject_lora(
        self: TModule,
        target_modules: Union[List[str], List[Type[nn.Module]]],
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        merge_weights: bool = False,
        gate: bool = False,
        dora: bool = False,
        gradient_checkpointing: bool = False
    ) -> TModule:
        from codon.utils.lora import inject
        inject(
            self,
            target_modules=target_modules,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            gate=gate,
            dora=dora,
            gradient_checkpointing=gradient_checkpointing
        )
        return self

    def inject_lora_from_file(
        self: TModule,
        path: str,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        merge_weights: bool = False,
        gradient_checkpointing: bool = False
    ) -> TModule:
        from codon.utils.lora import inject_from_file
        inject_from_file(
            self,
            path=path,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            gradient_checkpointing=gradient_checkpointing
        )
        return self

    def merge_lora(self: TModule) -> TModule:
        from codon.utils.lora import merge_all
        merge_all(self)
        return self

    def unmerge_lora(self: TModule) -> TModule:
        from codon.utils.lora import unmerge_all
        unmerge_all(self)
        return self