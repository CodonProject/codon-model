import torch
import torch.nn as nn
from typing import Union
from codon.mixins._types import TModule


class DeviceDtypeMixin:
    @property
    def original_model(self) -> nn.Module:
        model = self
        while True:
            if hasattr(model, '_orig_mod'):
                model = getattr(model, '_orig_mod')
            elif hasattr(model, 'module') and isinstance(
                model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)
            ):
                model = getattr(model, 'module')
            else:
                break
        return model

    @property
    def device(self) -> torch.device:
        try: return next(self.original_model.parameters()).device
        except StopIteration:
            try: return next(self.original_model.buffers()).device
            except StopIteration: return torch.device('cpu')
        
    @property
    def dtypes(self) -> list[torch.dtype]:
        return list(dict.fromkeys(p.dtype for p in self.original_model.parameters()))

    @property
    def dtype(self) -> torch.dtype:
        dtypes = [p.dtype for p in self.original_model.parameters()]
        return max(set(dtypes), key=dtypes.count) if dtypes else torch.float32

    def to_precision(self: TModule, dtype: torch.dtype) -> TModule:
        self.to(dtype=dtype); return self

    def to_lora_precision(self: TModule, dtype: torch.dtype = torch.float32) -> TModule:
        from codon.utils.lora import cast_lora_precision
        cast_lora_precision(self.original_model, dtype=dtype)
        return self

    def to_device(self: TModule, device: Union[str, torch.device]) -> TModule:
        self.to(device=device); return self
    
    def compiled(self: TModule, *args, **kwargs) -> TModule:
        return torch.compile(self, *args, **kwargs)