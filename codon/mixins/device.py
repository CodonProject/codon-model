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

    @staticmethod
    def _pick_cuda_device() -> torch.device:
        '''
        Pick the CUDA device with the most free memory as the preferred single device.
        Falls back to visible device 0 when free memory cannot be queried.
        '''
        count = torch.cuda.device_count()
        best, best_free = 0, -1.0
        for i in range(count):
            try:
                free, _ = torch.cuda.mem_get_info(i)
            except Exception:
                free = -1.0
            if free > best_free:
                best_free, best = free, i
        return torch.device(f'cuda:{best}')

    @staticmethod
    def _detect_tpu() -> Union[torch.device, None]:
        '''
        Detect a TPU through PyTorch/XLA. Returns the first XLA TPU device, or
        None when torch_xla is unavailable or no TPU device kind is exposed.
        The import is lazy so CPU/GPU-only machines never pay for torch_xla.
        '''
        try:
            import torch_xla.core.xla_model as xm
        except Exception:
            return None
        try:
            if not xm.get_xla_supported_devices(devkind='TPU'):
                return None
            return xm.xla_device()
        except Exception:
            return None

    @staticmethod
    def _detect_device() -> torch.device:
        '''
        Detect the best available device for this environment:
        CUDA > TPU (PyTorch/XLA) > MPS > CPU.
        '''
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return DeviceDtypeMixin._pick_cuda_device()
        tpu = DeviceDtypeMixin._detect_tpu()
        if tpu is not None:
            return tpu
        mps_backend = getattr(torch.backends, 'mps', None)
        if mps_backend is not None and getattr(mps_backend, 'is_available', lambda: False)():
            return torch.device('mps')
        return torch.device('cpu')

    def auto_device(self: TModule) -> TModule:
        '''
        Automatically detect the environment and move the model onto the best
        single device.

        Detection precedence: CUDA > TPU (PyTorch/XLA) > MPS > CPU. Under CUDA the
        device with the most free memory is preferred. Multi-device parallelism is
        intentionally left to the caller / training pipeline (DDP, torch_xla SPMD),
        so this always moves onto exactly one device.

        Returns:
            ``self`` after being moved onto the detected device.
        '''
        self.to(device=self._detect_device())
        return self

    def compiled(self: TModule, *args, **kwargs) -> TModule:
        return torch.compile(self, *args, **kwargs)