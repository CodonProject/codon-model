import torch
import dataclasses
import contextlib
from typing import Any, Iterator

class ExecutionContextMixin:
    @contextlib.contextmanager
    def inference_mode(self) -> Iterator[None]:
        was_training = self.training
        self.eval()
        with torch.no_grad(): yield
        if was_training: self.train()

    @contextlib.contextmanager
    def autocast(self, enabled: bool = True, dtype: torch.dtype = torch.float16, is_accumulation_step: bool = False) -> Iterator[None]:
        device = getattr(self, 'device', None)
        if device is None:
            try: device = next(self.parameters()).device
            except StopIteration: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        device_type = device.type
        if device_type not in ['cuda', 'cpu']: device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        with torch.amp.autocast(device_type=device_type, enabled=enabled, dtype=dtype):
            if is_accumulation_step and hasattr(self, 'no_sync') and callable(getattr(self, 'no_sync')):
                with self.no_sync(): yield
            else: yield
    
    @contextlib.contextmanager
    def capture_activations(self, module_names: list[str], detach: bool = True, clone: bool = False, cpu: bool = True) -> Iterator[dict[str, Any]]:
        activations, handles = {}, []
        def process_data(data: Any) -> Any:
            if isinstance(data, torch.Tensor):
                out = data.detach() if detach else data
                out = out.clone() if clone else out
                return out.cpu() if cpu else out
            elif isinstance(data, dict):
                processed = {k: process_data(v) for k, v in data.items()}
                return processed if type(data) is dict else type(data)(**processed)
            elif dataclasses.is_dataclass(data):
                field_values = {f.name: process_data(getattr(data, f.name)) for f in dataclasses.fields(data)}
                try:
                    obj = type(data)(**{f.name: v for f, v in zip(dataclasses.fields(data), field_values.values()) if f.init})
                    for f in dataclasses.fields(data):
                        if not f.init: setattr(obj, f.name, field_values[f.name])
                    return obj
                except Exception: return field_values
            elif isinstance(data, (list, tuple)):
                processed = [process_data(v) for v in data]
                try: return type(data)(processed)
                except Exception: return processed
            return data

        def get_hook(name: str):
            return lambda module, input, output: activations.update({name: process_data(output)})

        name_to_module = dict(self.named_modules())
        try:
            for name in module_names:
                if name in name_to_module: handles.append(name_to_module[name].register_forward_hook(get_hook(name)))
            yield activations
        finally:
            for handle in handles: handle.remove()