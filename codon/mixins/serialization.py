import torch
import torch.nn as nn
from typing import Union

from safetensors.torch import save_model as safe_save_model
from safetensors.torch import save_file  as safe_save_file
from safetensors.torch import load_model as safe_load_model

from codon.mixins._types import TModule


class SerializationMixin:
    def load_pretrained(self: TModule, path: str, strict: bool = False) -> TModule:
        if path.endswith('.safetensors'):
            safe_load_model(self, path, strict=strict)
            return self
        
        device = getattr(self, 'device', torch.device('cpu'))
        state_dict = torch.load(path, map_location=device)
        if isinstance(state_dict, dict):
            for key in ['model_state_dict', 'state_dict', 'model']:
                if key in state_dict: state_dict = state_dict[key]; break
        
        clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        self.load_state_dict(clean_state_dict, strict=strict)
        return self
    
    def save_pretrained(self: TModule, path: str, trainable_only: bool = False, include_buffer: bool = True, 
                        exclude_modules: list[Union[type, nn.Module]] = None, only: list[str] = None, exclude: list[str] = None) -> TModule:
        state_dict = self.state_dict()
        is_modified = False
        exclude_prefixes = []
        
        if exclude_modules:
            exclude_types = tuple(t for t in exclude_modules if isinstance(t, type))
            exclude_instances = set(m for m in exclude_modules if not isinstance(m, type))
            for name, module in self.named_modules():
                if module in exclude_instances or (exclude_types and isinstance(module, exclude_types)):
                    if name != '': exclude_prefixes.append(name + '.')
        exclude_prefixes = tuple(exclude_prefixes)

        if trainable_only or not include_buffer or exclude_prefixes or only or exclude:
            trainable_names = {name for name, p in self.named_parameters() if p.requires_grad}
            buffer_names = {name for name, _ in self.named_buffers()}
            filtered_dict = {}
            
            for key, tensor in state_dict.items():
                keep = True
                if exclude_prefixes and key.startswith(exclude_prefixes): keep = False
                elif exclude and any(kw in key for kw in exclude): keep = False
                elif only and not any(kw in key for kw in only): keep = False
                else:
                    is_buffer = key in buffer_names
                    if not include_buffer and is_buffer: keep = False
                    elif trainable_only and not is_buffer and key not in trainable_names: keep = False
                
                if keep: filtered_dict[key] = tensor
                else: is_modified = True
            if is_modified: state_dict = filtered_dict

        if path.endswith('.safetensors'):
            safe_save_model(self, path) if not is_modified else safe_save_file(state_dict, path)
        else:
            torch.save(state_dict, path)
        return self