import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Any, Iterator, Union, TypeVar

from safetensors.torch import save_model as safe_save_model
from safetensors.torch import save_file  as safe_save_file
from safetensors.torch import load_model as safe_load_model

from codon.builtins.mixins.remote import RemoteResourceMixin

from codon.utils.safecode import safecode as utils_safecode


TBasicModel = TypeVar('TBasicModel', bound='BasicModel')

class BasicModel(nn.Module, RemoteResourceMixin):
    '''
    Base class for all models, providing common functionality like gradient checkpointing and parameter counting.
    '''
    def __init__(self):
        '''
        Initialize the BasicModel.
        '''
        super(BasicModel, self).__init__()
        self.gradient_checkpointing: bool = False
    
    @property
    def device(self) -> torch.device:
        '''
        Get the device of the model.

        Returns:
            torch.device: The device where the model parameters are located.
                          Returns 'cpu' if the model has no parameters.
        '''
        try: return next(self.parameters()).device
        except StopIteration: return torch.device('cpu')
    
    @property
    def dtype(self) -> list[torch.dtype]:
        '''
        Get the unique data types of the model's parameters, preserving the order of occurrence.
        Returns:
            list[torch.dtype]: A list of unique dtypes ordered by their occurrence in the parameters.
        '''
        return list(dict.fromkeys(p.dtype for p in self.parameters()))
    
    def set_checkpoint(self, value:bool) -> None:
        '''
        Enable or disable gradient checkpointing for the model and its sub-modules.

        Args:
            value (bool): True to enable gradient checkpointing, False to disable.
        '''
        self.gradient_checkpointing = value
        for model in self.modules():
            if isinstance(model, BasicModel) and model is not self:
                model.gradient_checkpointing = value

    def checkpoint(self, function:Callable, *args, **kwargs) -> Any:
        '''
        Apply gradient checkpointing to a function if enabled and in training mode.

        Args:
            function (Callable): The function to be checkpointed.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Any: The output of the function.
        '''
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                function, *args, use_reentrant=False, **kwargs
            )
        return function(*args, **kwargs)
    
    @property
    def trainable_params(self) -> Iterator[torch.nn.Parameter]:
        return self.get_params(trainable_only=True)
    
    def get_params(self, trainable_only:bool=False) -> Iterator[torch.nn.Parameter]:
        '''
        Get an iterator over the model parameters.

        Args:
            trainable_only (bool, optional): If True, only yield parameters that require gradients.
                                             Defaults to False.

        Returns:
            Iterator[torch.nn.Parameter]: An iterator over the model parameters.
        '''
        if trainable_only:
            return (p for p in self.parameters() if p.requires_grad)
        return self.parameters()
    
    def count_params(self, trainable_only:bool=False, active_only:bool=False, human_readable:bool=False, seen:set=None) -> Union[int, str]:
        '''
        Count the number of parameters in the model.

        Args:
            trainable_only (bool, optional): If True, count only trainable parameters.
                                             Defaults to False.
            active_only (bool, optional): If True, count only active parameters (e.g. for MoE).
                                          Defaults to False.
            human_readable (bool, optional): If True, return a string representation with units (e.g. M, B).
                                             Defaults to False.
            seen (set, optional): A set of already counted parameters to avoid duplicates.
                                  Defaults to None.

        Returns:
            Union[int, str]: The total number of parameters.
        '''
        if seen is None:
            seen = set()

        if not active_only:
            total = 0
            for p in self.get_params(trainable_only):
                if p not in seen:
                    seen.add(p)
                    total += p.numel()
        else:
            total = self._count_params_recursive(self, trainable_only, active_only, seen)
        
        if human_readable:
            if total >= 1e9:
                return f'{total / 1e9:.2f}B'
            elif total >= 1e6:
                return f'{total / 1e6:.2f}M'
            elif total >= 1e3:
                return f'{total / 1e3:.2f}K'
            return str(total)

        return total

    @staticmethod
    def _count_params_recursive(module: nn.Module, trainable_only: bool, active_only: bool, seen: set) -> int:
        total = 0
        for p in module.parameters(recurse=False):
            if p not in seen:
                if not trainable_only or p.requires_grad:
                    seen.add(p)
                    total += p.numel()
        
        for child in module.children():
            if isinstance(child, BasicModel):
                total += child.count_params(trainable_only, active_only, seen=seen)
            else:
                total += BasicModel._count_params_recursive(child, trainable_only, active_only, seen)
        
        return total
    
    def load_pretrained(self: TBasicModel, path: str, strict: bool = False) -> TBasicModel:
        '''
        Load a pretrained model from a file.
        Args:
            path (str): The path to the model file.
            strict (bool, optional): Whether to strictly enforce that the keys
                                     in state_dict match. Defaults to False.
        '''
        if path.endswith('.safetensors'):
            safe_load_model(self, path, strict=strict)
            return self
        state_dict = torch.load(path, map_location=self.device)
        if isinstance(state_dict, dict):
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
        
        clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        self.load_state_dict(clean_state_dict, strict=strict)
        return self
    
    def save_pretrained(
            self: TBasicModel, 
            path: str, 
            trainable_only: bool = False, 
            include_buffer: bool = True, 
            exclude_modules: list[Union[type, nn.Module]] = None,
            only: list[str] = None,
            exclude: list[str] = None
        ) -> TBasicModel:
        '''
        Save the model to a file.

        Args:
            path (str): The path to save the model file.
            trainable_only (bool, optional): If True, only save parameters that require gradients.
            include_buffer (bool, optional): If False, exclude registered buffers from the saved file.
            exclude_modules (list[Union[type, nn.Module]], optional): Module types or instances to exclude.
            only (list[str], optional): If provided, only save parameters whose keys contain ANY of these strings.
            exclude (list[str], optional): If provided, exclude parameters whose keys contain ANY of these strings.
        '''
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

        has_filter = trainable_only or not include_buffer or exclude_prefixes or only or exclude

        if has_filter:
            trainable_names = {name for name, p in self.named_parameters() if p.requires_grad}
            buffer_names = {name for name, _ in self.named_buffers()}
            
            filtered_dict = {}
            for key, tensor in state_dict.items():
                keep = True
                
                if exclude_prefixes and key.startswith(exclude_prefixes):
                    keep = False
                
                elif exclude and any(kw in key for kw in exclude):
                    keep = False
                
                elif only and not any(kw in key for kw in only):
                    keep = False
                
                else:
                    is_buffer = key in buffer_names
                    if not include_buffer and is_buffer:
                        keep = False
                    elif trainable_only and not is_buffer and key not in trainable_names:
                        keep = False
                
                if keep:
                    filtered_dict[key] = tensor
                else:
                    is_modified = True
            
            if is_modified:
                state_dict = filtered_dict

        if path.endswith('.safetensors'):
            if not is_modified:
                safe_save_model(self, path)
            else:
                safe_save_file(state_dict, path)
        else:
            torch.save(state_dict, path)
            
        return self
    
    def freeze(self: TBasicModel) -> TBasicModel:
        '''
        Freeze all parameters in the model by setting requires_grad to False.

        Returns:
            BasicModel: The model itself for method chaining.
        '''
        for param in self.parameters():
            param.requires_grad = False
        return self

    def unfreeze(self: TBasicModel) -> TBasicModel:
        '''
        Unfreeze all parameters in the model by setting requires_grad to True.

        Returns:
            BasicModel: The model itself for method chaining.
        '''
        for param in self.parameters():
            param.requires_grad = True
        return self
    
    def compile(self: TBasicModel, dynamic:bool|None=None) -> TBasicModel:
        return torch.compile(self, dynamic=dynamic)
    
    def safecode(self: TBasicModel, length: int = 4, exclude_confusing: bool = False) -> str:
        '''
        Generates a random safe code consisting of letters and digits.

        Args:
            length (int): The length of the code to generate. Defaults to 4.
            exclude_confusing (bool): If True, excludes confusing characters
                ('0oO1iIlLq9g') to reduce human error. Defaults to False.

        Returns:
            str: The generated random code.
        '''
        return utils_safecode(length=length, exclude_confusing=exclude_confusing)