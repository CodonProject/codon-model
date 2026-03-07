import torch
import torch.nn as nn

from typing import Callable, Any, Iterator, Union

from safetensors.torch import save_model as safe_save_model
from safetensors.torch import load_model as safe_load_model


class BasicModel(nn.Module):
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
    
    def load_pretrained(self, path: str) -> None:
        '''
        Load a pretrained model from a file.

        Args:
            path (str): The path to the model file.
        '''
        if path.endswith('.safetensors'):
            safe_load_model(self, path)
            return

        state_dict = torch.load(path, map_location=self.device)

        if isinstance(state_dict, dict):
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        
        self.load_state_dict(state_dict)
    
    def save_pretrained(self, path: str) -> None:
        '''
        Save the model to a file.

        Args:
            path (str): The path to save the model file.
        '''
        if path.endswith('.safetensors'):
            safe_save_model(self, path)
        else:
            state_dict = self.state_dict()
            torch.save(state_dict, path)
