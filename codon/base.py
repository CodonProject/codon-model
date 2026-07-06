import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Any, Iterator, Union, TypeVar

from safetensors.torch import save_model as safe_save_model
from safetensors.torch import save_file  as safe_save_file
from safetensors.torch import load_model as safe_load_model

from codon.mixin import RemoteResourceMixin

from codon.utils.safecode import safecode as utils_safecode

import dataclasses
import contextlib


@dataclasses.dataclass
class ModelIssues:
    model_class_name: str = 'Model'
    nan_params: list[str] = dataclasses.field(default_factory=list)
    inf_params: list[str] = dataclasses.field(default_factory=list)
    nan_grads:  list[str] = dataclasses.field(default_factory=list)
    inf_grads:  list[str] = dataclasses.field(default_factory=list)
    unused_params: list[str] = dataclasses.field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(
            self.nan_params or 
            self.inf_params or 
            self.nan_grads or 
            self.inf_grads or 
            self.unused_params
        )

    def __repr__(self) -> str:
        if not self.has_issues:
            return f'{self.model_class_name}(Status: HEALTHY)'
        
        details = []
        for field in dataclasses.fields(self):
            if field.name == 'model_class_name': continue
            val = getattr(self, field.name)
            if val and isinstance(val, list):
                details.append(f'  {field.name} ({len(val)}): {val[:3]}... (total {len(val)})' if len(val) > 3 else f'  {field.name}: {val}')
        return f'{self.model_class_name}(Status: UNHEALTHY):\n' + '\n'.join(details)

@dataclasses.dataclass
class MemoryFootprint:
    parameters_bytes: int = 0
    trainable_parameters_bytes: int = 0
    buffers_bytes: int = 0

    trainable_parameters_count: int = 0
    gradients_bytes: int = 0 
    optimizer_state_bytes: int = 0
    temporary_cache_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        '''Get the total memory footprint in bytes (parameters + buffers).'''
        return self.parameters_bytes + self.buffers_bytes

    @property
    def training_static_bytes(self) -> int:
        return (self.parameters_bytes + self.buffers_bytes + 
                self.gradients_bytes + self.optimizer_state_bytes + 
                self.temporary_cache_bytes)
    
    @staticmethod
    def _format_bytes(size_bytes: int) -> str:
        '''Helper to format bytes into a human-readable string.'''
        if size_bytes == 0:
            return '0 B'
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                if unit == 'B':
                    return f'{int(size_bytes)} B'
                return f'{size_bytes:.2f} {unit}'
            size_bytes /= 1024.0
        return f'{size_bytes:.2f} PB'

    @property
    def human_readable_parameters(self) -> str:
        '''Human-readable size of all parameters (e.g., '350.23 MB').'''
        return self._format_bytes(self.parameters_bytes)

    @property
    def human_readable_trainable(self) -> str:
        '''Human-readable size of trainable parameters (e.g., '12.45 MB').'''
        return self._format_bytes(self.trainable_parameters_bytes)

    @property
    def human_readable_buffers(self) -> str:
        '''Human-readable size of buffers (e.g., '4.12 KB').'''
        return self._format_bytes(self.buffers_bytes)

    @property
    def human_readable_total(self) -> str:
        '''Human-readable size of the total memory footprint (e.g., '362.68 MB').'''
        return self._format_bytes(self.total_bytes)
    
    @property
    def human_readable_gradients(self) -> str:
        return self._format_bytes(self.gradients_bytes)

    @property
    def human_readable_optimizer_state(self) -> str:
        return self._format_bytes(self.optimizer_state_bytes)

    @property
    def human_readable_temporary_cache(self) -> str:
        return self._format_bytes(self.temporary_cache_bytes)

    @property
    def human_readable_training_static(self) -> str:
        return self._format_bytes(self.training_static_bytes)

    def __repr__(self) -> str:
        lines = [
            'MemoryFootprint:',
            f'  Parameters:  {self.human_readable_parameters} (Trainable: {self.human_readable_trainable})',
            f'  Buffers:     {self.human_readable_buffers}',
            f'  Total:       {self.human_readable_total}',
        ]
        if self.trainable_parameters_count > 0:
            lines += [
                '  <Training static estimate (Adam[W], fp32, cache=15%)>',
                f'  Gradients:   {self.human_readable_gradients}',
                f'  Optimizer:   {self.human_readable_optimizer_state}',
                f'  Temp Cache:  {self.human_readable_temporary_cache}',
                f'  Static Peak: {self.human_readable_training_static}',
            ]
        return '\n'.join(lines)


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
        self._snapshots = {}

        self._cached_issues: ModelIssues | None = None
        self._cached_state_signature: tuple | None = None
    
    @property
    def state_signature(self) -> tuple:
        '''
        Generate a lightweight signature representing the current state of 
        all parameters and their gradients.
        '''
        signature = []
        for _, param in self.named_parameters():
            grad = param.grad
            grad_sig = (id(grad), grad._version) if grad is not None else (None, 0)
            signature.append((param._version, grad_sig))
        return tuple(signature)

    @property
    def device(self) -> torch.device:
        '''
        Get the device of the model.

        Returns:
            torch.device: The device where the model parameters are located.
                          Returns 'cpu' if the model has no parameters.
        '''
        try: return next(self.parameters()).device
        except StopIteration:
            try: return next(self.buffers()).device
            except StopIteration: return torch.device('cpu')
        
    @property
    def dtypes(self) -> list[torch.dtype]:
        '''
        Get the unique data types of the model's parameters, preserving the order of occurrence.
        Returns:
            list[torch.dtype]: A list of unique dtypes ordered by their occurrence in the parameters.
        '''
        return list(dict.fromkeys(p.dtype for p in self.parameters()))

    @property
    def dtype(self) -> torch.dtype:
        '''Get the most common dtype of the model parameters.'''
        dtypes = [p.dtype for p in self.parameters()]
        if not dtypes: return torch.float32
        return max(set(dtypes), key=dtypes.count)
    
    @property
    def trainable_params(self) -> Iterator[torch.nn.Parameter]:
        return self.get_params(trainable_only=True)
    
    @property
    def unused_params(self) -> list[str]:
        '''
        Find parameters that require gradients but did not receive any (grad is None).
        Useful for debugging DDP issues or identifying dead code paths.
        '''
        return self.issue.unused_params

    @property
    def memory_footprint(self) -> MemoryFootprint:
        '''
        Estimate the memory footprint of the model parameters and buffers.
        Returns a MemoryFootprint dataclass.
        '''
        param_bytes = 0
        trainable_bytes = 0
        trainable_count = 0
        buffer_bytes = 0

        for p in self.parameters():
            numel = p.numel()
            elem_size = p.element_size()
            param_bytes += numel * elem_size
            if p.requires_grad:
                trainable_bytes += numel * elem_size
                trainable_count += numel

        for b in self.buffers():
            buffer_bytes += b.numel() * b.element_size()

        gradients = trainable_bytes
        
        optimizer_state = trainable_count * 8
        
        cache = int((param_bytes + optimizer_state) * 0.15)

        return MemoryFootprint(
            parameters_bytes=param_bytes,
            trainable_parameters_bytes=trainable_bytes,
            buffers_bytes=buffer_bytes,
            trainable_parameters_count=trainable_count,
            gradients_bytes=gradients,
            optimizer_state_bytes=optimizer_state,
            temporary_cache_bytes=cache,
        )
    
    @property
    def grad_norm(self) -> float:
        '''
        Calculate the total L2 gradient norm of all trainable parameters.
        Useful for training diagnostics and logging.
        '''
        total_norm = 0.0
        for p in self.trainable_params:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    @property
    def has_grad(self) -> bool:
        for p in self.trainable_params:
            if p.grad is not None: return True
        return False
    
    @property
    def issue(self) -> ModelIssues:
        '''
        Get the current model issues (NaN/Inf in weights/gradients, and unused parameters).
        Results are cached and only re-calculated when parameters or gradients change.
        '''
        current_signature = self.state_signature
        
        if self._cached_issues is not None and self._cached_state_signature == current_signature:
            return self._cached_issues
        
        nan_params = []
        inf_params = []
        nan_grads  = []
        inf_grads  = []
        unused_params = []

        for name, param in self.named_parameters():
            
            if param.requires_grad and param.grad is None:
                unused_params.append(name)
                
            if param.numel() > 0:
                if not torch.isfinite(param).all():
                    if torch.isnan(param).any():
                        nan_params.append(name)
                    if torch.isinf(param).any():
                        inf_params.append(name)
                        
            if param.grad is not None and param.grad.numel() > 0:
                if not torch.isfinite(param.grad).all():
                    if torch.isnan(param.grad).any():
                        nan_grads.append(name)
                    if torch.isinf(param.grad).any():
                        inf_grads.append(name)
        
        unused_params = unused_params if self.has_grad else []

        issues = ModelIssues(
            model_class_name=self.__class__.__name__,
            nan_params=nan_params,
            inf_params=inf_params,
            nan_grads=nan_grads,
            inf_grads=inf_grads,
            unused_params=unused_params
        )
        
        self._cached_issues = issues
        self._cached_state_signature = current_signature
        
        return issues

    @contextlib.contextmanager
    def inference_mode(self) -> Iterator[None]:
        '''
        A context manager that temporarily sets the model to evaluation mode 
        and enables torch.no_grad(), restoring the original training state afterwards.
        '''
        was_training = self.training
        self.eval()
        with torch.no_grad(): yield
        if was_training: self.train()

    @contextlib.contextmanager
    def autocast_context(self, enabled: bool = True, dtype: torch.dtype = torch.float16) -> Iterator[None]:
        '''
        Context manager for PyTorch Automatic Mixed Precision (AMP).
        '''
        device_type = self.device.type
        
        if device_type not in ['cuda', 'cpu']:
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        with torch.amp.autocast(device_type=device_type, enabled=enabled, dtype=dtype): yield

    @contextlib.contextmanager
    def accumulation_context(self, is_accumulation_step: bool) -> Iterator[None]:
        '''
        Context manager for gradient accumulation. 
        If in DDP mode and it is an accumulation step (not the step to step optimizer),
        it disables gradient synchronization to speed up training.
        
        Args:
            is_accumulation_step (bool): True if this forward/backward pass is an accumulation step.
        '''
        if is_accumulation_step and hasattr(self, 'no_sync') and callable(getattr(self, 'no_sync')):
            with self.no_sync(): yield
        else: yield
    
    @contextlib.contextmanager
    def capture_activations(
        self, 
        module_names: list[str], 
        detach: bool = True, 
        clone: bool = False,
        cpu: bool = True
    ) -> Iterator[dict[str, Any]]:
        '''
        Temporarily capture intermediate activations of specified sub-modules during forward pass.
        Supports nested structures including Tensors, dicts, lists, tuples, and dataclasses.
        Automatically removes hooks upon exit to prevent memory leaks.

        Args:
            module_names (list[str]): List of sub-module names to capture.
            detach (bool): If True, detach tensors from the computation graph to prevent memory leaks.
            clone (bool): If True, clone the tensors to get independent copies.
            cpu (bool): If True, move captured tensors to CPU memory to save GPU memory.
        '''
        activations = {}
        handles = []

        def process_data(data: Any) -> Any:
            if isinstance(data, torch.Tensor):
                out = data
                if detach:
                    out = out.detach()
                if clone:
                    out = out.clone()
                if cpu:
                    out = out.cpu()
                return out
            
            elif isinstance(data, dict):
                processed = {k: process_data(v) for k, v in data.items()}
                if type(data) is dict:
                    return processed
                try:
                    return type(data)(**processed)
                except Exception:
                    try:
                        return type(data)(processed)
                    except Exception:
                        return processed
            
            elif dataclasses.is_dataclass(data):
                field_values = {}
                init_values = {}
                for f in dataclasses.fields(data):
                    val = getattr(data, f.name)
                    processed_val = process_data(val)
                    field_values[f.name] = processed_val
                    if f.init:
                        init_values[f.name] = processed_val
                try:
                    obj = type(data)(**init_values)
                    for f in dataclasses.fields(data):
                        if not f.init:
                            try:
                                setattr(obj, f.name, field_values[f.name])
                            except AttributeError: pass
                    return obj
                except Exception:
                    return field_values
            
            elif isinstance(data, (list, tuple)):
                processed = [process_data(v) for v in data]
                try:
                    return type(data)(processed)
                except Exception:
                    return processed
            
            return data

        def get_hook(name: str):
            def hook(module, input, output):
                activations[name] = process_data(output)
            return hook

        name_to_module = dict(self.named_modules())
        try:
            for name in module_names:
                if name in name_to_module:
                    handle = name_to_module[name].register_forward_hook(get_hook(name))
                    handles.append(handle)
            yield activations
        finally:
            for handle in handles:
                handle.remove()

    def clip_grad_norm(self, max_norm: float, norm_type: float = 2.0) -> float:
        '''
        Clip the gradients of all trainable parameters.
        
        Args:
            max_norm (float): Max norm of the gradients.
            norm_type (float): Type of the used p-norm.
            
        Returns:
            float: Total norm of the parameters (viewed as a single vector).
        '''
        return torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm, norm_type=norm_type)

    def optimizer_groups(self, weight_decay: float = 1e-2) -> list[dict[str, Any]]:
        '''
        Separate parameters into decay and no_decay groups.
        Typically, biases and normalization layer parameters are not decayed.
        '''
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, 
                                    torch.nn.BatchNorm3d, torch.nn.GroupNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = f'{mn}.{pn}' if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        uncategorized = param_dict.keys() - (decay | no_decay)

        for fpn in uncategorized:
            if any(x in fpn.lower() for x in ['bias', 'norm', 'ln_', 'embed', 'scale', 'logit_scale']):
                no_decay.add(fpn)
            else:
                decay.add(fpn)
        
        inter_params = decay & no_decay
        assert len(inter_params) == 0, f'Parameters {str(inter_params)} made it into both decay/no_decay sets!'
        optim_groups = [
            {'params': [param_dict[pn] for pn in sorted(list(decay)) if param_dict[pn].requires_grad], 'weight_decay': weight_decay},
            {'params': [param_dict[pn] for pn in sorted(list(no_decay)) if param_dict[pn].requires_grad], 'weight_decay': 0.0},
        ]
        return optim_groups

    def tie_weights(self: TBasicModel, source_module_path: str, target_module_path: str) -> TBasicModel:
        '''
        Tie weights between two modules (e.g., embedding and lm_head).
        
        Args:
            source_module_path (str): Path to the source module (e.g., 'transformer.wte').
            target_module_path (str): Path to the target module (e.g., 'lm_head').
        '''
        modules = dict(self.named_modules())
        source_module = modules.get(source_module_path)
        target_module = modules.get(target_module_path)
        
        if source_module is None or target_module is None:
            raise ValueError(f"Modules '{source_module_path}' or '{target_module_path}' not found.")
        
        if hasattr(source_module, 'weight') and hasattr(target_module, 'weight'):
            target_module.weight = source_module.weight
            if hasattr(source_module, 'bias') and hasattr(target_module, 'bias') and target_module.bias is not None:
                target_module.bias = source_module.bias
        else:
            raise AttributeError("One of the modules does not have a 'weight' attribute to tie.")
        
        return self
    
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

    def to_precision(self: TBasicModel, dtype: torch.dtype) -> TBasicModel:
        '''
        Cast the model parameters to a specific precision (e.g., torch.float16, torch.bfloat16)
        and return self for method chaining.
        '''
        self.to(dtype=dtype)
        return self

    def to_device(self: TBasicModel, device: Union[str, torch.device]) -> TBasicModel:
        '''
        Move the model to a specific device and return self for method chaining.
        '''
        self.to(device=device)
        return self
    
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
    
    def freeze(
        self: TBasicModel, 
        exclude_modules: list[Union[type, nn.Module]] = None,
        only: list[str] = None,
        exclude: list[str] = None
    ) -> TBasicModel:
        '''
        Freeze parameters in the model by setting requires_grad to False.
        Allows fine-grained control via module exclusion and name filtering.

        Args:
            exclude_modules (list[Union[type, nn.Module]], optional): Module types or instances to exclude from freezing.
            only (list[str], optional): Only freeze parameters whose names contain any of these substrings.
            exclude (list[str], optional): Exclude parameters whose names contain any of these substrings.
        '''
        exclude_params = set()
        if exclude_modules:
            exclude_types = tuple(t for t in exclude_modules if isinstance(t, type))
            exclude_instances = set(m for m in exclude_modules if not isinstance(m, type))
            for _, module in self.named_modules():
                if module in exclude_instances or (exclude_types and isinstance(module, exclude_types)):
                    exclude_params.update(module.parameters())

        for name, param in self.named_parameters():
            if param in exclude_params:continue
            if exclude and any(kw in name for kw in exclude):continue
            if only and not any(kw in name for kw in only):continue
            param.requires_grad = False
        return self

    def unfreeze(
        self: TBasicModel, 
        exclude_modules: list[Union[type, nn.Module]] = None,
        only: list[str] = None,
        exclude: list[str] = None
    ) -> TBasicModel:
        '''
        Unfreeze parameters in the model by setting requires_grad to True.
        Allows fine-grained control via module exclusion and name filtering.

        Args:
            exclude_modules (list[Union[type, nn.Module]], optional): Module types or instances to exclude from unfreezing.
            only (list[str], optional): Only unfreeze parameters whose names contain any of these substrings.
            exclude (list[str], optional): Exclude parameters whose names contain any of these substrings.
        '''
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
            param.requires_grad = True
        return self

    @contextlib.contextmanager
    def frozen_context(
        self, 
        exclude_modules: list[Union[type, nn.Module]] = None,
        only: list[str] = None,
        exclude: list[str] = None
    ) -> Iterator[None]:
        '''
        A context manager that temporarily freezes specified parameters,
        restoring their original requires_grad states upon exit.
        '''
        target_params = []
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
            
            target_params.append((param, param.requires_grad))
            param.requires_grad = False
        
        try: yield
        finally:
            for param, original_state in target_params: param.requires_grad = original_state

    @contextlib.contextmanager
    def unfrozen_context(
        self, 
        exclude_modules: list[Union[type, nn.Module]] = None,
        only: list[str] = None,
        exclude: list[str] = None
    ) -> Iterator[None]:
        '''
        A context manager that temporarily unfreezes specified parameters,
        restoring their original requires_grad states upon exit.
        '''
        target_params = []
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
            
            target_params.append((param, param.requires_grad))
            param.requires_grad = True
        
        try: yield
        finally:
            for param, original_state in target_params: param.requires_grad = original_state
    
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
    
    def trigger(self, func_name: str, *args, **kwargs) -> None:
        '''
        Traverse all sub-modules (including this model). If a sub-module is an
        instance of BasicModel and has the attribute/method specified by func_name,
        trigger/call it once.

        Args:
            func_name (str): The name of the method to trigger.
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.
        '''
        for module in self.modules():
            if isinstance(module, BasicModel):
                func = getattr(module, func_name, None)
                if func is not None and callable(func): func(*args, **kwargs)

    def snapshot(self, name: str = 'default', device: str = 'cpu') -> None:
        '''
        Take an in-memory snapshot of the current model state (weights and buffers) 
        and store it on the specified device.
        '''
        state = self.state_dict()
        
        snapshot_state = {
            k: v.detach().clone().to(device=device) for k, v in state.items()
        }

        self._snapshots[name] = snapshot_state

    def restore_snapshot(self, name: str = 'default', strict: bool = True) -> None:
        '''
        Restore the model state from an in-memory snapshot.
        '''
        if name not in self._snapshots:
            raise KeyError(f"No snapshot found with name '{name}'")
            
        snapshot_state = self._snapshots[name]
        
        target_device = self.device
        restored_state = {
            k: v.to(device=target_device) for k, v in snapshot_state.items()
        }

        self.load_state_dict(restored_state, strict=strict)

    def clear_snapshots(self) -> None:
        '''Free memory occupied by snapshots.'''
        if hasattr(self, '_snapshots'):
            self._snapshots.clear()
    