from codon import *
from codon.block.lora import *

import os
from safetensors.torch import save_file, load_file
import contextlib
from typing import Iterator


@contextlib.contextmanager
def disable_lora(model: nn.Module) -> Iterator[None]:
    orig = getattr(model, 'original_model', model)
    original_scalings = {}
    unmerged_modules = []
    
    try:
        for name, module in orig.named_modules():
            module_orig = getattr(module, 'original_model', module)
            if isinstance(module_orig, BasicLoRA):
                if module_orig.merged:
                    module_orig.unmerge()
                    unmerged_modules.append(module_orig)
                
                original_scalings[name] = module_orig.scaling
                module_orig.scaling = 0.0
                
        yield
        
    finally:
        for name, module in orig.named_modules():
            module_orig = getattr(module, 'original_model', module)
            if isinstance(module_orig, BasicLoRA) and name in original_scalings:
                module_orig.scaling = original_scalings[name]
                
        for module in unmerged_modules:
            module.merge()
    
def _get_submodule(model: nn.Module, target_path: str) -> Tuple[nn.Module, str, nn.Module]:
    parent = model
    path_parts = target_path.split('.')
    for part in path_parts[:-1]:
        parent = getattr(parent, part)
    child_name = path_parts[-1]
    child = getattr(parent, child_name)
    return parent, child_name, child


def inject(
    model: nn.Module,
    target_modules: Union[str, Type[nn.Module], List[Union[str, Type[nn.Module]]]],
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    merge_weights: bool = False,
    gate: bool = False,
    dora: bool = False,
    gradient_checkpointing: bool = False
) -> nn.Module:
    orig = getattr(model, 'original_model', model)
    targets_to_replace = []
    
    if isinstance(target_modules, (str, type)):
        target_modules = [target_modules]
        
    str_targets = [t for t in target_modules if isinstance(t, str)]
    type_targets = [t for t in target_modules if isinstance(t, type)]
    
    for name, module in orig.named_modules():
        if name == '': continue
        
        module_orig = getattr(module, 'original_model', module)
        is_target = False
        
        if str_targets and any(name.endswith(t) or name == t for t in str_targets):
            is_target = True
            
        elif type_targets and any(isinstance(module_orig, t) for t in type_targets):
            is_target = True
                
        if is_target:
            targets_to_replace.append(name)

    for target_path in targets_to_replace:
        parent, child_name, child = _get_submodule(orig, target_path)
        child_orig = getattr(child, 'original_model', child)
        
        lora_wrapper = None
        kwargs = dict(
            original_layer=child_orig, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            merge_weights=merge_weights, gate=gate, dora=dora, gradient_checkpointing=gradient_checkpointing
        )
        
        if isinstance(child_orig, nn.Linear):
            lora_wrapper = LinearLoRA(**kwargs)
        elif isinstance(child_orig, nn.Conv2d):
            lora_wrapper = Conv2dLoRA(**kwargs)
        elif isinstance(child_orig, nn.Conv1d):
            lora_wrapper = Conv1dLoRA(**kwargs)
        elif isinstance(child_orig, nn.Embedding):
            lora_wrapper = EmbeddingLoRA(**kwargs)
        else:
            print(f"Warning: Skipping {target_path}, unsupported module type {type(child_orig)}")
            continue
            
        setattr(parent, child_name, lora_wrapper)

    return model


def freeze_backbone(model: nn.Module) -> nn.Module:
    orig = getattr(model, 'original_model', model)
    for param in orig.parameters():
        param.requires_grad = False
        
    for module in orig.modules():
        module_orig = getattr(module, 'original_model', module)
        if isinstance(module_orig, BasicLoRA):
            for name, param in module_orig.named_parameters(recurse=False):
                if any(kw in name for kw in ['lora_', 'dora_']):
                    param.requires_grad = True
                    
    return model


def save_lora(model: nn.Module, path: str) -> None:
    orig = getattr(model, 'original_model', model)
    state_dict = orig.state_dict()
    lora_state_dict = {
        k: v for k, v in state_dict.items() 
        if ('lora_' in k or 'dora_' in k) and 'original_layer' not in k
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    if path.endswith('.safetensors'):
        save_file(lora_state_dict, path)
    else:
        torch.save(lora_state_dict, path)
        

def inject_from_file(
    model: nn.Module, 
    path: str,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    merge_weights: bool = False,
    gradient_checkpointing: bool = False
) -> nn.Module:
    orig = getattr(model, 'original_model', model)
    
    if path.endswith('.safetensors'):
        state_dict = load_file(path)
    else:
        state_dict = torch.load(path, map_location='cpu')

    lora_configs = {}
    for key, tensor in state_dict.items():
        if '.lora_a' in key:
            prefix = key.split('.lora_a')[0]
            
            r = tensor.shape[0] if not key.endswith('lora_a.weight') or len(tensor.shape) > 2 else tensor.shape[1] 
            
            if 'lora_a.weight' not in key and tensor.shape[0] > tensor.shape[1]: 
                r = tensor.shape[1]
                
            lora_configs[prefix] = {
                'r': r, 
                'gate': False, 
                'dora': False
            }

    for key in state_dict.keys():
        if '.lora_gate' in key:
            prefix = key.split('.lora_gate')[0]
            if prefix in lora_configs: lora_configs[prefix]['gate'] = True
        elif '.dora_m' in key:
            prefix = key.split('.dora_m')[0]
            if prefix in lora_configs: lora_configs[prefix]['dora'] = True

    for target_path, config in lora_configs.items():
        try:
            parent, child_name, child = _get_submodule(orig, target_path)
            child_orig = getattr(child, 'original_model', child)
            
            kwargs = dict(
                original_layer=child_orig, 
                r=config['r'], 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
                merge_weights=False,
                gate=config['gate'], 
                dora=config['dora'], 
                gradient_checkpointing=gradient_checkpointing
            )
            
            if isinstance(child_orig, nn.Linear): lora_wrapper = LinearLoRA(**kwargs)
            elif isinstance(child_orig, nn.Conv2d): lora_wrapper = Conv2dLoRA(**kwargs)
            elif isinstance(child_orig, nn.Conv1d): lora_wrapper = Conv1dLoRA(**kwargs)
            elif isinstance(child_orig, nn.Embedding): lora_wrapper = EmbeddingLoRA(**kwargs)
            else: continue
            
            setattr(parent, child_name, lora_wrapper)
            
        except AttributeError:
            print(f"Warning: Could not find module {target_path} in the model.")

    orig.load_state_dict(state_dict, strict=False)

    if merge_weights:
        for module in orig.modules():
            module_orig = getattr(module, 'original_model', module)
            if isinstance(module_orig, BasicLoRA):
                module_orig.merge()

    return model

def merge_all(model: nn.Module) -> nn.Module:
    orig = getattr(model, 'original_model', model)
    count = 0
    for module in orig.modules():
        module_orig = getattr(module, 'original_model', module)
        if isinstance(module_orig, BasicLoRA):
            module_orig.merge()
            count += 1
    print(f"Successfully merged {count} LoRA modules.")
    return model

def unmerge_all(model: nn.Module) -> nn.Module:
    orig = getattr(model, 'original_model', model)
    count = 0
    for module in orig.modules():
        module_orig = getattr(module, 'original_model', module)
        if isinstance(module_orig, BasicLoRA):
            module_orig.unmerge()
            count += 1
    print(f"Successfully unmerged {count} LoRA modules.")
    return model

def cast_lora_precision(model: nn.Module, dtype: torch.dtype = torch.float32) -> nn.Module:
    orig = getattr(model, 'original_model', model)
    for module in orig.modules():
        module_orig = getattr(module, 'original_model', module)
        if isinstance(module_orig, BasicLoRA):
            if hasattr(module_orig, 'lora_a') and module_orig.lora_a is not None:
                module_orig.lora_a.to(dtype)
            if hasattr(module_orig, 'lora_b') and module_orig.lora_b is not None:
                module_orig.lora_b.to(dtype)
            if hasattr(module_orig, 'lora_gate') and module_orig.lora_gate is not None:
                module_orig.lora_gate.to(dtype)
            if hasattr(module_orig, 'dora_m') and module_orig.dora_m is not None:
                module_orig.dora_m.to(dtype)
    return model

def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    orig = getattr(model, 'original_model', model)
    state_dict = orig.state_dict()  # 修复了缺少圆括号的 Bug
    lora_state_dict = {
        k: v for k, v in state_dict.items() 
        if any(kw in k for kw in ['lora_', 'dora_']) 
        and 'original_layer' not in k 
        and 'weight_backup' not in k
    }
    return lora_state_dict