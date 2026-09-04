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


# 内置 LoRA 目标族：target_modules 传族名即可展开成 (types, include, exclude)
_LORA_FAMILIES: Dict[str, Tuple[list, list, list]] = {
    # 全部 nn.Linear（不含额外的 exclude 时即旧行为）
    'all-linear': ([nn.Linear], [], []),
    # attention 内的投影线性层
    'attn': ([nn.Linear], ['.attn.'], []),
    # MoE 前馈（expert + shared expert，默认不含 router moe.gate）
    'mlp': ([nn.Linear], ['.moe.'], ['moe.gate']),
    # 仅各专家 / 共享专家的 MLP
    'expert': ([nn.Linear], ['.moe.experts.', '.moe.shared_experts'], []),
    # attention 各投影 + 相关压缩投影
    'qkv': ([nn.Linear], ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'g_proj',
                          'q_a_proj', 'q_b_proj', 'kv_a_proj', 'kv_b_proj', 'k_p_proj'], []),
}


def inject(
    model: nn.Module,
    target_modules: Union[str, Type[nn.Module], List[Union[str, Type[nn.Module]]]] = None,
    *,
    include: Optional[list] = None,
    exclude: Optional[list] = None,
    module_exclude: Optional[list] = None,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    merge_weights: bool = False,
    gate: bool = False,
    dora: bool = False,
    gradient_checkpointing: bool = False
) -> nn.Module:
    '''
    向模型注入 LoRA。

    target_modules:
        - 族名（'all-linear' / 'attn' / 'mlp' / 'expert' / 'qkv'）见 _LORA_FAMILIES；
        - nn.Module 类型或具体名字（旧语义，ends_with/== 匹配）；
        - 可混成 list。
      缺省 = 'all-linear'。

    include    : list[str]，模块路径须含任一片段才注入（与类型/族叠加）。
    exclude    : list[str]，模块路径含任一片段则跳过。
    module_exclude: list[str | type | nn.Module]，按路径子串/类型/实例排除整模块。
    '''
    orig = getattr(model, 'original_model', model)

    if target_modules is None:
        target_modules = ['all-linear']
    if isinstance(target_modules, (str, type)):
        target_modules = [target_modules]

    fam_targets: list = []
    fam_include: list = []
    fam_exclude: list = []
    for t in target_modules:
        if isinstance(t, str) and t in _LORA_FAMILIES:
            tt, ti, te = _LORA_FAMILIES[t]
            fam_targets += tt; fam_include += ti; fam_exclude += te
        else:
            fam_targets.append(t)

    inc = list(include or []) + fam_include
    exc = list(exclude or []) + fam_exclude

    me_types: tuple = tuple(t for t in (module_exclude or []) if isinstance(t, type))
    me_inst: set = set(t for t in (module_exclude or []) if not isinstance(t, type) and not isinstance(t, str))
    for t in (module_exclude or []):
        if isinstance(t, str):
            exc.append(t)

    str_targets = [t for t in fam_targets if isinstance(t, str)]
    type_targets = [t for t in fam_targets if isinstance(t, type)]

    targets_to_replace = []
    for name, module in orig.named_modules():
        if name == '': continue

        module_orig = getattr(module, 'original_model', module)
        if me_inst and (module_orig in me_inst or module in me_inst): continue
        if me_types and isinstance(module_orig, me_types): continue
        if exc and any(e in name for e in exc): continue

        is_target = False
        if str_targets and any(name.endswith(t) or name == t for t in str_targets):
            is_target = True
        elif type_targets and any(isinstance(module_orig, t) for t in type_targets):
            is_target = True

        if not is_target: continue
        if inc and not any(i in name for i in inc): continue
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


def _iter_lora(orig: nn.Module):
    from codon.block.lora import BasicLoRA
    for module in orig.modules():
        if isinstance(module, BasicLoRA):
            yield module


def has_lora(model: nn.Module) -> bool:
    '''模型是否已注入 LoRA（存在 BasicLoRA wrapper）。'''
    orig = getattr(model, 'original_model', model)
    for _ in _iter_lora(orig):
        return True
    return False


def count_lora(model: nn.Module) -> Dict[str, Any]:
    '''统计注入情况：注入层数 / 可训练参数 / 总参数量 / rank 分布。'''
    orig = getattr(model, 'original_model', model)
    modules = list(_iter_lora(orig))
    ranks: set = set()
    gated = dora = 0
    for m in modules:
        if getattr(m, 'r', 0) > 0: ranks.add(getattr(m, 'r'))
        if getattr(m, 'gate', False): gated += 1
        if getattr(m, 'dora', False): dora += 1

    trainable_params = sum(p.numel() for p in orig.parameters() if p.requires_grad)
    lora_params = sum(
        p.numel() for m in modules
        for _, p in m.named_parameters(recurse=False)
        if any(kw in _ for kw in ['lora_', 'dora_'])
    )
    return {
        'injected_modules': len(modules),
        'ranks': sorted(ranks),
        'gated': gated,
        'dora': dora,
        'trainable_params': trainable_params,
        'lora_params': lora_params,
    }