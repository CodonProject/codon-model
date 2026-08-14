'''
Group-based optimizer and scheduler wrapper for PyTorch.

Provides a unified interface for managing multiple named optimizers and their
associated learning rate schedulers. Supports per-group optimizer selection
(Adam, SGD, etc.) and per-group scheduler selection (StepLR, CosineAnnealingLR, etc.).

Two convenient accessors are provided:
    - optim['group'] -> optimizer instance
    - sched['group'] -> scheduler instance (or None if not set)
'''

from codon import *
from codon.mixins.training import OptimizerGroups
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import types

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from .muon import Muon


class GroupOptimizer:
    '''
    Wrapper for one or more PyTorch optimizers with named group access,
    optionally with per-group learning rate schedulers.

    Attributes:
        optim (MappingProxyType): Read-only dict-like view of optimizers.
        sched (MappingProxyType): Read-only dict-like view of schedulers.
        names (List[str]): List of group names in insertion order.
    '''

    def __init__(
        self,
        groups: Dict[str, Dict[str, Any]],
        optimizer_class: Optional[Any] = None,
        defaults: Optional[Dict[str, Any]] = None,
        schedulers: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        '''
        Initialises the GroupOptimizer with optional per-group schedulers.

        Args:
            groups (Dict[str, Dict[str, Any]]): Dictionary mapping group names to group configs.
                Each group dict MUST contain 'params' (iterable of parameters).
                Optionally, it can contain 'optimizer' (an optimizer class) and any
                optimizer-specific arguments (lr, weight_decay, momentum, etc.).
                If 'optimizer' is not specified in a group, the global `optimizer_class` is used.
            optimizer_class (Optional[Any]): Default optimizer class for groups that do not specify one.
                If None, every group must specify its own 'optimizer'.
            defaults (Optional[Dict[str, Any]]): Default hyperparameters applied to all groups
                (overridden by group-specific values and **kwargs).
            schedulers (Optional[Dict[str, Dict[str, Any]]]): Optional dictionary mapping group names
                to scheduler configurations. Each scheduler config must contain:
                - 'scheduler': the scheduler class (e.g., torch.optim.lr_scheduler.StepLR)
                - Additional kwargs for the scheduler (e.g., 'step_size', 'gamma', 'T_max', etc.)
                Example:
                    {
                        'encoder': {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR, 'T_max': 100},
                        'decoder': {'scheduler': torch.optim.lr_scheduler.StepLR, 'step_size': 30, 'gamma': 0.5},
                    }
            **kwargs: Global optimizer arguments applied to all groups (e.g., weight_decay).
                These have the lowest priority, overridden by defaults and group configs.

        Raises:
            ValueError: If groups is empty, or a group is missing 'params', or an optimizer class is missing.
        '''
        if not groups:
            raise ValueError('groups cannot be empty')

        self.optimizers: Dict[str, Optimizer] = {}
        self.schedulers: Dict[str, _LRScheduler] = {}
        self.names: List[str] = list(groups.keys())

        # Merge default hierarchy: group > defaults > kwargs
        base_opts = defaults.copy() if defaults else {}
        base_opts.update(kwargs)

        for name, group in groups.items():
            # Pop mandatory params
            if 'params' not in group:
                raise ValueError(f'Group "{name}" missing required key "params"')
            params = group.pop('params')
            if isinstance(params, nn.Module):
                params = params.parameters()
            if isinstance(params, BasicModel):
                params = params.trainable_params

            # Determine optimizer class for this group
            opt_cls = group.pop('optimizer', optimizer_class)
            if opt_cls is None:
                raise ValueError(
                    f'Group "{name}" does not specify "optimizer", and no global optimizer_class provided.'
                )

            # Build final kwargs for this optimizer: base_opts + group-specific
            opt_kwargs = base_opts.copy()
            opt_kwargs.update(group)

            # Instantiate the optimizer
            self.optimizers[name] = opt_cls(params, **opt_kwargs)

        # Setup schedulers if provided
        if schedulers is not None:
            for name, sched_cfg in schedulers.items():
                if name not in self.optimizers:
                    raise ValueError(f'Scheduler group "{name}" does not match any optimizer group.')
                sched_cls = sched_cfg.pop('scheduler')
                if not hasattr(sched_cls, '__bases__') or _LRScheduler not in sched_cls.__bases__:
                    raise TypeError(f'"{sched_cls}" is not a valid scheduler class (must inherit _LRScheduler).')
                # Instantiate scheduler with the corresponding optimizer
                self.schedulers[name] = sched_cls(self.optimizers[name], **sched_cfg)

        # Provide dict-like read-only views (using MappingProxyType)
        self.optim = types.MappingProxyType(self.optimizers)
        self.sched = types.MappingProxyType(self.schedulers)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        '''
        Performs a single optimization step for ALL optimizers.

        Args:
            closure (Optional[Callable]): A closure that reevaluates the model and returns the loss.
                If provided, it is called for each optimizer sequentially (rarely used in practice).

        Returns:
            Optional[float]: The loss value from the last optimizer's closure, if closures are used.
        '''
        ret = None
        for opt in self.optimizers.values():
            ret = opt.step(closure)
        return ret

    def zero_grad(self, set_to_none: bool = True) -> None:
        '''
        Zeroes out gradients for ALL optimizers.

        Args:
            set_to_none (bool): If True, sets gradients to None instead of zeroing.
        '''
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        '''
        Returns the state of ALL optimizers and schedulers as a dict.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary with keys:
                - 'optimizers': mapping group names to optimizer state_dict.
                - 'schedulers': mapping group names to scheduler state_dict (if schedulers exist).
        '''
        state = {
            'optimizers': {name: opt.state_dict() for name, opt in self.optimizers.items()}
        }
        if self.schedulers:
            state['schedulers'] = {name: sched.state_dict() for name, sched in self.schedulers.items()}
        return state

    def load_state_dict(self, state_dict: Dict[str, Dict[str, Any]]) -> None:
        '''
        Loads the optimizer and scheduler states.

        Args:
            state_dict (Dict[str, Dict[str, Any]]): State dictionary produced by state_dict().
                May contain 'optimizers' and/or 'schedulers'.
        '''
        if 'optimizers' in state_dict:
            for name, opt_state in state_dict['optimizers'].items():
                if name in self.optimizers:
                    self.optimizers[name].load_state_dict(opt_state)
                else:
                    print(f'Warning: Skipping optimizer state for unknown group "{name}"')
        if 'schedulers' in state_dict:
            if not self.schedulers:
                print('Warning: State dict contains schedulers but no schedulers were created.')
            else:
                for name, sched_state in state_dict['schedulers'].items():
                    if name in self.schedulers:
                        self.schedulers[name].load_state_dict(sched_state)
                    else:
                        print(f'Warning: Skipping scheduler state for unknown group "{name}"')

    def get_lr(self, group_name: str) -> List[float]:
        '''Gets learning rate(s) for a specific group.'''
        opt = self.optimizers[group_name]
        lrs = [pg['lr'] for pg in opt.param_groups]
        return lrs

    def set_lr(self, group_name: str, lr: Union[float, List[float]]) -> None:
        '''Sets learning rate(s) for a specific group.'''
        opt = self.optimizers[group_name]
        num_groups = len(opt.param_groups)
        if isinstance(lr, (int, float)):
            for pg in opt.param_groups:
                pg['lr'] = lr
        else:
            if len(lr) != num_groups:
                raise ValueError(
                    f'Length of lr list ({len(lr)}) does not match number of param_groups ({num_groups}) for "{group_name}"'
                )
            for pg, val in zip(opt.param_groups, lr):
                pg['lr'] = val

    def scale_lr(self, group_name: str, scale: float) -> None:
        '''Scales learning rate(s) for a specific group.'''
        opt = self.optimizers[group_name]
        for pg in opt.param_groups:
            pg['lr'] *= scale

    def set_all_lr(self, lr: Union[float, Dict[str, float]]) -> None:
        '''Sets learning rates for all groups.'''
        if isinstance(lr, (int, float)):
            for name in self.optimizers:
                self.set_lr(name, lr)
        else:
            for name, val in lr.items():
                self.set_lr(name, val)

    def get_all_lr(self) -> Dict[str, Union[float, List[float]]]:
        '''Returns dictionary mapping group names to learning rates.'''
        return {name: self.get_lr(name) for name in self.optimizers}

    def step_schedulers(self, epoch: Optional[int] = None) -> None:
        '''
        Calls step() on all schedulers.

        Args:
            epoch (Optional[int]): Current epoch number to pass to schedulers that require it
                (e.g., ExponentialLR, MultiStepLR). If None, schedulers use their internal counter.
        '''
        for sched in self.schedulers.values():
            if epoch is None:
                sched.step()
            else:
                sched.step(epoch)

    def get_scheduler(self, group_name: str) -> Optional[_LRScheduler]:
        '''
        Returns the scheduler for a specific group, or None if not present.

        Args:
            group_name (str): Name of the group.

        Returns:
            Optional[_LRScheduler]: The scheduler instance.
        '''
        return self.schedulers.get(group_name)

    def get_optimizer(self, group_name: str) -> Optimizer:
        '''
        Returns the underlying optimizer for a specific group.

        Args:
            group_name (str): Name of the group.

        Returns:
            Optimizer: The underlying optimizer.
        '''
        return self.optimizers[group_name]

    def __repr__(self) -> str:
        sched_info = f', schedulers={list(self.schedulers.keys())}' if self.schedulers else ''
        return f'GroupOptimizer(groups={list(self.optimizers.keys())}{sched_info})'

    @staticmethod
    def _create_scheduler_from_config(optimizer: Optimizer, config: Dict[str, Any]) -> _LRScheduler:
        """
        根据配置字典创建调度器。

        Args:
            optimizer: 优化器实例。
            config: 调度器配置，支持两种格式：
                1. 普通调度器：必须包含 'scheduler' 键（调度器类），其余键作为参数。
                2. Warmup + Cosine 组合调度器：包含 'warmup_steps', 'total_steps', 'lr_min'。
                   可选 'start_factor'（默认 1e-8），'end_factor'（默认 1.0）。
                   将自动构建 SequentialLR(LinearLR, CosineAnnealingLR)。

        Returns:
            _LRScheduler: 调度器实例。

        Raises:
            ValueError: 如果配置不合法。
        """
        # 检查是否为 Warmup + Cosine 组合模式
        if 'warmup_steps' in config and 'total_steps' in config and 'lr_min' in config:
            warmup_steps = config['warmup_steps']
            total_steps = config['total_steps']
            lr_min = config['lr_min']
            start_factor = config.get('start_factor', 1e-8)
            end_factor = config.get('end_factor', 1.0)

            if warmup_steps <= 0 or total_steps <= warmup_steps:
                raise ValueError(
                    f"warmup_steps ({warmup_steps}) must be positive and less than total_steps ({total_steps})"
                )

            # 构建 Warmup 调度器 (LinearLR)
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=end_factor,
                total_iters=warmup_steps,
            )
            # 构建 Cosine 衰减调度器
            decay_steps = total_steps - warmup_steps
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, decay_steps),
                eta_min=lr_min,
            )
            # 组合为 SequentialLR
            return SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )

        # 否则为普通调度器
        if 'scheduler' not in config:
            raise ValueError("Scheduler config must contain either 'scheduler' key or 'warmup_steps/total_steps/lr_min'.")
        sched_cls = config.pop('scheduler')
        # 检查是否继承自 _LRScheduler
        if not hasattr(sched_cls, '__bases__') or _LRScheduler not in sched_cls.__bases__:
            raise TypeError(f'"{sched_cls}" is not a valid scheduler class (must inherit _LRScheduler).')
        return sched_cls(optimizer, **config)

    @staticmethod
    def build(
        opt_groups: OptimizerGroups,
        optimizer_mapping: Optional[Dict[str, Any]] = None,
        scheduler_mapping: Optional[Dict[str, Dict[str, Any]]] = None,
        unified_scheduler: Optional[Dict[str, Any]] = None,
        defaults: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'GroupOptimizer':
        """
        从 OptimizerGroups 构建 GroupOptimizer 实例。

        Args:
            opt_groups: 由 TrainingUtilsMixin.optimizer_groups() 返回的对象。
            optimizer_mapping: 映射组类型到优化器类。默认全部使用 AdamW。
            scheduler_mapping: 按组类型指定调度器配置。键为 'standard', 'adamw', 'muon'。
                值可为 None（不使用调度器）或字典（格式同 unified_scheduler）。
                如果某类型在此映射中指定，将覆盖 unified_scheduler。
            unified_scheduler: 统一调度器配置，作为所有组的默认配置。
                若某组类型未在 scheduler_mapping 中指定，则使用此配置。
                格式：普通调度器需包含 'scheduler' 键；Warmup+Cosine 组合需包含
                'warmup_steps', 'total_steps', 'lr_min'。
            defaults: 所有组的默认超参数（被组内同名键覆盖）。
            **kwargs: 全局优化器参数（最低优先级）。

        Returns:
            GroupOptimizer 实例。
        """
        from copy import deepcopy

        # 默认优化器映射
        if optimizer_mapping is None:
            optimizer_mapping = {'standard': torch.optim.AdamW, 'adamw': torch.optim.AdamW, 'muon': Muon}
        for key in ['standard', 'adamw', 'muon']:
            if key not in optimizer_mapping:
                optimizer_mapping[key] = torch.optim.AdamW

        # 构建 groups 字典
        groups = {}
        for prefix, group_list in [
            ('standard', opt_groups.standard),
            ('adamw', opt_groups.adamw),
            ('muon', opt_groups.muon),
        ]:
            opt_cls = optimizer_mapping.get(prefix)
            if opt_cls is None:
                continue
            for idx, pg in enumerate(group_list):
                group_cfg = deepcopy(pg)
                params = group_cfg.pop('params')
                group_name = f"{prefix}_{idx}"
                groups[group_name] = {
                    'params': params,
                    'optimizer': opt_cls,
                    **group_cfg,
                }

        if not groups:
            raise ValueError("No parameter groups found; all opt_groups lists are empty or optimizer_mapping is None.")

        # 构建 schedulers 字典
        schedulers = {}
        # 预处理：统一调度器必须有效（若有）
        if unified_scheduler is not None:
            # 不做完整校验，由 _create_scheduler_from_config 在执行时校验
            pass

        for prefix, group_list in [
            ('standard', opt_groups.standard),
            ('adamw', opt_groups.adamw),
            ('muon', opt_groups.muon),
        ]:
            # 获取该类型的特定调度器配置
            prefix_sched_cfg = scheduler_mapping.get(prefix) if scheduler_mapping else None
            # 决定使用哪种配置：优先使用 scheduler_mapping 中的，否则使用 unified_scheduler
            if prefix_sched_cfg is None:
                if unified_scheduler is None:
                    continue  # 该类型不使用调度器
                else:
                    cfg = deepcopy(unified_scheduler)
            else:
                # 如果明确为 None，表示禁用该类型的调度器
                if prefix_sched_cfg is None:
                    continue
                cfg = deepcopy(prefix_sched_cfg)

            # 为该类型下的所有子组添加相同的调度器配置
            for idx in range(len(group_list)):
                group_name = f"{prefix}_{idx}"
                schedulers[group_name] = cfg  # 保存配置，稍后实例化

        wrapper = GroupOptimizer(
            groups=groups,
            schedulers=None,  # 稍后设置
            defaults=defaults,
            **kwargs
        )

        # 现在，为每个组创建调度器
        for group_name, cfg in schedulers.items():
            if group_name not in wrapper.optimizers:
                continue
            optimizer = wrapper.optimizers[group_name]
            try:
                sched = GroupOptimizer._create_scheduler_from_config(optimizer, cfg)
                wrapper.schedulers[group_name] = sched
            except Exception as e:
                raise RuntimeError(f"Failed to create scheduler for group '{group_name}': {e}")

        # 更新只读视图（因为我们修改了 wrapper.schedulers，需重新创建 MappingProxyType）
        wrapper.sched = types.MappingProxyType(wrapper.schedulers)

        return wrapper