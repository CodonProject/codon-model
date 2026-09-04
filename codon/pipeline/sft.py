from codon import *
from codon.config import field, configclass
from codon.pipeline.base import BasicPipeline, PipelinePhase
from codon.pipeline.callback import Callback
from codon.model.types.language import CausalLanguageModel, CausalLanguageModelOutput
from codon.utils.tokens import PackedTokenizer

from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

# LoRA 注入目标的声明类型：族名 / nn.Module 类型 / 具体模块名（可混合列表）
_LoraTarget = Union[str, type, List[Union[str, type]]]

# =========================================================================
# 一、配置
# =========================================================================

@configclass
class LoRAConfig:
    '''SFTPipeline 内化的 LoRA 注入配置。enabled 时由 setup() 自动注入并冻结主干。'''
    enabled: bool = field(default=True)
    target: str = field(default='all-linear')                     # 'all-linear' | 'attn' | 'mlp' | 'expert' | 'qkv'
    target_modules: Optional[_LoraTarget] = field(default=None)   # 显式覆盖 target：类型/名字/族名（可 list）
    include: Optional[List[str]] = field(default=None)            # 模块路径须含任一片段才注入
    exclude: Optional[List[str]] = field(default_factory=lambda: ['proj_out', 'moe.gate'])  # 命中片段则跳过
    module_exclude: Optional[List[Any]] = field(default=None)     # 按路径子串 / 类型 / nn.Module 实例排除整模块
    r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    gate: bool = field(default=False)
    dora: bool = field(default=False)
    gradient_checkpointing: bool = field(default=False)
    lr_multiplier: float = field(default=1.0)        # LoRA 参数学习率倍率（>1 加速适配，可 <1）
    weight_decay: Optional[float] = field(default=None)  # None=沿用 SFTConfig.weight_decay
    save_merged: bool = field(default=False)         # stage/final 额外导出 merged 全量（adapter 之外）


@configclass
class SFTConfig:
    compiled: bool = field(default=False)           # SFT 参考脚本未 compile，默认关

    # ---- 优化器 ----
    learning_rate: float = field(default=5e-5)
    weight_decay: float = field(default=0.01)
    warmup_steps: int = field(default=100)
    lr_min_ratio: float = field(default=0.1)        # 余弦终点 = lr_min_ratio * peak
    grad_clip_norm: float = field(default=1.0)

    # ---- checkpoint（内化）----
    ckpt_dir: str = field(default='./sft_ckpt')
    save_every_steps: int = field(default=2000)
    keep_last: int = field(default=3)

    # ---- 进度条 ----
    use_progress: bool = field(default=True)

    # ---- 聊天探针（run_chat_turn）----
    probe_prompts: List[str] = field(default_factory=lambda: [
        '用一句话解释什么是注意力机制。',
        'Translate to English: 今天天气真好。',
        'Write a haiku about the ocean.',
    ])
    probe_every_steps: int = field(default=2000)
    probe_max_new_tokens: int = field(default=128)
    probe_temperature: float = field(default=0.8)

    # ---- LoRA 增强适配 ----
    # True=启用默认 LoRAConfig；False/None=普通全参 SFT；亦可传 LoRAConfig(...) 自定义。
    lora: Optional[Union[LoRAConfig, bool]] = field(default=None)
    aux_weight: float = field(default=1.0)             # loss = CE + aux_weight * aux_loss（1.0 保持原行为）

    # ---- 数据集声明（stage 内化）：不传 stages 时 pipeline 据此自动 build_sft_stages ----
    stage_specs: List[Dict[str, Any]] = field(default_factory=list)  # [{name, folder, epochs, ckpt}, ...]
    pad_length: int = field(default=2048)
    batch_size: int = field(default=8)
    dataset_cls: Optional[type] = field(default=None)                # None=CodonSFT（默认）；显式传 MotifSFT 回旧行为
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)     # 透传数据集类：two_turn_prob/three_turn_prob/system_prompts/pattern/recursive/seed


def _coerce_lora_config(lora: Any) -> Optional[LoRAConfig]:
    '''把 SFTConfig.lora（bool / LoRAConfig / None）归一化为 Optional[LoRAConfig]。'''
    if lora is None or lora is False:
        return None
    if lora is True:
        return LoRAConfig()          # enabled 默认 True
    if isinstance(lora, LoRAConfig):
        return lora
    raise TypeError(
        f'SFTConfig.lora 需为 bool 或 LoRAConfig 实例，got {type(lora).__name__}')

# =========================================================================
# 二、SFT 阶段
# =========================================================================

@dataclass
class SFTStage:
    name: str
    dataset: Any                    # MotifSFT（__getitem__ 返回预批处理 dict）
    epochs: int = 1
    ckpt: Optional[str] = None      # 阶段结束 save_pretrained 到该路径


def build_sft_stages(stage_specs, tokenizer, pad_length, batch_size, dataset_cls=None, **ds_kwargs):
    '''stage_specs: [{name, folder, epochs, ckpt}, ...] -> List[SFTStage]
    dataset_cls: 数据集类，默认 codon.utils.data.sft.CodonSFT（自动识别 MotifSFT 行 /
    session / messages / parquet 等混合格式）；显式传 codon.motif.data.MotifSFT 回到旧行为。'''
    from codon.utils.data.sft import CodonSFT
    cls = dataset_cls or CodonSFT
    stages = []
    for s in stage_specs:
        ds = cls(
            folder=s['folder'],
            tokenizer=tokenizer,
            pad_length=pad_length,
            batch_size=batch_size,
            **ds_kwargs,
        )
        stages.append(SFTStage(
            name=s['name'],
            dataset=ds,
            epochs=s.get('epochs', 1),
            ckpt=s.get('ckpt'),
        ))
    return stages


# =========================================================================
# 三、内置回调：进度条 / 自动 checkpoint+阶段保存 / 聊天探针
# =========================================================================

class _ProgressBar(Callback):
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.pbar = None

    def on_train_start(self, pipeline):
        self.pbar = tqdm(
            total=pipeline.total_steps,              # 总优化步数
            initial=pipeline.state.global_step,      # 续训后从恢复步数继续
            desc='SFT Training',
            dynamic_ncols=True,
        )

    def on_step_end(self, pipeline, metrics):
        if self.pbar is None:
            return
        self.pbar.set_postfix({
            'Stage': getattr(pipeline.current_stage, 'name', '?'),
            'Loss':  f"{metrics.get('loss/train', 0.0):.4f}",
            'LR':    f"{metrics.get('lr', 0.0):.2e}",
        })
        self.pbar.update(1)

    def on_train_end(self, pipeline):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class _AutoCheckpoint(Callback):
    '''周期 last.pt + 阶段结束 save_pretrained + 正常 final + 中断/异常紧急保存。'''

    def __init__(self, pipeline, directory, every_steps, keep_last=3):
        self.pipeline = pipeline
        self.dir = directory
        self.every = every_steps
        self.keep = keep_last
        self._ckpts: List[str] = []
        self._stage_epoch_count = 0
        os.makedirs(directory, exist_ok=True)

    def _save(self, tag):
        path = os.path.join(self.dir, f'{tag}.pt')
        self.pipeline.save_checkpoint(path)          # 基类原子写，含全部可续训状态
        tqdm.write(f'[*] Checkpoint saved to {path}')
        if tag not in ('last', 'interrupted', 'emergency'):
            self._ckpts.append(path)
            while len(self._ckpts) > self.keep:
                victim = self._ckpts.pop(0)
                try:
                    os.remove(victim)
                except OSError:
                    pass

    def _save_stage_ckpt(self, pipeline):
        '''阶段全部 epoch 跑完 -> 输出该阶段权重（对应参考脚本的 save_pretrained）。'''
        stage = pipeline.current_stage
        if stage is None or not stage.ckpt:
            return
        try:
            pipeline.save_stage(stage)
            tqdm.write(f'[*] Stage [{stage.name}] saved -> {stage.ckpt}')
        except Exception as e:
            tqdm.write(f'[!] Failed to save stage ckpt {stage.ckpt}: {e}')

    def on_epoch_end(self, pipeline):
        stage = pipeline.current_stage
        if stage is None:
            return
        self._stage_epoch_count += 1
        if self._stage_epoch_count >= stage.epochs:
            self._save_stage_ckpt(pipeline)
            self._stage_epoch_count = 0

    def on_step_end(self, pipeline, metrics):
        if self.every and pipeline.state.global_step % self.every == 0:
            self._save('last')

    def on_interrupt(self, pipeline):                # Ctrl+C / SIGINT
        self._save('interrupted')
        self._save('last')

    def on_train_end(self, pipeline):
        if pipeline.state.phase == PipelinePhase.FINISHED:
            self._save('final')
            self._save('last')
        else:
            tqdm.write('[!] Training ended abnormally — saving emergency checkpoint')
            self._save('emergency')
            self._save('last')


class _ChatProbe(Callback):
    '''按 probe_prompts 逐个跑 run_chat_turn（对应参考脚本的 Chat Probe）。'''

    def __init__(self, pipeline, prompts, every_steps, max_new_tokens, temperature):
        self.pipeline = pipeline
        self.prompts = list(prompts) if prompts else []
        self.every = every_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _run(self):
        if not self.prompts:
            return
        from codon.kit.train import run_chat_turn
        step = self.pipeline.state.global_step
        tqdm.write(f'\n===== [Chat Probe @ step {step}] =====')
        for prompt in self.prompts:
            try:
                run_chat_turn(
                    self.pipeline.model,
                    self.pipeline._tokenizer,
                    self.pipeline.device,
                    step=step,
                    user_prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
            except Exception as e:
                tqdm.write(f'[!] Chat probe failed for {prompt!r}: {e}')
        tqdm.write('=' * 56)

    def on_train_start(self, pipeline):
        self._run()                                   # 训练开始（含续训后）先跑一轮

    def on_step_end(self, pipeline, metrics):
        if self.every and pipeline.state.global_step % self.every == 0:
            self._run()


# =========================================================================
# 四、Pipeline 本体
# =========================================================================

class SFTPipeline(BasicPipeline):
    def __init__(
        self,
        model: CausalLanguageModel,
        tokenizer: PackedTokenizer,
        stages: Optional[List[SFTStage]] = None,
        config: SFTConfig = None,
        device=None,
        callbacks=None,
        seed=None,
    ):
        # stage 内化：config 缺省时可从 SFTConfig.stage_specs/pad_length/batch_size 声明构建，
        # 保持旧调用（显式传 stages + config）完全兼容。
        if config is None:
            raise ValueError('SFTPipeline 需要 config=SFTConfig(...)。')
        device = model.device if not device else device
        super().__init__(device, callbacks, seed)
        self.callbacks = list(self.callbacks)

        self._model = model
        self._model_compiled: Optional[CausalLanguageModel] = (
            torch.compile(self._model, dynamic=True) if config.compiled else None
        )
        self._tokenizer = tokenizer
        self._config = config
        self._stages: List[SFTStage] = (
            list(stages) if stages is not None else self._build_stages_from_config()
        )
        self._current_stage: Optional[SFTStage] = None

        # ---- LoRA 适配状态（lora 支持 True/False/LoRAConfig 归一化）----
        self._lora_cfg: Optional[LoRAConfig] = _coerce_lora_config(config.lora)
        self._lora_active: bool = bool(self._lora_cfg is not None and self._lora_cfg.enabled)
        self._aux_weight: float = config.aux_weight if config is not None else 1.0

        self._total_steps: int = 0
        self._optimizer = None
        self._scheduler = None

        # 续训：base 的 load_checkpoint 在 setup() 之前调用，先缓存载荷
        self._pending_payload: Optional[Dict[str, Any]] = None

        # ---- 注入内置回调 ----
        self._progress_cb = None
        self._ckpt_cb = None
        self._probe_cb = None
        if config.use_progress:
            self._progress_cb = _ProgressBar(pipeline=self)
            self.callbacks.append(self._progress_cb)
        if config.save_every_steps:
            self._ckpt_cb = _AutoCheckpoint(
                pipeline=self, directory=config.ckpt_dir,
                every_steps=config.save_every_steps, keep_last=config.keep_last,
            )
            self.callbacks.append(self._ckpt_cb)
        if config.probe_every_steps:
            self._probe_cb = _ChatProbe(
                pipeline=self, prompts=config.probe_prompts,
                every_steps=config.probe_every_steps,
                max_new_tokens=config.probe_max_new_tokens,
                temperature=config.probe_temperature,
            )
            self.callbacks.append(self._probe_cb)

    # ------------------------------------------------------------ stage 内化
    def _build_stages_from_config(self) -> List[SFTStage]:
        '''按 SFTConfig.stage_specs / pad_length / batch_size / dataset_kwargs 自动构建数据集阶段。'''
        cfg = self._config
        specs = list(cfg.stage_specs or [])
        if not specs:
            raise ValueError(
                'SFTPipeline 需要显式 stages=... 或在 SFTConfig 声明 stage_specs '
                '（形如 [{"name","folder","epochs","ckpt"}]）以自动构建数据集。')
        ds_kwargs = dict(cfg.dataset_kwargs or {})
        return build_sft_stages(
            specs, self._tokenizer,
            pad_length=cfg.pad_length, batch_size=cfg.batch_size,
            dataset_cls=cfg.dataset_cls, **ds_kwargs,
        )

    # ------------------------------------------------------------ 属性
    @property
    def model(self) -> CausalLanguageModel:
        return self._model_compiled if self._model_compiled is not None else self._model

    @property
    def current_stage(self) -> Optional[SFTStage]:
        return self._current_stage

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def current_lr(self) -> float:
        return self._scheduler.get_last_lr()[0] if self._scheduler is not None else 0.0

    def optimizers(self):
        return {'main': self._optimizer} if self._optimizer is not None else {}

    # ------------------------------------------------------------ setup
    def _assemble_lora(self) -> None:
        '''按 LoRAConfig 注入 LoRA 并冻结主干。幂等：已注入（外部手配）则只补 freeze + 一致性提示。'''
        from codon.utils.lora import has_lora, count_lora
        cfg = self._lora_cfg
        if not has_lora(self._model):
            # 基础权重加载属模型装配职责（脚本在 pipeline 外完成），这里只负责注入
            self._model = self._model.to(self.device)
            tm = cfg.target_modules if cfg.target_modules is not None else cfg.target
            self._model.inject_lora(
                tm,
                include=cfg.include,
                exclude=cfg.exclude,
                module_exclude=cfg.module_exclude,
                r=cfg.r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                gate=cfg.gate,
                dora=cfg.dora,
                gradient_checkpointing=cfg.gradient_checkpointing,
            )
        self._model.freeze_backbone()   # 幂等：冻结主干、放开 lora_/dora_ 参数
        stats = count_lora(self._model)
        print(f'[*] LoRA 装配: {stats["injected_modules"]} 层, '
              f'trainable {stats["trainable_params"]:,} params '
              f'(lora {stats["lora_params"]:,}), rank={stats["ranks"]}')

    def setup(self):
        if self._lora_active:
            self._assemble_lora()
        self._model = self._model.to(self.device)
        # 总步数 = 各阶段 len(dataset) * epochs（构建调度器需要）
        self._total_steps = sum(len(s.dataset) * s.epochs for s in self._stages)
        self._build_optimizer_and_scheduler()

        if self._pending_payload is not None:
            self._apply_payload(self._pending_payload)
            self._pending_payload = None

        # 进程级兜底保存（SIGTERM / kill / 崩溃后的 atexit）
        try:
            from codon.utils.lifecycle import exit_manager
            exit_manager.register(self._exit_safety_save)
        except Exception:
            pass

    def _build_optimizer_and_scheduler(self):
        cfg = self._config
        total = self._total_steps
        warmup = max(1, min(cfg.warmup_steps, total - 1)) if total > 1 else 1
        eta_min = cfg.lr_min_ratio * cfg.learning_rate

        # bias / LayerNorm 等 1 维参数不 weight decay；LoRA 参数独立 lr_multiplier / weight_decay
        lora_mult = self._lora_cfg.lr_multiplier if self._lora_active else 1.0
        lora_wd = cfg.weight_decay
        if self._lora_active and self._lora_cfg.weight_decay is not None:
            lora_wd = self._lora_cfg.weight_decay

        decay_main, decay_lora, no_decay_main, no_decay_lora = [], [], [], []
        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                continue
            is_lora = ('lora_' in name or 'dora_' in name)
            one_d = (param.ndim <= 1 or 'norm' in name.lower())
            bucket = (no_decay_lora if is_lora else no_decay_main) if one_d else (decay_lora if is_lora else decay_main)
            bucket.append(param)

        param_groups = []
        if decay_main:
            param_groups.append({'params': decay_main, 'weight_decay': cfg.weight_decay, 'lr': cfg.learning_rate})
        if decay_lora:
            param_groups.append({'params': decay_lora, 'weight_decay': lora_wd, 'lr': cfg.learning_rate * lora_mult})
        if no_decay_main:
            param_groups.append({'params': no_decay_main, 'weight_decay': 0.0, 'lr': cfg.learning_rate})
        if no_decay_lora:
            param_groups.append({'params': no_decay_lora, 'weight_decay': 0.0, 'lr': cfg.learning_rate * lora_mult})
        if not param_groups:
            raise RuntimeError('没有任何可训练参数：主干被冻结且未注入 LoRA。请检查 LoRAConfig / 模型冻结状态。')

        self._optimizer = torch.optim.AdamW(param_groups, lr=cfg.learning_rate)

        # 与参考脚本一致的 warmup -> cosine（跨阶段统一调度）
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            self._optimizer,
            start_factor=1e-8 / cfg.learning_rate,
            end_factor=1.0,
            total_iters=warmup,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer,
            T_max=max(1, total - warmup),
            eta_min=eta_min,
        )
        self._scheduler = torch.optim.lr_scheduler.SequentialLR(
            self._optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup],
        )

    # ------------------------------------------------------------ 数据 / epoch
    @staticmethod
    def _epoch_to_stage_pos(epoch: int, stages: List[SFTStage]):
        '''全局 epoch 序号 -> (stage_idx, epoch_idx)。用于断点续训后跳过已跑阶段。'''
        acc = 0
        for si, stage in enumerate(stages):
            for ei in range(stage.epochs):
                if acc == epoch:
                    return si, ei
                acc += 1
        return len(stages) - 1, stages[-1].epochs

    def iterate_epochs(self, dataset):
        '''逐个 stage、逐个 epoch 地 yield 该 stage 的 dataset（每 item 即一个 batch）。'''
        start_si, start_ei = self._epoch_to_stage_pos(self.state.epoch, self._stages)
        for si in range(start_si, len(self._stages)):
            stage = self._stages[si]
            start = start_ei if si == start_si else 0
            for _ in range(start, stage.epochs):
                self._current_stage = stage
                yield stage.dataset
        # 若 num_epochs 超过总阶段 epoch 数，从头再来（无限循环）
        while True:
            for stage in self._stages:
                for _ in range(stage.epochs):
                    self._current_stage = stage
                    yield stage.dataset

    # ------------------------------------------------------------ 单步训练
    def train_step(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        self._optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            output: CausalLanguageModelOutput = self.model(input_ids)
            # 与参考脚本一致：shift 后做交叉熵
            shift_logits = output.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            if output.aux_loss is not None:
                loss = loss + self._aux_weight * output.aux_loss

        loss.backward()
        if self._config.grad_clip_norm and self._config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._config.grad_clip_norm)
        self._optimizer.step()
        self._scheduler.step()

        return {
            'loss/train': float(loss.detach().float()),
            'lr': self.current_lr,
        }

    # ------------------------------------------------------------ checkpoint
    def _lora_spec(self) -> Dict[str, Any]:
        '''序列化 LoRA 注入配置，随 adapter 存档用于一致性提示。'''
        cfg = self._lora_cfg if self._lora_active else None
        if cfg is None:
            return {}
        tm = cfg.target_modules if cfg.target_modules is not None else cfg.target
        items = tm if isinstance(tm, (list, tuple)) else [tm]
        return {
            'target': [t.__name__ if isinstance(t, type) else str(t) for t in items],
            'include': list(cfg.include or []),
            'exclude': list(cfg.exclude or []),
            'module_exclude': [t.__name__ if isinstance(t, type) else str(t)
                               for t in (cfg.module_exclude or [])],
            'r': cfg.r, 'lora_alpha': cfg.lora_alpha,
            'gate': cfg.gate, 'dora': cfg.dora,
        }

    def state_payload(self):
        if self._lora_active:
            from codon.utils.lora import get_lora_state_dict
            return {
                'adapter': get_lora_state_dict(self._model),
                'lora_spec': self._lora_spec(),
                'optimizer': self._optimizer.state_dict() if self._optimizer is not None else None,
                'scheduler': self._scheduler.state_dict() if self._scheduler is not None else None,
            }
        return {
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict() if self._optimizer is not None else None,
            'scheduler': self._scheduler.state_dict() if self._scheduler is not None else None,
        }

    def load_state_payload(self, payload):
        '''base.load_checkpoint() 在 setup() 前调用，先缓存、setup 后真正恢复。'''
        self._pending_payload = payload
        if self._setup_done:
            self._apply_payload(payload)

    def _apply_payload(self, payload):
        adapter = payload.get('adapter')
        if adapter is not None:
            from codon.utils.lora import has_lora
            if not has_lora(self._model):
                raise RuntimeError(
                    'checkpoint 为 LoRA adapter，但模型未注入 LoRA。'
                    '请设置与存档一致的 lora=LoRAConfig(enabled=True, ...)。')
            cur = self._model.state_dict()
            missing = [k for k in adapter if k not in cur]
            if missing:
                raise RuntimeError(
                    f'adapter 与当前模型结构不匹配，缺失 key {missing[:6]}。'
                    f'请保持 LoRA 注入配置一致；存档 spec={payload.get("lora_spec")}')
            self._model.load_state_dict(adapter, strict=False)
        elif payload.get('model') is not None:
            self._model.load_state_dict(payload['model'])
        if self._optimizer is not None and payload.get('optimizer') is not None:
            self._optimizer.load_state_dict(payload['optimizer'])
        if self._scheduler is not None and payload.get('scheduler') is not None:
            self._scheduler.load_state_dict(payload['scheduler'])

    # ------------------------------------------------------------ train（自动续训）
    def train(
        self,
        dataset: Optional[List[Any]] = None,
        num_epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        batch_fn: Optional[Callable[[List[Any]], Iterator[Any]]] = None,
        grad_accum_steps: int = 1,
        use_amp: bool = False,
        max_grad_norm: Optional[float] = None,
        eval_every_epochs: Optional[int] = None,
        eval_dataset: Optional[List[Any]] = None,
        resume_from: Optional[str] = None,
    ) -> Dict[str, float]:
        # 不传 num_epochs 时自动 = 所有阶段 epochs 之和
        if num_epochs is None:
            num_epochs = sum(s.epochs for s in self._stages)

        # checkpoint 内化：不传 resume_from 时自动加载 last.pt
        if resume_from is None:
            auto = os.path.join(self._config.ckpt_dir, 'last.pt')
            if os.path.exists(auto):
                resume_from = auto
                print(f'[*] Auto-resuming from {auto}')
            else:
                print(f'[*] No checkpoint at {auto}, starting from scratch.')

        return super().train(
            dataset if dataset is not None else self._stages,
            num_epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            batch_fn=batch_fn,
            grad_accum_steps=grad_accum_steps,
            use_amp=use_amp,
            max_grad_norm=max_grad_norm,
            eval_every_epochs=eval_every_epochs,
            eval_dataset=eval_dataset,
            resume_from=resume_from,
        )

    # ------------------------------------------------------------ stage / final 导出
    def save_stage(self, stage: SFTStage) -> None:
        '''阶段完成权重导出。
        LoRA 模式默认 adapter-only（save_lora 为主）；save_merged=True 时额外 merge 后
        导出全量到 stage.ckpt 并立即 unmerge，保持训练态不中断。'''
        if self._lora_active:
            if self._lora_cfg.save_merged:
                self._model.merge_lora()
                try:
                    self._model.save_pretrained(stage.ckpt)
                finally:
                    self._model.unmerge_lora()
            else:
                self._model.save_lora(stage.ckpt)
        else:
            self._model.save_pretrained(stage.ckpt)

    # ------------------------------------------------------------ 兜底保存 / 清理
    def _exit_safety_save(self):
        '''exit_manager 兜底：进程退出（含 SIGTERM/崩溃）时再存一次 last.pt。'''
        try:
            if self._optimizer is not None:
                path = os.path.join(self._config.ckpt_dir, 'last.pt')
                self.save_checkpoint(path)
                print(f'[*] Exit safety checkpoint saved to {path}')
        except Exception as e:
            print(f'[!] Exit safety save failed: {e}')

    def teardown(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()