from codon import *
from codon.config import field, configclass
from codon.pipeline.base import BasicPipeline, PipelinePhase
from codon.pipeline.callback import Callback
from codon.model.types.language import CausalLanguageModel, CausalLanguageModelOutput
from codon.utils.tokens import PackedTokenizer

from tqdm import tqdm

# =========================================================================
# 一、配置
# =========================================================================

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


# =========================================================================
# 二、SFT 阶段
# =========================================================================

@dataclass
class SFTStage:
    name: str
    dataset: Any                    # MotifSFT（__getitem__ 返回预批处理 dict）
    epochs: int = 1
    ckpt: Optional[str] = None      # 阶段结束 save_pretrained 到该路径


def build_sft_stages(stage_specs, tokenizer, pad_length, batch_size, **ds_kwargs):
    '''stage_specs: [{name, folder, epochs, ckpt}, ...] -> List[SFTStage]'''
    from codon.motif.data import MotifSFT
    stages = []
    for s in stage_specs:
        ds = MotifSFT(
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
            pipeline._model.save_pretrained(stage.ckpt)
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
        stages: List[SFTStage],
        config: SFTConfig,
        device=None,
        callbacks=None,
        seed=None,
    ):
        device = model.device if not device else device
        super().__init__(device, callbacks, seed)
        self.callbacks = list(self.callbacks)

        self._model = model
        self._model_compiled: Optional[CausalLanguageModel] = (
            torch.compile(self._model, dynamic=True) if config.compiled else None
        )
        self._tokenizer = tokenizer
        self._config = config
        self._stages: List[SFTStage] = list(stages)
        self._current_stage: Optional[SFTStage] = None

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
    def setup(self):
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

        # bias / LayerNorm 等 1 维参数不 weight decay
        decay, no_decay = [], []
        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or 'norm' in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)

        self._optimizer = torch.optim.AdamW(
            [
                {'params': decay, 'weight_decay': cfg.weight_decay},
                {'params': no_decay, 'weight_decay': 0.0},
            ],
            lr=cfg.learning_rate,
        )

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
                loss = loss + output.aux_loss

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
    def state_payload(self):
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
        if payload.get('model') is not None:
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