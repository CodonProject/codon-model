from codon import *
from codon.config import field, configclass
from codon.pipeline.base import BasicPipeline, PipelinePhase
from codon.pipeline.callback import Callback
from codon.model.types.language import CausalLanguageModel, CausalLanguageModelOutput
from codon.utils.plan import ContextTrainingPlanner, StatefulPlanRunner

from tqdm import tqdm


# =========================================================================
# 一、配置（替代 build_config）
# =========================================================================

@configclass
class PretrainConfig:
    # ---- 模型 ----
    compiled: bool = field(default=True)

    # ---- 分段上下文计划 ----
    base_context: int = field(default=512)
    target_context: int = field(default=4096)
    global_batch_tokens: int = field(default=8192 * 2)
    step_mode: str = field(default='recommended')   # 'min' | 'recommended' | 'overfit'

    # ---- 优化器 ----
    learning_rate: float = field(default=3e-4)
    weight_decay: float = field(default=0.1)
    beta1: float = field(default=0.9)
    beta2: float = field(default=0.95)
    eps: float = field(default=1e-8)
    grad_clip_norm: float = field(default=1.0)
    min_lr_ratio: float = field(default=0.1)
    warmup_steps: Optional[int] = field(default=None)

    # ---- checkpoint（内化）----
    ckpt_dir: str = field(default='./ckpt')
    save_every_steps: int = field(default=2000)
    keep_last: int = field(default=3)

    # ---- 进度条 ----
    use_progress: bool = field(default=True)

    # ---- sanity check（prompt 前缀列表）----
    sanity_prompts: List[str] = field(default_factory=lambda: ['The'])
    sanity_every_steps: int = field(default=2000)
    sanity_max_new_tokens: int = field(default=50)


def cal_warmup(total_steps: int) -> int:
    if total_steps >= 20000:
        return max(2000, min(10000, int(total_steps * 0.05)))
    return max(1, int(total_steps * 0.08))


# =========================================================================
# 二、自实现学习率调度器（替代 build_optim_and_scheduler 里的 scheduler）
# =========================================================================

class WarmupCosineSchedule(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1, last_epoch=-1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            scale = step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos
        return [base * scale for base in self.base_lrs]


# =========================================================================
# 三、内置回调：进度条 / 自动 checkpoint / sanity check
# =========================================================================

class _ProgressBar(Callback):
    '''挂在基类 on_step_end 上的 tqdm 进度条。'''

    def __init__(self, pipeline, num_epochs=1):
        self.pipeline = pipeline
        self.num_epochs = num_epochs
        self.pbar = None

    def on_train_start(self, pipeline):
        total = pipeline.total_steps * max(1, self.num_epochs)
        self.pbar = tqdm(
            total=total,
            initial=pipeline.state.global_step,   # 续训后从恢复的步数继续
            desc='Pretraining',
            dynamic_ncols=True,
        )

    def on_step_end(self, pipeline, step_metrics):
        if self.pbar is None:
            return
        self.pbar.set_postfix({
            'Loss': f"{step_metrics.get('loss/train', 0.0):.4f}",
            'LR':   f"{step_metrics.get('lr', 0.0):.2e}",
            'Seq':  int(step_metrics.get('seq_len', 0)),
        })
        self.pbar.update(1)

    def on_train_end(self, pipeline):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class _AutoCheckpoint(Callback):
    '''周期 last.pt + 正常 final.pt + 中断/异常紧急保存。'''

    def __init__(self, pipeline, directory, every_steps, keep_last=3):
        self.pipeline = pipeline
        self.dir = directory
        self.every = every_steps
        self.keep = keep_last
        self._ckpts: List[str] = []
        os.makedirs(directory, exist_ok=True)

    def _save(self, tag):
        path = os.path.join(self.dir, f'{tag}.pt')
        self.pipeline.save_checkpoint(path)       # 基类原子写
        tqdm.write(f'[*] Checkpoint saved to {path}')
        # 仅对编号型（非 last/interrupted/emergency）做 keep_last 清理
        if tag not in ('last', 'interrupted', 'emergency'):
            self._ckpts.append(path)
            while len(self._ckpts) > self.keep:
                victim = self._ckpts.pop(0)
                try:
                    os.remove(victim)
                except OSError:
                    pass

    def on_step_end(self, pipeline, metrics):
        if self.every and pipeline.state.global_step % self.every == 0:
            self._save('last')

    def on_interrupt(self, pipeline):             # Ctrl+C / SIGINT（基类已捕获）
        self._save('interrupted')
        self._save('last')

    def on_train_end(self, pipeline):
        # 基类 finally 里必调 on_train_end：
        #   phase==FINISHED → 正常结束；否则是异常退出 → 紧急保存
        if pipeline.state.phase == PipelinePhase.FINISHED:
            self._save('final')
            self._save('last')
        else:
            tqdm.write('[!] Training ended abnormally — saving emergency checkpoint')
            self._save('emergency')
            self._save('last')


class _SanityCheck(Callback):
    '''按 sanity_prompts 列表逐个跑 run_sanity_check。'''

    def __init__(self, pipeline, prompts, every_steps, max_new_tokens, eos_id):
        self.pipeline = pipeline
        self.prompts = list(prompts) if prompts else ['The']
        self.every = every_steps
        self.max_new_tokens = max_new_tokens
        self.eos_id = eos_id

    def _run(self):
        from codon.kit.train import run_sanity_check
        for prompt in self.prompts:
            try:
                run_sanity_check(
                    self.pipeline.model,
                    self.pipeline._tokenizer,
                    self.pipeline.device,
                    self.eos_id,
                    self.pipeline.state.global_step,
                    prompt,
                )
            except Exception as e:
                tqdm.write(f'[!] Sanity check failed for prompt {prompt!r}: {e}')

    def on_train_start(self, pipeline):
        self._run()                               # 训练开始先跑一轮（含续训后）

    def on_step_end(self, pipeline, metrics):
        if self.every and pipeline.state.global_step % self.every == 0:
            self._run()


# =========================================================================
# 四、Pipeline 本体
# =========================================================================

class PretrainPipeline(BasicPipeline):
    def __init__(self, model, tokenizer, config, device=None, callbacks=None, seed=None):
        device = model.device if not device else device
        super().__init__(device, callbacks, seed)
        self.callbacks = list(self.callbacks)     # 拷贝，避免污染调用方传入的列表

        self._model = model
        self._model_compiled: Optional[CausalLanguageModel] = (
            torch.compile(self._model, dynamic=True) if config.compiled else None
        )
        self._tokenizer = tokenizer
        self._config = config
        self._eos_id = tokenizer.fast_tokenizer.eos_token_id

        # ---- 延迟构建（setup / iterate_epochs 里填充）----
        self._plan = None
        self._total_steps = 0
        self._optimizer = None
        self._scheduler = None
        self._dataset = None
        self._runner = None

        # ---- 续训：base 的 load_checkpoint 在 setup() 前调用，先缓存 ----
        self._pending_payload: Optional[Dict[str, Any]] = None
        self._pending_runner_state: Optional[Dict[str, Any]] = None

        # ---- 注入内置回调 ----
        self._progress_cb = None
        self._ckpt_cb = None
        self._sanity_cb = None
        if config.use_progress:
            self._progress_cb = _ProgressBar(pipeline=self)
            self.callbacks.append(self._progress_cb)
        if config.save_every_steps:
            self._ckpt_cb = _AutoCheckpoint(
                pipeline=self, directory=config.ckpt_dir,
                every_steps=config.save_every_steps, keep_last=config.keep_last,
            )
            self.callbacks.append(self._ckpt_cb)
        if config.sanity_every_steps:
            self._sanity_cb = _SanityCheck(
                pipeline=self, prompts=config.sanity_prompts,
                every_steps=config.sanity_every_steps,
                max_new_tokens=config.sanity_max_new_tokens, eos_id=self._eos_id,
            )
            self.callbacks.append(self._sanity_cb)

    # ------------------------------------------------------------ 属性
    @property
    def model(self) -> CausalLanguageModel:
        return self._model_compiled if self._model_compiled is not None else self._model

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

        planner = ContextTrainingPlanner(
            self._model,
            step_mode=self._config.step_mode,
            base_context=self._config.base_context,
            target_context=self._config.target_context,
            global_batch_tokens=self._config.global_batch_tokens,
        )
        self._plan = planner.generate_plan()
        self._total_steps = self._plan.total_steps
        self._plan.print_report()

        self._build_optimizer_and_scheduler()

        # 续训载荷：optimizer/scheduler 已就绪，现在真正恢复
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
        warmup = cfg.warmup_steps if cfg.warmup_steps is not None else cal_warmup(self._total_steps)

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
            lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
        )
        self._scheduler = WarmupCosineSchedule(
            self._optimizer, warmup_steps=warmup,
            total_steps=self._total_steps, min_lr_ratio=cfg.min_lr_ratio,
        )

    # ------------------------------------------------------------ 数据 / epoch
    def _ensure_runner(self, dataset):
        if self._runner is not None and self._dataset is dataset:
            return
        self._dataset = dataset
        self._runner = StatefulPlanRunner(self._plan, dataset, self._eos_id)
        if self._pending_runner_state is not None:
            self._runner.load_state_dict(self._pending_runner_state)
            self._pending_runner_state = None

    def iterate_epochs(self, dataset):
        '''一个 "epoch" == 跑完整张计划（Foundation -> Expansion_* -> Stabilization）。'''
        self._ensure_runner(dataset)
        while True:
            yield self._runner
            self._runner = StatefulPlanRunner(self._plan, self._dataset, self._eos_id)

    # ------------------------------------------------------------ 单步训练
    def train_step(self, batch):
        stage, inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        self._optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            output: CausalLanguageModelOutput = self.model(inputs)
            loss = F.cross_entropy(output.logits.view(-1, output.logits.size(-1)), labels.view(-1))
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
            'seq_len': float(stage.seq_len),
        }

    # ------------------------------------------------------------ checkpoint
    def state_payload(self):
        return {
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict() if self._optimizer is not None else None,
            'scheduler': self._scheduler.state_dict() if self._scheduler is not None else None,
            'runner': self._runner.state_dict() if self._runner is not None else None,
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
        if payload.get('runner') is not None:
            self._pending_runner_state = payload['runner']
            if self._runner is not None:
                self._runner.load_state_dict(self._pending_runner_state)
                self._pending_runner_state = None

    # ------------------------------------------------------------ train（自动续训）
    def train(
        self,
        dataset: List[Any],
        num_epochs: int = 1,
        steps_per_epoch: Optional[int] = None,
        batch_fn: Optional[Callable[[List[Any]], Iterator[Any]]] = None,
        grad_accum_steps: int = 1,
        use_amp: bool = False,
        max_grad_norm: Optional[float] = None,
        eval_every_epochs: Optional[int] = None,
        eval_dataset: Optional[List[Any]] = None,
        resume_from: Optional[str] = None,
    ) -> Dict[str, float]:
        # ---- checkpoint 内化：不传 resume_from 时自动加载 last.pt ----
        if resume_from is None:
            auto = os.path.join(self._config.ckpt_dir, 'last.pt')
            if os.path.exists(auto):
                resume_from = auto
                print(f'[*] Auto-resuming from {auto}')
            else:
                print(f'[*] No checkpoint at {auto}, starting from scratch.')

        if self._progress_cb is not None:
            self._progress_cb.num_epochs = num_epochs

        return super().train(
            dataset,
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

    # ------------------------------------------------------------ 兜底保存
    def _exit_safety_save(self):
        '''exit_manager 兜底：进程退出（含 SIGTERM/崩溃）时再存一次 last.pt。'''
        try:
            if self._optimizer is not None:
                path = os.path.join(self._config.ckpt_dir, 'last.pt')
                self.save_checkpoint(path)
                print(f'[*] Exit safety checkpoint saved to {path}')
        except Exception as e:
            print(f'[!] Exit safety save failed: {e}')

    # ------------------------------------------------------------ 评估 / 清理
    def evaluate(self, dataset=None):
        if dataset is None:
            return {}
        self._model.eval()
        total, n = 0.0, 0
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            for item in dataset:
                if len(item) == 3:
                    _, inputs, labels = item
                else:
                    inputs, labels = item
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                output: CausalLanguageModelOutput = self.model(inputs)
                loss = F.cross_entropy(
                    output.logits.view(-1, output.logits.size(-1)), labels.view(-1))
                total += float(loss) * labels.numel()
                n += labels.numel()
        self._model.train()
        return {'loss/eval': total / max(n, 1)}

    def teardown(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()