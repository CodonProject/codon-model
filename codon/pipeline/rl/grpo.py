"""
GRPO Pipeline: Group Relative Policy Optimization (Shao et al., 2024)
with Dr.GRPO / DAPO options.

Data flow:
    prompts → group rollout (每 prompt 生成 G 条, Sampler)
    → reward_fn (整步单次调用, group-major 顺序)
    → GRPOLoss.group_advantages (组内归一化, static)
    → collate [prompt|response|pad] → no-grad old/ref token logprobs
    → inner iterations: token 级 policy logprobs → GRPOLoss → backward

Alignment convention (关键):
    utils.token_logprobs 返回 [B, T-1], 位置 i = token i+1 的 logprob。
    本 pipeline 所有逐 token 张量 (old/ref/policy logprobs, token_mask,
    advantages) 一律采用该约定; labels 中 -100 标记 prompt/pad。
"""

from codon import *
from codon.config import field, configclass
from codon.loss.rl.grpo import GRPOLoss
from codon.pipeline.base import BasicPipeline, register_pipeline, Callback
from codon.pipeline.rl.utils import token_logprobs
from codon.utils.tokens import PackedTokenizer
from codon.model.types.language import CausalLanguageModel
from codon.utils.session import Session, Message
from codon.model.sampler import Sampler

import copy
import random
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Literal, Union
from torch.optim import Optimizer


# =============================================================================
# Config
# =============================================================================

@configclass
class GRPOConfig:
    """GRPO training configuration."""

    # --- Group sampling ---
    group_size: int = field(
        default=8, validator=lambda x: x >= 2,
        description='Completions per prompt (G >= 2, 否则组内无信号)',
    )
    temperature: float = field(default=1.0, validator=lambda x: x > 0)
    batch_size: int = field(
        default=8, validator=lambda x: x > 0,
        description='Prompts per training step',
    )

    # --- Generation ---
    max_new_tokens: int = field(default=512, validator=lambda x: x > 0)
    max_prompt_tokens: int = field(default=1024, validator=lambda x: x > 0)

    # --- Loss (→ GRPOLoss constructor) ---
    clip_epsilon: float = field(default=0.2, validator=lambda x: x > 0)
    clip_epsilon_high: Optional[float] = field(
        default=None,
        description='DAPO clip-higher 上界; None → 与 clip_epsilon 对称',
    )
    kl_beta: float = field(
        default=0.001, validator=lambda x: x >= 0,
        description='KL-to-reference 惩罚系数; 0 关闭 (同时不构建 ref 模型)',
    )
    kl_estimator: Literal['k1', 'k3', 'none'] = 'k3'
    token_level_norm: bool = field(
        default=True,
        description='Token 级归一化; False 复现 sequence 级旧行为',
    )

    # --- Advantage computation ---
    scale_rewards: bool = field(
        default=True,
        description='False → Dr.GRPO (仅去均值, 不除组内 std)',
    )
    reward_std_eps: float = field(default=1e-6)
    drop_zero_variance_groups: bool = field(
        default=True,
        description='DAPO dynamic sampling: 全组同奖励的组不参与更新',
    )
    mask_truncated_completions: bool = field(
        default=False,
        description='DAPO overlong filtering: 未产出 EOS 的截断回复 mask 掉',
    )

    # --- Inner optimization ---
    num_iterations: int = field(
        default=1, validator=lambda x: x >= 1,
        description='每个 rollout batch 的内层更新次数',
    )
    target_kl: float = field(
        default=0.05, validator=lambda x: x >= 0,
        description='approx_kl 早停阈值 (loss 模块输出); 0 关闭',
    )
    learning_rate: float = field(default=1e-6, validator=lambda x: x > 0)
    weight_decay: float = field(default=0.0, validator=lambda x: x >= 0)
    max_grad_norm: float = field(default=1.0, validator=lambda x: x > 0)
    use_gradient_checkpointing: bool = True
    disable_dropout: bool = True

    # --- Logging ---
    log_interval: int = field(default=10, validator=lambda x: x > 0)
    print_sample_completions: bool = True
    num_print_samples: int = 1

    # --- Special tokens ---
    pad_token_str: str = '[pad]'
    eos_token_str: str = '[im_end]'

    def __post_init__(self):
        """Cross-field constraints."""
        if self.kl_beta > 0 and self.kl_estimator == 'none':
            raise ValueError(
                'kl_beta > 0 与 kl_estimator="none" 矛盾: '
                '要么关闭 kl_beta, 要么选择 k1/k3'
            )
        if (self.clip_epsilon_high is not None
                and self.clip_epsilon_high < self.clip_epsilon):
            raise ValueError(
                f'clip_epsilon_high ({self.clip_epsilon_high}) 应 >= '
                f'clip_epsilon ({self.clip_epsilon}) — DAPO 语义是放宽上界'
            )


# =============================================================================
# Data
# =============================================================================

@dataclass
class Rollout:
    """One generated completion (member of a group)."""
    prompt_ids: List[int]
    response_ids: List[int]
    response_text: str
    truncated: bool            # 未产出 EOS (hit max_new_tokens)
    reward: float = 0.0        # filled by reward_fn
    advantage: float = 0.0     # filled by group normalization


# =============================================================================
# Pipeline
# =============================================================================

@register_pipeline('grpo')
class GRPOPipeline(BasicPipeline):
    """
    GRPO training pipeline.

    Components:
    - policy (trainable) / reference (frozen, only if kl_beta > 0)
    - GRPOLoss module (clipped surrogate + KL + group advantages static)
    - token-level logprobs via codon.pipeline.rl.utils.token_logprobs
    - reward_fn contract: 单次调用接收整步全部 completions (group-major
      顺序), 返回等长 float 列表 — 便于批量奖励模型 / 远程 API
    """

    def __init__(
        self,
        model: CausalLanguageModel,
        tokenizer: PackedTokenizer,
        reward_fn: Callable[[List[str]], List[float]],
        config: Optional[GRPOConfig] = None,
        reference_model: Optional[CausalLanguageModel] = None,
        loss: Optional[GRPOLoss] = None,
        callbacks: Optional[List[Callback]] = None,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(device=device, callbacks=callbacks, seed=seed)
        self.config = config or GRPOConfig()
        self.raw_model = model
        self.raw_ref = reference_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self._loss = loss               # None → build at setup

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        cfg = self.config

        # --- Policy ---
        self.model = self.raw_model.to_device(str(self.device))
        if cfg.use_gradient_checkpointing:
            self.model.gradient_checkpointing = True
        if cfg.disable_dropout:
            self.model.eval()

        # --- Loss module ---
        if self._loss is None:
            self._loss = GRPOLoss(
                clip_epsilon=cfg.clip_epsilon,
                clip_epsilon_high=cfg.clip_epsilon_high,
                kl_beta=cfg.kl_beta,
                kl_estimator=cfg.kl_estimator,
                token_level_norm=cfg.token_level_norm,
            )
        self.loss = self._loss.to(self.device)
        self.loss.smoke_test(self.device)               # fail-fast

        # --- Reference model: only when KL is active ---
        self.need_ref = cfg.kl_beta > 0 and cfg.kl_estimator != 'none'
        if self.need_ref:
            if self.raw_ref is not None:
                self.ref_model = self.raw_ref.to_device(str(self.device))
            else:
                self.ref_model = copy.deepcopy(self.model)  # 初始 policy
            self.ref_model.eval()
            self.ref_model.gradient_checkpointing = False  # no_grad 下无需重算
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
        else:
            self.ref_model = None

        # --- Optimizer ---
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # --- Special tokens ---
        self.pad_id = self.tokenizer.token_to_id(cfg.pad_token_str)
        if self.pad_id is None:
            self.pad_id = 0
        self.eos_id = self.tokenizer.token_to_id(cfg.eos_token_str)

    def teardown(self) -> None:
        pass

    def optimizers(self) -> Dict[str, Optimizer]:
        return {'main': self.optimizer}

    def iterate_epochs(self, dataset: List[Union[str, List[Message]]]):
        """Yield batches of prompts."""
        bs = self.config.batch_size
        while True:
            data = list(dataset)
            random.shuffle(data)
            yield [data[i:i + bs] for i in range(0, len(data), bs)]

    # ------------------------------------------------------------------ #
    # Rollout
    # ------------------------------------------------------------------ #

    def _build_prompt_ids(self, prompt: Union[str, List[Message]]) -> List[int]:
        session = Session(self.tokenizer)
        messages = (
            [Message(role='user', content=prompt)] if isinstance(prompt, str)
            else prompt
        )
        for msg in messages:
            session.add_message(msg)
        session.add_generation_prompt(
            enable_thinking=False, disable_thinking=False,
        )
        ids = session.input_ids
        if len(ids) > self.config.max_prompt_tokens:
            ids = ids[-self.config.max_prompt_tokens:]   # keep recent
        return ids

    @torch.no_grad()
    def _rollout_group(self, prompt: Union[str, List[Message]]) -> List[Rollout]:
        """Generate one group (G completions) for a single prompt."""
        cfg = self.config
        prompt_ids = self._build_prompt_ids(prompt)
        P = len(prompt_ids)
        input_tensor = torch.tensor(
            prompt_ids, dtype=torch.long, device=self.device,
        ).unsqueeze(0)
        sampler = Sampler(temperature=cfg.temperature)

        rollouts: List[Rollout] = []
        for _ in range(cfg.group_size):
            generated = self.model.generate(
                input_ids=input_tensor,
                max_new_tokens=cfg.max_new_tokens,
                sampler=sampler,
                eos_token_id=self.eos_id,
            )
            resp = generated[0, P:].tolist()
            truncated = True
            try:
                eos_pos = resp.index(self.eos_id)
                resp = resp[:eos_pos + 1]
                truncated = False
            except ValueError:
                pass                          # no EOS → hit length limit
            if not resp:
                resp, truncated = [self.eos_id], True

            rollouts.append(Rollout(
                prompt_ids=list(prompt_ids),
                response_ids=resp,
                response_text=self.tokenizer.decode(
                    resp, skip_special_tokens=True,
                ),
                truncated=truncated,
            ))
        return rollouts

    # ------------------------------------------------------------------ #
    # Collation
    # ------------------------------------------------------------------ #

    def _collate(
        self,
        rollouts: List[Rollout],
        adv_flat: torch.Tensor,            # [B] per-sequence advantages
    ) -> Dict[str, Any]:
        """
        Collate kept rollouts into padded tensors.

        Layout: [prompt | response | pad]; response positions trainable.
        All per-token tensors use the [B, T-1] shifted convention of
        utils.token_logprobs (position i ↔ token i+1).
        """
        B = len(rollouts)
        T_max = max(len(r.prompt_ids) + len(r.response_ids) for r in rollouts)

        input_ids = torch.full((B, T_max), self.pad_id, dtype=torch.long,
                               device=self.device)
        attention_mask = torch.zeros((B, T_max), dtype=torch.long,
                                     device=self.device)
        labels = torch.full((B, T_max), -100, dtype=torch.long,
                            device=self.device)
        truncated = torch.zeros(B, dtype=torch.bool, device=self.device)

        for i, r in enumerate(rollouts):
            P, n = len(r.prompt_ids), len(r.response_ids)
            L = P + n
            input_ids[i, :L] = torch.tensor(
                r.prompt_ids + r.response_ids, device=self.device,
            )
            attention_mask[i, :L] = 1
            labels[i, P:L] = torch.tensor(r.response_ids, device=self.device)
            truncated[i] = r.truncated

        # Trainable mask in shifted space: labels[:, 1:] != -100
        token_mask = (labels[:, 1:] != -100).float()          # [B, T-1]
        if self.config.mask_truncated_completions:
            token_mask = token_mask * (~truncated).unsqueeze(-1).float()

        # Broadcast per-sequence advantage to token level
        adv_tok = adv_flat.to(self.device).unsqueeze(-1).expand_as(token_mask)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'token_mask': token_mask,
            'advantages': adv_tok,
        }

    # ------------------------------------------------------------------ #
    # Token logprobs — delegated to utils (thin adapter)
    # ------------------------------------------------------------------ #

    def _token_logprobs_of(
        self,
        model: CausalLanguageModel,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Forward + per-token logprobs [B, T-1].

        Grad mode inherits the caller: train_step 的 policy 分支带梯度,
        old/ref 分支包在 no_grad 上下文中 — 适配器保持纯转发。
        """
        out = model.forward(
            input_ids=batch['input_ids'],
            mask=batch['attention_mask'].float(),
        )
        tok_lp, _ = token_logprobs(out.logits, batch['labels'])
        return tok_lp

    # ------------------------------------------------------------------ #
    # Train step
    # ------------------------------------------------------------------ #

    def train_step(self, batch: List[Union[str, List[Message]]]) -> Dict[str, float]:
        """
        One GRPO iteration:
        rollout → rewards → group advantages → collate → inner iterations
        """
        cfg = self.config
        G = cfg.group_size
        if not batch:
            return {'skipped': 1.0}

        # --- Phase 1: rollout (group-major flat order) ---
        rollouts: List[Rollout] = []
        for prompt in batch:
            rollouts.extend(self._rollout_group(prompt))
        num_groups = len(batch)

        # --- Phase 2: rewards — reward_fn 单次调用, 全部 completions ---
        rewards = self.reward_fn([r.response_text for r in rollouts])
        assert len(rewards) == len(rollouts), (
            f'reward_fn returned {len(rewards)} rewards for '
            f'{len(rollouts)} completions'
        )
        for r, rw in zip(rollouts, rewards):
            r.reward = float(rw)

        group_rewards = torch.tensor(
            [r.reward for r in rollouts], dtype=torch.float32,
        ).view(num_groups, G)

        reward_stats = {
            'reward/mean': float(group_rewards.mean()),
            'reward/std': float(group_rewards.std()),
            'reward/max': float(group_rewards.max()),
            'reward/min': float(group_rewards.min()),
            'response_len/mean': float(
                sum(len(r.response_ids) for r in rollouts) / len(rollouts)
            ),
        }

        # --- Phase 3: zero-variance filter + group advantages ---
        if cfg.drop_zero_variance_groups:
            keep_mask = group_rewards.std(dim=-1) > cfg.reward_std_eps
        else:
            keep_mask = torch.ones(num_groups, dtype=torch.bool)
        kept_groups = int(keep_mask.sum())

        if kept_groups == 0:
            # 全部组零方差: 无梯度信号, 跳过更新 (DAPO dynamic sampling 语义)
            return {
                **reward_stats,
                'groups/kept': 0.0,
                'groups/total': float(num_groups),
                'skipped': 1.0,
            }

        kept_rollouts: List[Rollout] = []
        for gi in range(num_groups):
            if keep_mask[gi]:
                kept_rollouts.extend(rollouts[gi * G:(gi + 1) * G])
        for r, a in zip(kept_rollouts,
                        GRPOLoss.group_advantages(
                            group_rewards[keep_mask],
                            eps=cfg.reward_std_eps,
                            scale=cfg.scale_rewards,
                        ).reshape(-1).tolist()):
            r.advantage = a
        adv_flat = torch.tensor(
            [r.advantage for r in kept_rollouts], dtype=torch.float32,
        )

        # --- Phase 4: collate + no-grad old/ref logprobs ---
        # old_logprobs 走 teacher-forced 前向而非采样器逐步记录:
        # 与训练前向同代码路径 → num_iterations=1 时 ratio 精确为 1
        bt = self._collate(kept_rollouts, adv_flat)
        with torch.no_grad():
            old_lp = self._token_logprobs_of(self.model, bt)
            ref_lp = (
                self._token_logprobs_of(self.ref_model, bt)
                if self.ref_model is not None else None
            )

        # --- Phase 5: inner iterations (KL-gated) ---
        it_metrics: List[Dict[str, float]] = []
        iterations_run = 0
        for _ in range(cfg.num_iterations):
            new_lp = self._token_logprobs_of(self.model, bt)   # with grad

            loss_out = self.loss(
                policy_logprobs=new_lp,
                old_logprobs=old_lp,
                advantages=bt['advantages'],
                token_mask=bt['token_mask'],
                ref_logprobs=ref_lp,
            )

            loss_out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            iterations_run += 1
            it_metrics.append(loss_out.metrics)

            # KL 早停: 消费 loss 模块的 approx_kl, pipeline 不自算 KL
            if (cfg.target_kl > 0
                    and loss_out.metrics['approx_kl'] > 1.5 * cfg.target_kl):
                break

        # --- Aggregate inner-iteration metrics ---
        agg: Dict[str, float] = {}
        for m in it_metrics:
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + float(v)
        n_it = max(len(it_metrics), 1)
        agg = {k: v / n_it for k, v in agg.items()}

        metrics = {
            **reward_stats,
            'groups/kept': float(kept_groups),
            'groups/total': float(num_groups),
            'iterations_run': float(iterations_run),
            'lr': self.optimizer.param_groups[0]['lr'],
            **agg,
        }

        if (cfg.print_sample_completions
                and self.state.global_step % cfg.log_interval == 0):
            self._print_samples(rollouts, num_groups)
        return metrics

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate(
        self,
        dataset: Optional[List[Union[str, List[Message]]]] = None,
    ) -> Dict[str, float]:
        """
        Held-out prompts: mean reward + 组内 std + 回复长度。

        eval/reward_std_within_group → 0 意味着策略塌缩 (组内无差异,
        归一化失去信号) — GRPO 特有的健康度指标。
        """
        if not dataset:
            return {}
        all_rewards: List[float] = []
        all_lens: List[int] = []
        group_stds: List[float] = []
        for prompt in dataset:
            group = self._rollout_group(prompt)
            rs = [float(x) for x in self.reward_fn(
                [r.response_text for r in group],
            )]
            all_rewards.extend(rs)
            all_lens.extend(len(r.response_ids) for r in group)
            t = torch.tensor(rs, dtype=torch.float32)
            group_stds.append(float(t.std()) if len(rs) > 1 else 0.0)
        n = max(len(all_rewards), 1)
        return {
            'eval/reward': sum(all_rewards) / n,
            'eval/reward_std_within_group': (
                sum(group_stds) / max(len(group_stds), 1)
            ),
            'eval/response_len': sum(all_lens) / max(len(all_lens), 1),
        }

    # ------------------------------------------------------------------ #
    # Debug: single-group inspection (reward hacking 排查)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def inspect_group(
        self,
        prompt: Union[str, List[Message]],
    ) -> Dict[str, Any]:
        """
        单 prompt 全组报告: 每条 completion 的奖励 + 归一化优势 + 文本。
        定位 reward hacking (奖励高但文本退化的 completion) 的第一入口。
        """
        if not self._setup_done:
            self.setup()
        cfg = self.config
        group = self._rollout_group(prompt)
        rs = [float(x) for x in self.reward_fn(
            [r.response_text for r in group],
        )]
        for r, rw in zip(group, rs):
            r.reward = rw
        adv = GRPOLoss.group_advantages(
            torch.tensor(rs, dtype=torch.float32).unsqueeze(0),
            eps=cfg.reward_std_eps,
            scale=cfg.scale_rewards,
        ).squeeze(0)
        return {
            'rewards': rs,
            'advantages': adv.tolist(),
            'completions': [
                {
                    'reward': r.reward,
                    'advantage': float(a),
                    'truncated': r.truncated,
                    'len': len(r.response_ids),
                    'text': r.response_text,
                }
                for r, a in zip(group, adv)
            ],
        }

    # ------------------------------------------------------------------ #
    # Checkpoint
    # ------------------------------------------------------------------ #

    def state_payload(self) -> Dict[str, Any]:
        # Reference model 不入档: 它定义为初始 policy (或显式给定的冻结
        # 模型), setup() 时重建 — 绝不能吸收训练后的权重
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def load_state_payload(self, payload: Dict[str, Any]) -> None:
        """
        Setup-on-demand + 顺序保证: ref model 必须先于 checkpoint 权重
        从初始 policy deepcopy, 否则 resume 后 KL 锚点漂移。
        """
        if not self._setup_done:
            self.setup()
        self.model.load_state_dict(payload['model'])
        self.optimizer.load_state_dict(payload['optimizer'])

    # ------------------------------------------------------------------ #
    # Sample printing
    # ------------------------------------------------------------------ #

    def _print_samples(self, rollouts: List[Rollout], num_groups: int) -> None:
        """打印奖励差最大的一组: best vs worst completion."""
        cfg = self.config
        if not rollouts:
            return
        G = cfg.group_size
        best_g, best_spread = 0, float('-inf')
        for gi in range(num_groups):
            rs = [r.reward for r in rollouts[gi * G:(gi + 1) * G]]
            spread = max(rs) - min(rs)
            if spread > best_spread:
                best_spread, best_g = spread, gi
        group = rollouts[best_g * G:(best_g + 1) * G]
        winner = max(group, key=lambda r: r.reward)
        loser = min(group, key=lambda r: r.reward)
        print(f"\n[group sample | spread={best_spread:.2f}]")
        print(f"  best  r={winner.reward:+.2f}: {winner.response_text[:150]}")
        print(f"  worst r={loser.reward:+.2f}: {loser.response_text[:150]}")
