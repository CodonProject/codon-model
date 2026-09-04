"""
DPO Pipeline: Direct Preference Optimization (Rafailov et al., 2023)
with IPO / cDPO variants, optional NLL anchoring, length normalization.

Data flow:
    PreferencePair → Session 格式化 → padded batch
    → policy/reference 双前向 × (chosen/rejected)
    → utils.sequence_logprob → DPOLoss (纯目标函数) → backward
"""

from codon import *
from codon.config import field, configclass
from codon.loss.rl.dpo import DPOLoss
from codon.pipeline.base import BasicPipeline, register_pipeline, Callback
from codon.pipeline.rl.utils import token_logprobs, sequence_logprob
from codon.utils.tokens import PackedTokenizer
from codon.model.types.language import CausalLanguageModel
from codon.utils.session import Session, Message

import copy
import random
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any, Literal
from torch.optim import Optimizer


# =============================================================================
# Config
# =============================================================================

@configclass
class DPOConfig:
    """DPO training configuration."""

    # --- Loss variant ---
    loss_type: Literal['dpo', 'ipo', 'cdpo'] = 'dpo'
    beta: float = field(
        default=0.1,
        validator=lambda x: x > 0,
        description='Inverse temperature β for implicit reward scaling',
    )

    # --- Variant-specific ---
    ipo_tau: float = field(
        default=0.1, validator=lambda x: x > 0,
        description='IPO regularization strength (loss_type="ipo")',
    )
    cdpo_epsilon: float = field(
        default=0.05, validator=lambda x: 0 <= x < 0.5,
        description='cDPO label-noise tolerance (loss_type="cdpo")',
    )

    # --- NLL anchor on chosen ---
    nll_alpha: float = field(
        default=0.0, validator=lambda x: x >= 0,
        description='SFT anchor weight on chosen; 0 disables',
    )

    # --- Sequence handling ---
    max_prompt_tokens: int = field(default=1024, validator=lambda x: x > 0)
    max_completion_tokens: int = field(default=1280, validator=lambda x: x > 0)
    truncate_side: Literal['left', 'right'] = 'left'
    length_normalized: bool = field(
        default=False,
        description='Per-token logprobs (length-bias mitigation)',
    )

    # --- Batching / training ---
    batch_size: int = field(default=8, validator=lambda x: x > 0)
    eval_chunk_size: int = field(default=16, validator=lambda x: x > 0)
    learning_rate: float = field(default=5e-7, validator=lambda x: x > 0)
    weight_decay: float = field(default=0.0, validator=lambda x: x >= 0)
    max_grad_norm: float = field(default=1.0, validator=lambda x: x > 0)
    use_gradient_checkpointing: bool = True
    disable_dropout: bool = True

    # --- Logging / diagnostics ---
    log_interval: int = field(default=10, validator=lambda x: x > 0)
    print_samples: bool = True
    num_print_samples: int = 1
    lowprob_threshold: float = field(
        default=-10.0,
        description='Token logprob below this counts as near-zero-prob '
                    '(degeneration monitor)',
    )

    # --- Special tokens ---
    pad_token_str: str = '<|pad|>'
    eos_token_str: str = '<|im_end|>'

    def __post_init__(self):
        """Cross-field constraints."""
        if self.loss_type == 'ipo' and self.ipo_tau <= 0:
            raise ValueError(
                f'ipo_tau must be > 0 when loss_type == "ipo", '
                f'got {self.ipo_tau}'
            )
        if self.loss_type == 'cdpo' and not (0 <= self.cdpo_epsilon < 0.5):
            raise ValueError(
                f'cdpo_epsilon must be in [0, 0.5), got {self.cdpo_epsilon}'
            )


# =============================================================================
# Data
# =============================================================================

@dataclass
class PreferencePair:
    """One preference example: prompt + chosen/rejected completions."""
    prompt: List[Message]
    chosen: List[Message]
    rejected: List[Message]


# =============================================================================
# Pipeline
# =============================================================================

@register_pipeline('dpo')
class DPOPipeline(BasicPipeline):
    """
    DPO training pipeline.

    Components:
    - policy (trainable) / reference (frozen, no_grad) models
    - DPOLoss module (dpo / ipo / cdpo + NLL anchor + length norm)
    - Session-based sequence formatting, padding-aware batches
    - logprob extraction delegated to codon.pipeline.rl.utils
    """

    def __init__(
        self,
        model: CausalLanguageModel,
        tokenizer: PackedTokenizer,
        config: Optional[DPOConfig] = None,
        reference_model: Optional[CausalLanguageModel] = None,
        loss: Optional[DPOLoss] = None,
        callbacks: Optional[List[Callback]] = None,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(device=device, callbacks=callbacks, seed=seed)
        self.config = config or DPOConfig()
        self.raw_model = model
        self.raw_ref = reference_model      # None → deepcopy policy at setup
        self.tokenizer = tokenizer
        self._loss = loss                   # None → build at setup

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

        # --- Reference: clone of INITIAL policy (DPO 语义: ref = SFT init) ---
        if self.raw_ref is not None:
            self.ref_model = self.raw_ref.to_device(str(self.device))
        else:
            self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        # --- Loss module ---
        if self._loss is None:
            self._loss = DPOLoss(
                beta=cfg.beta,
                loss_type=cfg.loss_type,
                ipo_tau=cfg.ipo_tau,
                cdpo_epsilon=cfg.cdpo_epsilon,
                nll_alpha=cfg.nll_alpha,
                length_normalized=cfg.length_normalized,
            )
        self.loss = self._loss.to(self.device)
        self.loss.smoke_test(self.device)           # fail-fast

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

    def iterate_epochs(self, dataset: List[PreferencePair]):
        """Yield batches of PreferencePair (base train() treats items as batches)."""
        bs = self.config.batch_size
        while True:
            data = list(dataset)
            random.shuffle(data)
            yield [data[i:i + bs] for i in range(0, len(data), bs)]

    # ------------------------------------------------------------------ #
    # Sequence construction (Session-based)
    # ------------------------------------------------------------------ #

    def _build_prompt_ids(self, prompt: List[Message]) -> List[int]:
        """Prompt → token ids with generation header (masked in loss)."""
        session = Session(self.tokenizer)
        for msg in prompt:
            session.add_message(msg)
        session.add_generation_prompt(
            enable_thinking=False, disable_thinking=False,
        )
        ids = session.input_ids
        if len(ids) > self.config.max_prompt_tokens:
            if self.config.truncate_side == 'left':
                ids = ids[-self.config.max_prompt_tokens:]   # keep recent
            else:
                ids = ids[:self.config.max_prompt_tokens]
        return ids

    def _build_completion_ids(self, completion: List[Message]) -> List[int]:
        """Assistant completion → token ids (loss-trainable)."""
        session = Session(self.tokenizer)
        for msg in completion:
            session.add_message(msg)
        ids = session.input_ids
        if len(ids) > self.config.max_completion_tokens:
            ids = ids[:self.config.max_completion_tokens]    # keep beginning
        return ids

    def _make_batch(
        self,
        pairs: List[PreferencePair],
        which: Literal['chosen', 'rejected'],
    ) -> Dict[str, Any]:
        """
        Collate prompt+completion into padded tensors for one branch.

        Layout: [prompt | completion | pad]; completion positions trainable
        (labels != -100), prompt/pad positions masked (labels == -100).
        """
        seqs, prompt_lens, completion_lens = [], [], []
        for pair in pairs:
            p_ids = self._build_prompt_ids(pair.prompt)
            c_ids = self._build_completion_ids(
                pair.chosen if which == 'chosen' else pair.rejected
            )
            seqs.append(p_ids + c_ids)
            prompt_lens.append(len(p_ids))
            completion_lens.append(len(c_ids))

        B = len(seqs)
        T_max = max(len(s) for s in seqs)

        input_ids = torch.full((B, T_max), self.pad_id, dtype=torch.long,
                               device=self.device)
        attention_mask = torch.zeros((B, T_max), dtype=torch.long,
                                     device=self.device)
        labels = torch.full((B, T_max), -100, dtype=torch.long,
                            device=self.device)

        for i, (seq, pl, cl) in enumerate(zip(seqs, prompt_lens, completion_lens)):
            L = pl + cl
            input_ids[i, :L] = torch.tensor(seq, device=self.device)
            attention_mask[i, :L] = 1
            labels[i, pl:L] = torch.tensor(seq[pl:], device=self.device)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'prompt_lens': prompt_lens,
            'completion_lens': completion_lens,
        }

    # ------------------------------------------------------------------ #
    # Logprob extraction — delegated to utils
    # ------------------------------------------------------------------ #

    def _branch_logprobs(
        self,
        model: CausalLanguageModel,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One model forward → branch statistics.

        Returns:
            seq_lp:  [B]   — summed sequence logprobs (DPO quantity)
            ntok:    [B]   — trainable token counts
            tok_lp:  [B, T-1] — per-token logprobs (diagnostics)
            mask:    [B, T-1] — trainable mask
        """
        out = model.forward(
            input_ids=batch['input_ids'],
            mask=batch['attention_mask'].float(),
        )
        seq_lp, mask, tok_lp = sequence_logprob(out.logits, batch['labels'])
        ntok = mask.sum(dim=-1).clamp(min=1)
        return seq_lp, ntok, tok_lp, mask

    # ------------------------------------------------------------------ #
    # Train step
    # ------------------------------------------------------------------ #

    def train_step(self, batch: List[PreferencePair]) -> Dict[str, float]:
        """One optimization iteration over a batch of PreferencePairs."""
        cfg = self.config

        chosen_batch = self._make_batch(batch, 'chosen')
        rejected_batch = self._make_batch(batch, 'rejected')

        # --- Policy forward (grad) ---
        pc_lp, pc_ntok, _, _ = self._branch_logprobs(self.model, chosen_batch)
        pr_lp, pr_ntok, _, _ = self._branch_logprobs(self.model, rejected_batch)

        # --- Reference forward (no grad) ---
        with torch.no_grad():
            rc_lp, _, _, _ = self._branch_logprobs(self.ref_model, chosen_batch)
            rr_lp, _, _, _ = self._branch_logprobs(self.ref_model, rejected_batch)

        # --- Loss module (pure objective) ---
        loss_out = self.loss(
            policy_chosen_logprobs=pc_lp,
            policy_rejected_logprobs=pr_lp,
            ref_chosen_logprobs=rc_lp,
            ref_rejected_logprobs=rr_lp,
            chosen_token_counts=pc_ntok,
            rejected_token_counts=pr_ntok,
        )

        # --- Backward + optimize ---
        loss_out.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), cfg.max_grad_norm,
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

        metrics = dict(loss_out.metrics)
        metrics['lr'] = self.optimizer.param_groups[0]['lr']

        if (cfg.print_samples
                and self.state.global_step % cfg.log_interval == 0):
            self._print_samples(batch)
        return metrics

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate(
        self, dataset: Optional[List[PreferencePair]] = None,
    ) -> Dict[str, float]:
        """
        Validation metrics on held-out pairs:
        - eval/loss, eval/accuracy: loss module 标准输出
        - eval/chosen_nll: DPO 已知失效模式监测 — chosen 似然随 rejected
          一起塌缩时该值持续上升 (Rafailov et al. §5.2, Pal et al. 2024)
        - eval/lowprob_frac: chosen 中近零概率 token 占比 (退化指示器)
        """
        if not dataset:
            return {}
        cfg = self.config
        self.model.eval()

        loss_sum = acc_sum = nll_sum = lowprob_sum = 0.0
        n = 0

        for i in range(0, len(dataset), cfg.eval_chunk_size):
            pairs = dataset[i:i + cfg.eval_chunk_size]
            cb = self._make_batch(pairs, 'chosen')
            rb = self._make_batch(pairs, 'rejected')

            pc_lp, pc_ntok, pc_tok, pc_mask = self._branch_logprobs(self.model, cb)
            pr_lp, pr_ntok, _, _ = self._branch_logprobs(self.model, rb)
            rc_lp, _, _, _ = self._branch_logprobs(self.ref_model, cb)
            rr_lp, _, _, _ = self._branch_logprobs(self.ref_model, rb)

            out = self.loss(
                pc_lp, pr_lp, rc_lp, rr_lp, pc_ntok, pr_ntok,
            )

            # Token-level diagnostics on chosen (来自 _branch_logprobs 的
            # tok_lp/mask, 无额外前向)
            chosen_nll = -(pc_tok.sum() / pc_mask.sum().clamp(min=1))
            lowprob = (((pc_tok < cfg.lowprob_threshold) & (pc_mask > 0)).sum()
                       / pc_mask.sum().clamp(min=1))

            w = len(pairs)
            loss_sum += out.loss.item() * w
            acc_sum += out.metrics['rewards/accuracy'] * w
            nll_sum += chosen_nll.item() * w
            lowprob_sum += lowprob.item() * w
            n += w

        denom = max(n, 1)
        return {
            'eval/loss': loss_sum / denom,
            'eval/accuracy': acc_sum / denom,
            'eval/chosen_nll': nll_sum / denom,
            'eval/lowprob_frac': lowprob_sum / denom,
        }

    # ------------------------------------------------------------------ #
    # Debug: per-token inspection (uses token_logprobs)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def inspect_pair(
        self,
        pair: PreferencePair,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Per-token logprob report for a single pair.

        Shows which tokens carry the probability mass in chosen and which
        get crushed in rejected — token-level granularity requires
        utils.token_logprobs (sequence sums come from inspect's own forward).
        """
        if not self._setup_done:
            self.setup()

        report: Dict[str, Any] = {}
        for name in ('chosen', 'rejected'):
            b = self._make_batch([pair], name)
            out = self.model.forward(
                input_ids=b['input_ids'],
                mask=b['attention_mask'].float(),
            )
            tok_lp, mask = token_logprobs(out.logits, b['labels'])
            valid = tok_lp[mask.bool()]
            report[name] = {
                'seq_logprob': float(valid.sum()) if valid.numel() else 0.0,
                'mean_token_logprob': (
                    float(valid.mean()) if valid.numel() else 0.0
                ),
                'worst_tokens': self._worst_tokens(b, tok_lp, mask, top_k),
            }
        return report

    def _worst_tokens(
        self,
        batch: Dict[str, Any],
        tok_lp: torch.Tensor,       # [1, T-1]
        mask: torch.Tensor,         # [1, T-1]
        k: int,
    ) -> List[Dict[str, Any]]:
        """k 个最低 logprob 的 token (shifted 对齐: tok_lp[:,t-1] ↔ ids[:,t])."""
        n_valid = int(mask.sum())
        if n_valid == 0:
            return []
        vals, idx = torch.topk(
            tok_lp.flatten(), k=min(k, n_valid), largest=False,
        )
        result = []
        for v, i in zip(vals.tolist(), idx.tolist()):
            t = i + 1                                   # shift alignment
            tok_id = int(batch['input_ids'][0, t])
            result.append({
                'token': self.tokenizer.decode([tok_id]),
                'logprob': v,
            })
        return result

    # ------------------------------------------------------------------ #
    # Checkpoint
    # ------------------------------------------------------------------ #

    def state_payload(self) -> Dict[str, Any]:
        # NOTE: reference model intentionally NOT saved — it is defined as
        # the initial policy (or an explicitly provided frozen model), so it
        # is rebuilt at setup() and must never absorb trained weights.
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def load_state_payload(self, payload: Dict[str, Any]) -> None:
        """
        Setup-on-demand: checkpoint load may arrive before setup() (base
        train() resumes prior to setup). Order matters — setup() deepcopies
        the reference model from the INITIAL policy weights, which must
        happen before trained weights land in self.model.
        """
        if not self._setup_done:
            self.setup()
        self.model.load_state_dict(payload['model'])
        self.optimizer.load_state_dict(payload['optimizer'])

    # ------------------------------------------------------------------ #
    # Sample printing
    # ------------------------------------------------------------------ #

    def _print_samples(self, pairs: List[PreferencePair]) -> None:
        cfg = self.config
        if not cfg.print_samples or not pairs:
            return
        k = min(cfg.num_print_samples, len(pairs))
        for pair in pairs[:k]:
            print(f"\n[pair sample]")
            print(f"  chosen   : {self._decode_messages(pair.chosen)[:120]}")
            print(f"  rejected : {self._decode_messages(pair.rejected)[:120]}")

    def _decode_messages(self, messages: List[Message]) -> str:
        try:
            session = Session(self.tokenizer)
            for msg in messages:
                session.add_message(msg)
            return self.tokenizer.decode(
                session.input_ids, skip_special_tokens=True,
            )
        except Exception:
            return '<decode-error>'
