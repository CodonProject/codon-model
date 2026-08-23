from codon import *
from codon.loss.base import *


@register_loss('grpo')
class GRPOLoss(BasicLoss):
    """
    GRPO policy-gradient loss: clipped surrogate + KL-to-reference penalty.

        L = −1/|ℳ| Σ min(ρ·Â, clip(ρ, 1−ε, 1+ε_h)·Â) + β·KL(π_θ ‖ π_ref)

    Defaults follow modern practice:
    - token-level normalization (Dr.GRPO / DAPO)
    - k3 KL estimator (non-negative, lower variance than k1)
    - optional clip-higher (DAPO): separate upper bound ε_h > ε
    """

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        clip_epsilon_high: Optional[float] = None,   # DAPO clip-higher
        kl_beta: float = 0.001,
        kl_estimator: Literal['k1', 'k3', 'none'] = 'k3',
        token_level_norm: bool = True,
    ):
        super().__init__()
        assert kl_estimator in ('k1', 'k3', 'none')
        self.clip_epsilon = clip_epsilon
        self.clip_epsilon_high = clip_epsilon_high
        self.kl_beta = kl_beta
        self.kl_estimator = kl_estimator
        self.token_level_norm = token_level_norm

    # ------------------------------------------------------------------ #
    # Data-side helper: group advantage normalization (tightly coupled
    # to the objective, kept static — pure function, no module state)
    # ------------------------------------------------------------------ #

    @staticmethod
    def group_advantages(
        rewards: torch.Tensor,       # [num_groups, G]
        eps: float = 1e-6,
        scale: bool = True,          # False → Dr.GRPO (mean-center only)
    ) -> torch.Tensor:
        """(r − mean_group) / (std_group + eps), broadcastable to tokens."""
        centered = rewards - rewards.mean(dim=-1, keepdim=True)
        if not scale:
            return centered
        return centered / (rewards.std(dim=-1, keepdim=True) + eps)

    # ------------------------------------------------------------------ #

    def forward(
        self,
        policy_logprobs: torch.Tensor,   # [B, T] with grad (current π_θ)
        old_logprobs: torch.Tensor,      # [B, T] no grad (rollout π_old)
        advantages: torch.Tensor,        # [B, T] no grad (group-normalized)
        token_mask: torch.Tensor,        # [B, T] 1.0 for trainable tokens
        ref_logprobs: Optional[torch.Tensor] = None,   # [B, T] no grad
    ) -> LossOutput:
        assert policy_logprobs.shape == old_logprobs.shape
        assert advantages.shape == token_mask.shape

        # --- Importance ratio ---
        log_ratio = policy_logprobs - old_logprobs
        ratio = torch.exp(log_ratio)

        # --- Clipped surrogate (optional asymmetric upper bound) ---
        eps_lo = 1.0 - self.clip_epsilon
        eps_hi = 1.0 + (
            self.clip_epsilon_high
            if self.clip_epsilon_high is not None
            else self.clip_epsilon
        )
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, eps_lo, eps_hi) * advantages
        per_token_policy = -torch.min(surr1, surr2)      # [B, T]

        # --- KL penalty to reference ---
        kl_per_token = torch.zeros_like(per_token_policy)
        if self.kl_beta > 0 and self.kl_estimator != 'none':
            if ref_logprobs is None:
                raise ValueError('kl_beta > 0 requires ref_logprobs')
            lr_ref = policy_logprobs - ref_logprobs      # log(π_θ/π_ref)
            if self.kl_estimator == 'k1':
                kl_per_token = (torch.exp(lr_ref) - 1.0) - lr_ref
            else:  # k3
                kl_per_token = torch.exp(lr_ref) - lr_ref - 1.0

        per_token = per_token_policy + self.kl_beta * kl_per_token

        # --- Normalization ---
        if self.token_level_norm:
            denom = token_mask.sum().clamp(min=1)
            loss = (per_token * token_mask).sum() / denom
            pol_loss = (per_token_policy * token_mask).sum() / denom
            kl_loss = (kl_per_token * token_mask).sum() / denom
            approx_kl = (((torch.exp(log_ratio) - 1.0) - log_ratio)
                         * token_mask).sum() / denom
            clip_ratio = (((ratio < eps_lo) | (ratio > eps_hi)).float()
                          * token_mask).sum() / denom
            mean_ratio = (ratio * token_mask).sum() / denom
            n_tok = float(denom)
        else:
            # Sequence-level: mean over per-sequence means
            seq_denom = token_mask.sum(-1).clamp(min=1)
            seq_loss = (per_token * token_mask).sum(-1) / seq_denom
            loss = seq_loss.mean()
            pol_loss = ((per_token_policy * token_mask).sum(-1)
                        / seq_denom).mean()
            kl_loss = ((kl_per_token * token_mask).sum(-1)
                       / seq_denom).mean()
            approx_kl = ((((torch.exp(log_ratio) - 1.0) - log_ratio)
                          * token_mask).sum(-1) / seq_denom).mean()
            clip_ratio = ((((ratio < eps_lo) | (ratio > eps_hi)).float()
                           * token_mask).sum(-1) / seq_denom).mean()
            mean_ratio = ((ratio * token_mask).sum(-1) / seq_denom).mean()
            n_tok = float(token_mask.sum())

        with torch.no_grad():
            metrics = {
                'loss/total': loss.item(),
                'loss/policy': float(pol_loss),
                'loss/kl': float(kl_loss) * self.kl_beta,
                'approx_kl': float(approx_kl),
                'clip_ratio': float(clip_ratio),
                'mean_ratio': float(mean_ratio),
                'num_tokens': n_tok,
            }
        return LossOutput(loss=loss, metrics=metrics)

    @torch.no_grad()
    def smoke_test(self, device: torch.device) -> None:
        B, T = 2, 16
        lp = torch.randn(B, T, device=device)
        out = self.forward(
            policy_logprobs=lp,
            old_logprobs=lp.clone(),
            advantages=torch.randn(B, T, device=device),
            token_mask=torch.ones(B, T, device=device),
            ref_logprobs=torch.randn(B, T, device=device),
        )
        assert out.loss.dim() == 0 and torch.isfinite(out.loss)