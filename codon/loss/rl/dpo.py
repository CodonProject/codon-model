from codon import *
from codon.loss.base import *


@register_loss('dpo')
class DPOLoss(BasicLoss):
    """
    Direct Preference Optimization loss family.

    Implicit rewards (per example):
        r_w = β·(log π_θ(y_w|x) − log π_ref(y_w|x))
        r_l = β·(log π_θ(y_l|x) − log π_ref(y_l|x))
        margin = r_w − r_l

    Variants:
        dpo  : −log σ(margin)                 [Rafailov et al., 2023]
        ipo  : (margin − 1/(2τ))²             [Azar et al., 2023]
        cdpo : −log((1−ε)·σ(margin) + ε)      [conservative, label noise]
    """

    def __init__(
        self,
        beta: float = 0.1,
        loss_type: Literal['dpo', 'ipo', 'cdpo'] = 'dpo',
        ipo_tau: float = 0.1,
        cdpo_epsilon: float = 0.05,
        nll_alpha: float = 0.0,          # SFT anchor on chosen (0 disables)
        length_normalized: bool = False, # per-token logprobs (length-bias mitigation)
    ):
        super().__init__()
        assert loss_type in ('dpo', 'ipo', 'cdpo')
        self.beta = beta
        self.loss_type = loss_type
        self.ipo_tau = ipo_tau
        self.cdpo_epsilon = cdpo_epsilon
        self.nll_alpha = nll_alpha
        self.length_normalized = length_normalized

    def forward(
        self,
        policy_chosen_logprobs: torch.Tensor,    # [B], with grad
        policy_rejected_logprobs: torch.Tensor,  # [B], with grad
        ref_chosen_logprobs: torch.Tensor,       # [B], no grad
        ref_rejected_logprobs: torch.Tensor,     # [B], no grad
        chosen_token_counts: Optional[torch.Tensor] = None,    # [B]
        rejected_token_counts: Optional[torch.Tensor] = None,  # [B]
    ) -> LossOutput:
        assert policy_chosen_logprobs.dim() == 1, 'expected [B] sequence logprobs'

        pc, pr = policy_chosen_logprobs, policy_rejected_logprobs
        rc, rr = ref_chosen_logprobs, ref_rejected_logprobs

        # Optional length normalization: sequence-sum → per-token average
        if self.length_normalized:
            if chosen_token_counts is None or rejected_token_counts is None:
                raise ValueError(
                    'length_normalized=True requires token counts'
                )
            cc = chosen_token_counts.clamp(min=1).float()
            cr = rejected_token_counts.clamp(min=1).float()
            pc, rc = pc / cc, rc / cc
            pr, rr = pr / cr, rr / cr

        # --- Implicit rewards & margin ---
        chosen_rewards = self.beta * (pc - rc)
        rejected_rewards = self.beta * (pr - rr)
        margin = chosen_rewards - rejected_rewards      # [B]

        # --- Variant ---
        if self.loss_type == 'dpo':
            per_example = -F.logsigmoid(margin)
        elif self.loss_type == 'ipo':
            per_example = (margin - 1.0 / (2.0 * self.ipo_tau)) ** 2
        else:  # cdpo
            eps = self.cdpo_epsilon
            per_example = -torch.log(
                (1 - eps) * torch.sigmoid(margin) + eps + 1e-12
            )

        core_loss = per_example.mean()

        # --- Optional NLL anchor on chosen ---
        nll_term = torch.tensor(0.0, device=core_loss.device)
        if self.nll_alpha > 0:
            # pc is per-token if length_normalized else sequence-sum
            nll_term = -pc.mean()
        loss = core_loss + self.nll_alpha * nll_term

        with torch.no_grad():
            metrics = {
                'loss/total': loss.item(),
                'loss/core': core_loss.item(),
                'loss/nll_term': float(nll_term),
                'rewards/chosen': chosen_rewards.mean().item(),
                'rewards/rejected': rejected_rewards.mean().item(),
                'rewards/margin': margin.mean().item(),
                'rewards/accuracy': (margin > 0).float().mean().item(),
            }
        return LossOutput(loss=loss, metrics=metrics)

    @torch.no_grad()
    def smoke_test(self, device: torch.device) -> None:
        """Fail-fast contract validation at pipeline init."""
        B = 4
        out = self.forward(
            policy_chosen_logprobs=torch.randn(B, device=device),
            policy_rejected_logprobs=torch.randn(B, device=device),
            ref_chosen_logprobs=torch.randn(B, device=device),
            ref_rejected_logprobs=torch.randn(B, device=device),
        )
        assert out.loss.dim() == 0 and torch.isfinite(out.loss)