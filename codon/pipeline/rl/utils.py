import torch
from typing import Tuple


def token_logprobs(
    logits: torch.Tensor,          # [B, T, V]
    labels: torch.Tensor,          # [B, T], -100 for masked positions
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token log-probs of labels under logits (causal shift).

    Returns:
        logprobs: [B, T-1] — log π(y_t | y_{<t}), masked positions = 0
        mask:     [B, T-1] — 1.0 for valid (label != -100) positions
    """
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = (shifted_labels != -100).float()
    safe = shifted_labels.clamp(min=0)
    lp = F.log_softmax(shifted_logits.float(), dim=-1)
    lp = lp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return lp * mask, mask


def sequence_logprob(
    logits: torch.Tensor, labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Summed sequence log-probs. Returns (seq_lp [B], mask, token_lp [B,T-1])."""
    tok_lp, mask = token_logprobs(logits, labels)
    return tok_lp.sum(dim=-1), mask, tok_lp