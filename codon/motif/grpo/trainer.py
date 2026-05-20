import torch
import torch.nn.functional as F
from typing import Optional

from codon.utils.session import Session
from codon.utils.tokens  import PackedTokenizer
from codon.motif.grpo.reward import compute_group_rewards
from codon.motif import CausalLanguageModel


def _logprobs_of_chosen(model, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids)
    logits = out.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def rollout_group(
    policy: CausalLanguageModel, tokenizer: PackedTokenizer, device,
    prompt_text: str, system_prompt: str,
    group_size: int, max_new_tokens: int,
    temperature: float = 1.0, top_k: Optional[int] = None,
):
    sess = Session(tokenizer)
    sess.add_message({'role': 'system', 'content': system_prompt})
    sess.add_message({'role': 'user',   'content': prompt_text})
    sess.add_generation_prompt(enable_thinking=True)
    prompt_ids = sess.input_ids
    prompt_len = len(prompt_ids)
    eos_id = tokenizer.token_to_id('[im_end]')

    p_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    full_seqs, response_texts, response_lens = [], [], []
    for _ in range(group_size):
        gen = policy.generate(
            p_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_id,
        )[0].tolist()
        full_seqs.append(gen)
        resp_ids = gen[prompt_len:]
        response_lens.append(len(resp_ids))
        response_texts.append(tokenizer.decode(resp_ids, skip_special_tokens=False))

    max_T = max(len(s) for s in full_seqs)
    pad_id = tokenizer.token_to_id('[pad]') or 0
    padded, resp_mask = [], []
    for seq in full_seqs:
        pad_n = max_T - len(seq)
        padded.append(seq + [pad_id] * pad_n)
        
        m = [0] * (prompt_len - 1)         # prompt tokens (in shifted view)
        m += [1] * (len(seq) - prompt_len) # response tokens
        m += [0] * pad_n                   # padding
        
        m = m[: max_T - 1]
        resp_mask.append(m)

    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    response_mask = torch.tensor(resp_mask, dtype=torch.float32, device=device)

    rewards = compute_group_rewards(response_texts)
    r = torch.tensor(rewards, dtype=torch.float32, device=device)
    adv = (r - r.mean()) / (r.std() + 1e-6)

    return {
        'input_ids':     input_ids,     # [G, T]
        'response_mask': response_mask, # [G, T-1]
        'rewards':       r,             # [G]
        'advantages':    adv,           # [G]
        'texts':         response_texts,
    }


def grpo_loss(
    policy_logp, old_logp, ref_logp, # [B, T-1]
    advantages,                      # [B]
    response_mask,                   # [B, T-1]
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
):
    log_ratio = policy_logp - old_logp
    ratio = torch.exp(log_ratio.clamp(-20, 20))

    A = advantages.unsqueeze(1)      # [B, 1]
    pg1 = ratio * A
    pg2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * A
    pg_loss = -torch.min(pg1, pg2)

    # k3 unbiased KL estimator: KL(policy || ref)
    delta = ref_logp - policy_logp
    kl = torch.exp(delta.clamp(-20, 20)) - delta - 1.0

    per_tok = pg_loss + kl_beta * kl
    denom = response_mask.sum().clamp(min=1.0)
    loss = (per_tok * response_mask).sum() / denom

    stats = {
        'pg_loss': (pg_loss * response_mask).sum().item() / denom.item(),
        'kl':      (kl * response_mask).sum().item() / denom.item(),
        'ratio':   (ratio * response_mask).sum().item() / denom.item(),
    }
    return loss, stats