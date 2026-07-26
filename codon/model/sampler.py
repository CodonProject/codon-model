from codon import *

class Sampler:
    def __init__(
        self,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.15
    ) -> None:
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    @torch.no_grad()
    def __call__(self, logits: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        Args:
            logits (torch.Tensor): [batch_size, vocab_size]
            input_ids (torch.Tensor, optional): token ids [batch_size, seq_len]
        '''
        # 0. Repetition Penalty
        if self.repetition_penalty != 1.0 and input_ids is not None:
            for i in range(logits.shape[0]):
                unique_tokens = torch.unique(input_ids[i])
                for token_id in unique_tokens:
                    val = logits[i, token_id]
                    if val > 0:
                        logits[i, token_id] = val / self.repetition_penalty
                    else:
                        logits[i, token_id] = val * self.repetition_penalty

        # 1. Temperature
        if self.temperature != 1.0:
            temp = max(self.temperature, 1e-5)
            logits = logits / temp

        # 2. Top-K
        if self.top_k is not None and self.top_k > 0:
            top_k = min(self.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # 3. Top-P
        if self.top_p is not None and 0.0 < self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token