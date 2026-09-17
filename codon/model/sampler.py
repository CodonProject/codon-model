from codon import *

if TYPE_CHECKING:
    from codon.model.grammar import JSONConstraint


class Sampler:
    '''
    采样器：重复惩罚 + 温度 + top-k / top-p 截断，外加可选的「语法约束采样」。

    ## 语法约束（强制 JSON）

    `constraint` 是一个逐步约束解码的对象，需要提供三个方法：

        mask_logits(logits)   -> Tensor     把该步不允许的 token 置为 -inf
        apply_forcing(tokens) -> Tensor     结果已完整时把该行 token 强制换成 eos
        advance(tokens)       -> None       记录本步采样的 token 并推进状态

    `codon.model.grammar.JSONConstraint` 就是这种对象（强制 JSON / JSON Schema）。
    约束在 top-k / top-p / softmax 之前生效，所以被禁掉的 token 不可能被采样到。

    Attributes:
        temperature (float): 采样温度。
        top_k (Optional[int]): top-k 截断。
        top_p (Optional[float]): nucleus 截断。
        repetition_penalty (float): 重复惩罚系数，1.0 表示关闭。
        constraint (Optional[JSONConstraint]): 语法约束；None 表示自由采样。
    '''

    def __init__(
        self,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.15,
        constraint: Optional['JSONConstraint'] = None,
    ) -> None:
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.constraint = constraint

    def with_constraint(
        self,
        constraint: Optional['JSONConstraint'],
        eos_token_id: Optional[int] = None,
    ) -> 'Sampler':
        '''
        返回一个绑定了语法约束的采样器副本（不修改自身）。

        Args:
            constraint (JSONConstraint, optional): 约束对象；None 表示解绑。
            eos_token_id (int, optional): 覆盖约束里的强制结束 token。

        Returns:
            Sampler: 绑定了约束的新采样器。
        '''
        bound = copy.copy(self)
        bound.constraint = constraint
        if constraint is not None and eos_token_id is not None:
            constraint.eos_token_id = int(eos_token_id)
        return bound

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

        # 2. Grammar constraint（必须在 top-k / top-p / softmax 之前）
        if self.constraint is not None:
            logits = self.constraint.mask_logits(logits)

        # 3. Top-K
        if self.top_k is not None and self.top_k > 0:
            top_k = min(self.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # 4. Top-P
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

        # 5. 结果已完整 -> 强制 eos；并推进约束状态
        if self.constraint is not None:
            next_token = self.constraint.apply_forcing(next_token)
            self.constraint.advance(next_token)

        return next_token
