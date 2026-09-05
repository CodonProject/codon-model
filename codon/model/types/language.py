from codon import *
from codon.model.cache import ModelCache
from codon.model.sampler import Sampler

import inspect


@dataclass
class CausalLanguageModelOutput:
    '''
    Output of causal language model.

    Attributes:
        logits (torch.Tensor): Prediction logits.
        past_key_values (ModelCache, optional): The updated ModelCache container.
        aux_loss (torch.Tensor, optional): Auxiliary loss.
        attentions (list, optional): List of attention weights.
        hidden_states (tuple, optional): Tuple of hidden states.
    '''
    logits: torch.Tensor
    past_key_values: Optional[ModelCache] = None
    aux_loss: Optional[torch.Tensor] = None
    attentions: Optional[List[torch.Tensor]] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None


class CausalLanguageModel(BasicModel):
    '''
    Causal 语言模型基类，提供优化的自回归生成管线（generate / compute_perplexity）。

    ## forward 契约（子类必须遵守）

    `generate()` 会以关键字方式调用子类的 `forward`：

        outputs = self.forward(input_ids=..., start_pos=..., past_key_values=...)

    因此任何 CausalLanguageModel 子类的 `forward` **必须以 `input_ids` 为第一参数**，
    并接受 `start_pos` 与 `past_key_values` 关键字，返回 `CausalLanguageModelOutput`
    （含 `logits`，形状 [batch, seq, vocab]）。本类在子类定义时即校验该约定，
    签名不含 `input_ids`（或没有 `**kwargs` 兜底）的子类会在 import 时直接报错，
    而不是等到 generate 运行到一半才暴露。
    '''
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        forward = cls.__dict__.get('forward')
        if forward is None:
            return  # 允许中间抽象层暂不实现；真正实例化前必须补全
        try:
            params = inspect.signature(forward).parameters
        except (TypeError, ValueError):
            return  # C 扩展/不可检视的 callable，跳过静态校验
        names = set(params)
        has_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
        if 'input_ids' not in names and not has_kwargs:
            raise TypeError(
                f'{cls.__name__}.forward must accept `input_ids` as its first '
                f'parameter (CausalLanguageModel contract used by generate()). '
                f'Got signature: {inspect.signature(forward)}'
            )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        sampler: Optional[Sampler] = None,
        temperature: float = 0.7,
        eos_token_id: Optional[int] = None,
        past_key_values: Optional[ModelCache] = None
    ) -> torch.Tensor:
        '''
        Generate text tokens autoregressively using a prefill-decode pipeline.

        Args:
            input_ids (torch.Tensor): Input prompt token IDs with shape [batch, seq_len].
            max_new_tokens (int): Maximum number of new tokens to generate.
            sampler (Sampler, optional): Instance of Sampler. If None, default Sampler(0.7) is used.
            temperature (float): Sampling temperature for the default Sampler. Only used if `sampler` is None.
            eos_token_id (int, optional): End-of-sequence token ID.
            past_key_values (ModelCache, optional): Cache container to reuse states across decode steps.

        Returns:
            torch.Tensor: Generated token IDs with shape [batch, seq_len + num_generated].
        '''
        self.eval()
        if sampler is None:
            sampler = Sampler(temperature=temperature)

        generated = input_ids.clone()
        
        if past_key_values is None: past_key_values = ModelCache().to(self.device)

        with torch.no_grad():
            # 1. Prefill 
            outputs = self.forward(
                input_ids=input_ids,
                start_pos=0,
                past_key_values=past_key_values,
            )
            
            logits = outputs.logits[:, -1, :]
            next_token = sampler(logits)
            generated = torch.cat([generated, next_token], dim=-1)

            # 2. Decode
            for _ in range(max_new_tokens - 1):
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

                current_pos = past_key_values.seq_length

                outputs = self.forward(
                    input_ids=next_token,
                    start_pos=current_pos,
                    past_key_values=past_key_values,
                )

                logits = outputs.logits[:, -1, :]
                next_token = sampler(logits, input_ids=generated)
                generated = torch.cat([generated, next_token], dim=-1)

            return generated

    def compute_perplexity(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        '''
        Compute perplexity from logits and target tokens.

        Args:
            logits (torch.Tensor): Model output logits with shape [batch, seq_len, vocab_size].
            targets (torch.Tensor): Target token IDs with shape [batch, seq_len].

        Returns:
            torch.Tensor: Perplexity value (lower is better).
        '''
        batch_size, seq_len, vocab_size = logits.shape

        logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
        targets_flat = targets.reshape(batch_size * seq_len)

        loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')
        perplexity = torch.exp(loss)

        return perplexity