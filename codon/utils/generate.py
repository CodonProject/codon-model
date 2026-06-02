from typing import Generator, List, Dict, Optional
from dataclasses import dataclass
import torch

from codon.utils.tokens  import PackedTokenizer
from codon.utils.session import Session
from codon.motif.base import CausalLanguageModel
from codon.motif.base import Sampler, KVCache


@dataclass
class ChatChunk:
    '''
    A data chunk returned during streaming generation.

    Attributes:
        content (str): The decoded text fragment from the current step.
        is_cot (bool): Whether the current fragment belongs to the Chain of Thought (thinking process).
        cot_ended (bool): Whether the thinking process has just ended, typically used to trigger UI rendering logic like line breaks.
    '''
    content: str
    is_cot: bool
    cot_ended: bool


def chat(
    model: CausalLanguageModel,
    tokenizer: PackedTokenizer,
    device: torch.device,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> Generator[ChatChunk, None, None]:
    '''
    Generates chat responses in a streaming fashion.

    This function supports Chain of Thought (CoT) state detection and incremental
    KV cache management for efficient decoding.

    Args:
        model (CausalLanguageModel): The causal language model used for text generation.
        tokenizer (PackedTokenizer): The tokenizer for encoding inputs and decoding outputs.
        device (torch.device): The device (CPU/CUDA) where the model computation is executed.
        messages (List[Dict[str, str]]): A list of dialogue messages, where each message is a dictionary containing 'role' and 'content'.
        max_new_tokens (int): The maximum number of new tokens to generate. Defaults to 1024.
        temperature (float): Sampling temperature. Defaults to 0.3.
        top_k (Optional[int]): The number of highest probability vocabulary tokens to keep for top-k filtering. Defaults to None.
        top_p (Optional[float]): Nucleus filtering probability threshold. Defaults to None.

    Yields:
        ChatChunk: Generated text chunks containing content and Chain of Thought states.
    '''
    model.eval()
    
    session = Session(tokenizer)
    session.add_messages(messages)
    session.add_generation_prompt(enable_thinking=True)
    
    tensors = session.to_tensors(device=device, batch_dim=True)
    input_ids = tensors['input_ids']

    sampler = Sampler(temperature=temperature, top_k=top_k, top_p=top_p)
    kv_cache = KVCache()

    generated = input_ids.clone()

    cot_start_id = tokenizer.token_to_id('[cot_start]')
    cot_end_id = tokenizer.token_to_id('[cot_end]')
    im_end_id = tokenizer.token_to_id('[im_end]')
    pad_id = tokenizer.token_to_id('[pad]')

    is_cot = True
    cot_ended = False

    with torch.no_grad():
        # Prefill
        outputs = model.forward(
            input_ids=input_ids,
            start_pos=0,
            past_key_values=None,
            use_cache=True
        )
        
        logits = outputs.logits[:, -1, :]
        
        next_token = sampler(logits, input_ids=generated)
        generated = torch.cat([generated, next_token], dim=-1)
        
        if outputs.past_key_values is not None:
            kv_cache.update(outputs.past_key_values)

        # Decode
        for _ in range(max_new_tokens - 1):
            token_val = next_token.item()

            if token_val == im_end_id or token_val == pad_id: break

            if token_val == cot_start_id:
                is_cot = True
                token_str = ''
            elif token_val == cot_end_id:
                is_cot = False
                cot_ended = True
                token_str = ''
            else:
                token_str = tokenizer.decode([token_val], skip_special_tokens=True)
                if token_str in ['[cot_start]', '[cot_end]', '[im_end]', '[im_start]']:
                    token_str = ''

            if token_str or cot_ended:
                yield ChatChunk(content=token_str, is_cot=is_cot, cot_ended=cot_ended)
                
            if cot_ended: cot_ended = False

            outputs = model.forward(
                input_ids=next_token,
                start_pos=kv_cache.current_len,
                past_key_values=kv_cache.states,
                use_cache=True
            )
            kv_cache.update(outputs.past_key_values)
            
            next_token = sampler(outputs.logits[:, -1, :], input_ids=generated)
            generated = torch.cat([generated, next_token], dim=-1)
