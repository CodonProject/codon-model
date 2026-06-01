from typing import Generator, List, Dict, Optional
from dataclasses import dataclass
import torch

from codon.motif.base import Sampler, KVCache
from codon.utils.tokens import PackedTokenizer
from codon.motif.base import CausalLanguageModel
from codon.utils.session import Session


@dataclass
class ChatChunk:
    '''
    流式生成中返回的数据块。
    
    Attributes:
        content (str): 本次解码出的文本片段。
        is_cot (bool): 当前片段是否属于思维链（思考过程）。
        cot_ended (bool): 是否刚刚结束思考（用于触发换行等界面渲染逻辑）。
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
    流式对话生成器，支持思维链状态检测与增量 KV 缓存管理。
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