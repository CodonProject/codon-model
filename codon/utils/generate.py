from typing import Generator, List, Dict, Optional, Any
from dataclasses import dataclass
import torch

from codon.utils.tokens  import PackedTokenizer
from codon.utils.session import Session
from codon.utils.media   import (
    modal_config,
    model_modalities,
    normalize_messages,
    has_video_frames,
)
from codon.model.types.language import CausalLanguageModel
from codon.model.sampler import Sampler
from codon.model.cache import ModelCache
from codon.model.grammar import constraint_from_response_format


@dataclass
class ChatChunk:
    '''
    A data chunk returned during streaming generation.

    Attributes:
        content (str): The decoded text fragment from the current step.
        is_cot (bool): Whether the current fragment belongs to the Chain of Thought (thinking process).
        cot_ended (bool): Whether the thinking process has just ended, typically used to trigger UI rendering logic like line breaks.
        finish_reason (str, optional): 只在最后一块上给出：'stop'（正常结束；开了强制 JSON 时表示
            JSON 已完整）或 'length'（token 预算用尽 / 没能产出完整 JSON，调用方应加大预算重试）。
    '''
    content: str
    is_cot: bool
    cot_ended: bool
    finish_reason: Optional[str] = None


def chat(
    model: CausalLanguageModel,
    tokenizer: PackedTokenizer,
    device: torch.device,
    messages: List[Dict[str, Any]],
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    enable_thinking: bool = True,
    response_format: Optional[Dict[str, Any]] = None,
    patch_size: Optional[int] = None,
    audio_pool_stride: Optional[int] = None,
    num_mel_bins: Optional[int] = None,
) -> Generator[ChatChunk, None, None]:
    '''
    Generates chat responses in a streaming fashion.

    This function supports Chain of Thought (CoT) state detection and incremental
    KV cache management for efficient decoding.

    Args:
        model (CausalLanguageModel): The causal language model used for text generation.
        tokenizer (PackedTokenizer): The tokenizer for encoding inputs and decoding outputs.
        device (torch.device): The device (CPU/CUDA) where the model computation is executed.
        messages (List[Dict[str, Any]]): A list of dialogue messages, where each message is a
            dictionary containing 'role' and 'content'. `content` 可以是字符串，也可以是
            codon.j2 风格的多模态列表：
                [{'type': 'text', 'text': ...},
                 {'type': 'image', 'image': <tensor / 路径 / base64 / data URL / PIL / ndarray>},
                 {'type': 'audio', 'audio': <mel 张量 / .npy / .pt / wav 波形>}]
            OpenAI 风格的 `image_url` / `input_audio` item 同样接受，媒体载荷由
            `codon.utils.media` 解码成张量；模型没有声明对应模态能力时抛
            `codon.utils.media.UnsupportedModalityError`。
        max_new_tokens (int): The maximum number of new tokens to generate. Defaults to 1024.
        temperature (float): Sampling temperature. Defaults to 0.3.
        top_k (Optional[int]): The number of highest probability vocabulary tokens to keep for top-k filtering. Defaults to None.
        top_p (Optional[float]): Nucleus filtering probability threshold. Defaults to None.
        enable_thinking (bool): Whether to open a [cot_start] section before the content. Defaults to True.
            词表里没有「思考段结束」token 时会自动退化成不思考（直接从答案开头开始生成/约束）。
        response_format (dict, optional): OpenAI 风格的强制 JSON 配置：
            `{'type': 'json_object'}` / `{'type': 'json'}` / `{'type': 'json_array'}` /
            `{'type': 'json_schema', 'json_schema': {'schema': {...}}}` / `{'type': 'text'}`。
            开启后**只在答案段**（[cot_end] / <|thought_end|> 之后）生效：思考过程自由生成，
            答案段被语法约束成合法 JSON（给定 schema 时还会约束键名 / 必填 / 类型 / enum）。
            非法配置抛 ValueError / UnsupportedSchemaError。
        patch_size (int, optional): 图像占位符展开用的 patch 尺寸。默认取模型的
            `image_patch_size`（MotifChord / DINOv3 是 16），纯文本模型退回 12。
        audio_pool_stride (int, optional): 音频占位符展开用的池化步长，默认取模型的
            `audio_pool_stride`（MotifChord / Whisper Tiny + pool 8 是 8）。
        num_mel_bins (int, optional): 波形转 mel 的维数，默认取模型的 `audio_num_mel_bins`（80）。

    Notes:
        特殊 token 一律按逻辑名经 `Session` 解析，因此 A1 词表（`[cot_end]` / `[im_end]`）与
        A2 / chord 词表（`<|thought_end|>` / `<|tool_name_divider|>` / `<|im_end|>`）都适用。
        强制 JSON 时工具调用、模态、分隔符等结构 token 都在约束词表之外，答案只可能是纯 JSON。

        多模态只在 prefill 阶段注入（`x[b, *_patch_indices] = 模态特征`，序列长度不变），
        decode 步只喂新 token；与 `CausalLanguageModel.forward` 的占位符约定一致。

    Yields:
        ChatChunk: Generated text chunks containing content and Chain of Thought states.
    '''
    model.eval()

    capabilities = model_modalities(model)
    default_patch_size, default_audio_stride, default_mel_bins = modal_config(model)
    patch_size = default_patch_size if patch_size is None else int(patch_size)
    audio_pool_stride = default_audio_stride if audio_pool_stride is None else int(audio_pool_stride)
    num_mel_bins = default_mel_bins if num_mel_bins is None else int(num_mel_bins)

    # 媒体载荷 -> 张量；模型没声明能力时在这里就报错，而不是把 unk token 喂给模型。
    messages = normalize_messages(
        messages,
        image_capab=capabilities['image'],
        audio_capab=capabilities['audio'],
        video_capab=capabilities['video'],
        num_mel_bins=num_mel_bins,
    )

    session = Session(
        tokenizer,
        patch_size=patch_size,
        audio_pool_stride=audio_pool_stride,
        # 逐帧视频（video + frames）复用图像塔，模板需要 video_capab 才会渲染帧
        video_capab=has_video_frames(messages),
    )

    # 特殊 token 一律按 Session 的逻辑名解析：A1 词表是 [cot_end] / [im_end]，
    # A2（codon/res 风格，如 chord.zip）是 <|thought_end|> / <|im_end|>，两种都能用。
    cot_start_id = session.token_id('cot_start')
    cot_end_id = session.token_id('cot_end')
    im_end_id = session.token_id('im_end')
    pad_id = session.token_id('pad')
    special_ids = session.special_ids()

    if im_end_id is None:                      # 没有 im_end 的词表退回到配置里的 eos
        eos_name = tokenizer.token_eos
        im_end_id = tokenizer.token_to_id(eos_name) if eos_name else None
        if im_end_id is not None:
            special_ids.add(im_end_id)

    # 词表里没有「思考段结束」token 时不存在可跳过的思考段：直接从答案开头就约束。
    thinking = bool(enable_thinking and cot_end_id is not None)

    session.add_messages(messages)
    session.add_generation_prompt(enable_thinking=thinking)
    
    tensors = session.to_tensors(device=device, batch_dim=True)
    input_ids = tensors['input_ids']

    # 只有真的带了模态才把模态关键字传给 forward：纯文本模型的 forward 可能不吃这些参数。
    modal_kwargs: Dict[str, Any] = {}
    if len(tensors['images']) > 0:
        modal_kwargs['images'] = tensors['images']
        modal_kwargs['image_patch_indices'] = tensors['image_patch_indices']
    if len(tensors['audios']) > 0:
        modal_kwargs['audios'] = tensors['audios']
        modal_kwargs['audio_patch_indices'] = tensors['audio_patch_indices']
    # 连续向量模态（动作 / 本体感觉 / 触觉）：prefill 时就地写入占位符位置
    for name in ('action', 'proprio', 'tactile'):
        items = tensors[f'{name}s']
        if len(items) > 0:
            modal_kwargs[f'{name}s'] = items
            modal_kwargs[f'{name}_patch_indices'] = tensors[f'{name}_patch_indices']
    if modal_kwargs:
        modal_kwargs['mask'] = tensors['attention_mask']

    sampler = Sampler(temperature=temperature, top_k=top_k, top_p=top_p)
    
    kv_cache = ModelCache()
    kv_cache.to(device)

    generated = input_ids.clone()

    # 强制 JSON：约束对象只在答案段绑定到采样器上，思考段保持自由生成。
    constraint = constraint_from_response_format(
        tokenizer, response_format, eos_token_id=im_end_id
    )

    is_cot = thinking
    cot_ended = False

    with torch.no_grad():
        # Prefill（多模态在这里注入）
        outputs = model.forward(
            input_ids=input_ids,
            start_pos=0,
            past_key_values=kv_cache,
            **modal_kwargs,
        )
        
        logits = outputs.logits[:, -1, :]
        
        sampler.constraint = constraint if (constraint is not None and not is_cot) else None
        next_token = sampler(logits, input_ids=generated)
        generated = torch.cat([generated, next_token], dim=-1)
        
        if outputs.past_key_values is not None:
            kv_cache = outputs.past_key_values

        # Decode
        stopped = False
        for _ in range(max_new_tokens - 1):
            token_val = next_token.item()

            if token_val == im_end_id or token_val == pad_id:
                stopped = True
                break

            if token_val == cot_start_id:
                is_cot = True
                token_str = ''
            elif token_val == cot_end_id:
                is_cot = False
                cot_ended = True
                token_str = ''
            elif token_val in special_ids:
                token_str = ''                       # 结构 token（工具/模态/分隔符）不算正文
            else:
                token_str = tokenizer.decode([token_val], skip_special_tokens=True)

            if token_str or cot_ended:
                yield ChatChunk(content=token_str, is_cot=is_cot, cot_ended=cot_ended)
                
            if cot_ended: cot_ended = False

            current_pos = 0
            if len(kv_cache.layer_caches) > 0:
                first_cache = next(iter(kv_cache.layer_caches.values()))
                current_pos = first_cache.seq_length

            outputs = model.forward(
                input_ids=next_token,
                start_pos=current_pos,
                past_key_values=kv_cache,
            )
            if outputs.past_key_values is not None:
                kv_cache = outputs.past_key_values

            sampler.constraint = constraint if (constraint is not None and not is_cot) else None
            next_token = sampler(outputs.logits[:, -1, :], input_ids=generated)
            generated = torch.cat([generated, next_token], dim=-1)

    # 最后一块只带结束原因：强制 JSON 时 'stop' 表示文档已完整，'length' 表示被预算截断。
    complete = stopped and (constraint is None or constraint.is_complete)
    yield ChatChunk(content='', is_cot=False, cot_ended=False,
                    finish_reason='stop' if complete else 'length')
