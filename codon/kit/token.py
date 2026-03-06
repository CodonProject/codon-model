from tokenizers import Tokenizer, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers import normalizers
from tokenizers.trainers import BpeTrainer

core_tokens = ['[unk]', '[pad]', '[sep]']
chat_tokens = [
    '[im_start]', '[im_end]',
    '[system]', '[user]', '[model]', '[tool]', '[train]',
    '[interruption]', '[fim]',
]
reasoning_tokens = ['[cot_start]', '[cot_end]', '[verification]', '[solution]']
code_tokens = ['[fim_pre]', '[fim_mid]', '[fim_suf]']
tool_tokens = ['[tool_start]', '[tool_name]', '[tool_args]', '[tool_end]']

multimodal_tokens = [
    '[image_start]', '[image_end]', '[audio_start]', '[audio_end]', 
    '[video_start]', '[video_end]'
]

base_special_tokens = (
    core_tokens + 
    chat_tokens + 
    reasoning_tokens + 
    code_tokens + 
    tool_tokens + 
    multimodal_tokens
)

base_special_tokens += [f'[unused_{i}]' for i in range(len(base_special_tokens), 64)]
base_special_tokens += [f'[mask_{i}]' for i in range(32)]

chat_template = (
    "{% for message in messages %}"
        "{{ '[im_start]' }}"

        "{% if message['role'] == 'fim' %}"
            "{{ '[fim]' }}"
            "{{ '[fim_pre]' + message['prefix'] + '[fim_suf]' + message['suffix'] + '[fim_mid]' }}"
            
            "{% if message['middle'] %}"
                "{{ message['middle'] + '[im_end]' }}"
            "{% endif %}"
            
        "{% else %}"
            
            "{% if message['role'] in ['system', 'instruction'] %}"
                "{{ '[system]' }}"
            "{% elif message['role'] == 'user' %}"
                "{{ '[user]' }}"
            "{% elif message['role'] in ['assistant', 'model'] %}"
                "{{ '[model]' }}"
            "{% elif message['role'] == 'tool' %}"
                "{{ '[tool]' }}"
            "{% elif message['role'] == 'train' %}"
                "{{ '[train]' }}"
            "{% else %}"
                "{{ message['role'] }}"
            "{% endif %}"

            "{{ '\n' }}"

            "{% set thought_content = message['thought'] or message['reasoning_content'] %}"
            "{% if thought_content %}"
                "{{ '[cot_start]' + thought_content + '[cot_end]\n' }}"
            "{% else %}"
                "{{ '[cot_start][cot_end]\n' }}"
            "{% endif %}"
            
            "{% if message['content'] is defined and message['content'] is not none %}"
                "{% if message['content'] is string %}"
                    "{{ message['content'] }}"
                "{% else %}"
                    "{% for item in message['content'] %}"
                        "{% if item['type'] == 'text' %}"
                            "{{ item['text'] }}"
                        "{% elif item['type'] == 'image' %}"
                            "{{ '[image_start][image_end]' }}"
                        "{% elif item['type'] == 'audio' %}"
                            "{{ '[audio_start][audio_end]' }}"
                        "{% elif item['type'] == 'video' %}"
                            "{{ '[video_start][video_end]' }}"
                        "{% endif %}"
                    "{% endfor %}"
                "{% endif %}"
            "{% endif %}"
            
            "{% if message['tool_calls'] is defined and message['tool_calls'] %}"
                "{% for tool_call in message['tool_calls'] %}"
                    "{{ '[tool_start][tool_name]' + tool_call.function.name + '[tool_args]' + tool_call.function.arguments + '[tool_end]' }}"
                "{% endfor %}"
            "{% endif %}"
            
            "{{ '[im_end]\n' }}"
        "{% endif %}"
    "{% endfor %}"
    
    "{% if add_generation_prompt %}"
        "{% if messages[-1]['role'] != 'fim' %}"
            "{{ '[im_start][model]\n' }}"
            "{% if enable_thinking is defined and enable_thinking %}"
                "{{ '[cot_start]' }}"
            "{% elif enable_thinking is defined and not enable_thinking %}"
                "{{ '[cot_start][cot_end]\n' }}"
            "{% endif %}"
        "{% endif %}"
    "{% endif %}"
)


def create_tokenizer_trainer(
    unk_token: str='[unk]',
    vocab_size: int=32000,
    special_tokens: list[str]=base_special_tokens
) -> tuple:
    ''' 创建一个BPE Tokenizer训练器

    配置并返回一个用于训练BPE (Byte-Pair Encoding) 模型的分词器训练对象。
    该训练器预置了NFKC标准化、数字分割和字节级预分词处理。

    Args:
        unk_token (str): 未知Token的标识符。默认为 '[unk]'。
        vocab_size (int): 目标词表大小。默认为 32000。
        special_tokens (list[str] | tuple[str]): 特殊Token列表。
            默认为 base_special_tokens，包含核心、聊天、推理、代码、工具及多模态Token。

    Returns:
        tuple: (tokenizer, trainer) 
    '''
    tokenizer = Tokenizer(BPE(unk_token=unk_token))

    tokenizer.normalizer = normalizers.NFKC()

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            use_regex=True
        )
    ])

    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=32,
        min_frequency=10
    )

    return tokenizer, trainer
