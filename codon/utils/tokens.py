import json
import os
import zipfile
import copy
from dataclasses import dataclass
from tokenizers  import Tokenizer, pre_tokenizers, decoders
from tokenizers  import normalizers
from tokenizers.models   import BPE
from tokenizers.trainers import BpeTrainer

from transformers import PreTrainedTokenizerFast

from typing import Union, Optional, Generator, Any, List, Dict

from codon.builtin.mixins.remote import RemoteResourceMixin

@dataclass
class TokenizerTrainerResult:
    '''
    Result of the tokenizer trainer creation.

    Attributes:
        tokenizer (Tokenizer): The configured tokenizer instance.
        trainer (BpeTrainer): The configured BPE trainer instance.
    '''
    tokenizer: Tokenizer
    trainer: BpeTrainer

    def train_from_iterator(self, iter: Generator) -> 'TokenizerTrainerResult':
        self.tokenizer.train_from_iterator(iter, self.trainer)
        return self
    
    @property
    def packed_tokenizer(self) -> 'PackedTokenizer':
        return PackedTokenizer(
            tokenizer=self.tokenizer
        )


core_tokens = ['[pad]', '[unk]', '[sep]', '[cls]']
chat_tokens = [
    '[im_start]', '[im_end]',
    '[system]', '[user]', '[model]', '[tool]',
    '[interruption]', '[fim]',
]
reasoning_tokens = ['[cot_start]', '[cot_end]']
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
            
            "{% if message['role'] in ['system', 'instruction', 'developer'] %}"
                "{{ '[system]' }}"
            "{% elif message['role'] == 'user' %}"
                "{{ '[user]' }}"
            "{% elif message['role'] in ['assistant', 'model'] %}"
                "{{ '[model]' }}"
                "{% set thought_content = message['thought'] or message['reasoning_content'] %}"
                "{% if thought_content %}"
                    "{{ '[cot_start]' + thought_content + '[cot_end]' }}"
                "{% else %}"
                    "{{ '[cot_start][cot_end]' }}"
                "{% endif %}"
            "{% elif message['role'] == 'tool' %}"
                "{{ '[tool]' }}"
            "{% else %}"
                "{{ message['role'] }}"
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

            "{% if message['tools'] is defined and message['tools'] %}"
                "{{ message['tools'] }}"
            "{% endif %}"
            
            "{% if message['tool_calls'] is defined and message['tool_calls'] %}"
                "{% for tool_call in message['tool_calls'] %}"
                    "{{ '[tool_start][tool_name]' + tool_call.function.name + '[tool_args]' + tool_call.function.arguments + '[tool_end]' }}"
                "{% endfor %}"
            "{% endif %}"
            
            "{{ '[im_end]' }}"
        "{% endif %}"
    "{% endfor %}"
    
    "{% if add_generation_prompt %}"
        "{{ '[im_start][model]' }}"
        "{% if enable_thinking is defined and enable_thinking %}"
            "{{ '[cot_start]' }}"
        "{% elif enable_thinking is defined and not enable_thinking %}"
            "{{ '[cot_start][cot_end]' }}"
        "{% endif %}"
    "{% endif %}"
)


def create_tokenizer_trainer(
    unk_token: str='[unk]',
    vocab_size: int=32000,
    special_tokens: list[str]=base_special_tokens
) -> TokenizerTrainerResult:
    '''
    Creates a BPE Tokenizer trainer.

    Configures and returns a tokenizer trainer object for training BPE (Byte-Pair Encoding) models.
    The trainer is pre-configured with NFKC normalization, digit splitting, and byte-level pre-tokenization.

    Args:
        unk_token (str): Identifier for unknown tokens. Defaults to '[unk]'.
        vocab_size (int): Target vocabulary size. Defaults to 32000.
        special_tokens (list[str]): List of special tokens.
            Defaults to base_special_tokens, including core, chat, reasoning, code, tool, and multimodal tokens.

    Returns:
        TokenizerTrainerResult: A dataclass containing the tokenizer and trainer instances.
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

    return TokenizerTrainerResult(tokenizer=tokenizer, trainer=trainer)


class PackedTokenizer(RemoteResourceMixin):
    def __init__(self, tokenizer: Optional[Union[Tokenizer, str]] = None):
        self._tokenizer: Optional[Tokenizer] = None
        self._fast_tokenizer: Optional[PreTrainedTokenizerFast] = None
        self.config = {}
        self.template = chat_template

        self.safe_escape = '[unused_42]'
        self.safe_escape_id: int = None

        if isinstance(tokenizer, str):
            self.load(tokenizer)
        elif isinstance(tokenizer, Tokenizer):
            self._tokenizer = tokenizer
            self.config = {
                'unk_token': '[unk]',
                'pad_token': '[pad]',
                'bos_token': '[im_start]',
                'eos_token': '[im_end]',
            }
            self._update_fast_tokenizer()

    def _update_fast_tokenizer(self) -> None:
        '''
        Updates the cached PreTrainedTokenizerFast instance.
        '''
        if self._tokenizer is None:
            self._fast_tokenizer = None
            return

        self._fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self._tokenizer,
            unk_token=self.config.get('unk_token', '[unk]'),
            pad_token=self.config.get('pad_token', '[pad]'),
            bos_token=self.config.get('bos_token', '[im_start]'),
            eos_token=self.config.get('eos_token', '[im_end]'),
            chat_template=self.template,
            clean_up_tokenization_spaces=False
        )
    
    def set_chat_template(self, template: str) -> 'PackedTokenizer':
        if not isinstance(template, str):
            raise TypeError(f'template must be str, got {type(template).__name__}')
        self.template = template
        self._update_fast_tokenizer()
        return self

    def reset_chat_template(self) -> 'PackedTokenizer':
        self.template = chat_template
        self._update_fast_tokenizer()
        return self

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            raise ValueError("Tokenizer is not loaded.")
        return self._tokenizer
    
    @property
    def fast_tokenizer(self) -> PreTrainedTokenizerFast:
        if self._fast_tokenizer is None:
            raise ValueError('Tokenizer is not loaded.')
        return self._fast_tokenizer
    
    def token_to_id(self, token: str) -> Optional[int]:
        return self._tokenizer.token_to_id(token)
    
    def ensure_escape(self) -> int:
        if self.safe_escape_id is None:
            tid = self._tokenizer.token_to_id(self.safe_escape)
            if tid is None:
                raise ValueError(f'Escape token {self.safe_escape} not found in vocab.')
            self.safe_escape_id = tid
        return self.safe_escape_id
    
    def _sanitize_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return content.replace(']', f'{self.safe_escape}]')
        elif isinstance(content, list):
            return [
                {**item, 'text': self._sanitize_content(item['text'])} if item.get('type') == 'text' else item 
                for item in content
            ]
        
    def apply_chat_template(
        self, 
        messages: List[Dict[str, Any]], 
        add_generation_prompt: bool = True,
        **kwargs
    ) -> List[int]:
        escape_id = self.ensure_escape()
        
        safe_messages = copy.deepcopy(messages)
        for msg in safe_messages:
            if 'content' in msg:
                msg['content'] = self._sanitize_content(msg['content'])
        
        kwargs.pop('tokenize', None)
        
        raw = self.fast_tokenizer.apply_chat_template(
            safe_messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            **kwargs
        )
        
        if isinstance(raw, dict) or hasattr(raw, 'input_ids'):
            raw = raw['input_ids']
        if raw and isinstance(raw[0], list):
            raw = raw[0]
        
        return [tid for tid in raw if tid != escape_id]

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs) -> List[int]:
        escape_id = self.ensure_escape()

        safe_text = text.replace(']', f'{self.safe_escape}]')
        
        raw_ids = self.fast_tokenizer.encode(safe_text, add_special_tokens=add_special_tokens, **kwargs)
        
        return [tid for tid in raw_ids if tid != escape_id]
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        return self.fast_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def save(self, path: str) -> 'PackedTokenizer':
        if self._tokenizer is None:
            raise ValueError('No tokenizer to save.')

        with zipfile.ZipFile(path, 'w') as z:
            # Save tokenizer.json
            z.writestr('tokenizer.json', self._tokenizer.to_str())
            
            # Save tokenizer_config.json
            z.writestr('tokenizer_config.json', json.dumps(self.config, indent=2))
            
            # Save chat_template.jinja
            z.writestr('chat_template.jinja', self.template)
            
        return self
    
    def load(self, path: str) -> 'PackedTokenizer':
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with zipfile.ZipFile(path, 'r') as z:
            file_list = z.namelist()
            
            # Helper to find file in zip (ignoring directory prefix if any)
            def find_file(name):
                for f in file_list:
                    if f == name or f.endswith(f'/{name}'):
                        return f
                return None

            # Load tokenizer.json
            tokenizer_file = find_file('tokenizer.json')
            if tokenizer_file:
                tokenizer_json = z.read(tokenizer_file).decode('utf-8')
                self._tokenizer = Tokenizer.from_str(tokenizer_json)
            else:
                raise ValueError("tokenizer.json not found in zip file")

            # Load tokenizer_config.json
            config_file = find_file('tokenizer_config.json')
            if config_file:
                config_json = z.read(config_file).decode('utf-8')
                self.config = json.loads(config_json)

            # Load chat_template.jinja
            template_file = find_file('chat_template.jinja')
            if template_file:
                self.template = z.read(template_file).decode('utf-8')

        self._update_fast_tokenizer()
        return self
