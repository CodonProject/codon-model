import json
import os
import zipfile
import copy
from dataclasses import dataclass
from tokenizers  import Tokenizer, pre_tokenizers, decoders, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from transformers import PreTrainedTokenizerFast

from typing import Union, Optional, Generator, Any, List, Dict

from codon.mixins import RemoteResourceMixin


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


def create_tokenizer_trainer(
    unk_token: str='[unk]',
    vocab_size: int=32000,
    special_tokens: list[str]=[],
    use_norm: bool = True,
    use_nfc: bool = True
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

    if use_norm:
        tokenizer.normalizer = normalizers.NFC() if use_nfc else normalizers.NFKC()

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
    def __init__(
        self, 
        tokenizer: Optional[Union[Tokenizer, str]] = None,
        safe_escape: str = '[unused_42]',
        safe_rules: Optional[Dict[str, str]] = None
    ):
        self._tokenizer: Optional[Tokenizer] = None
        self._fast_tokenizer: Optional[PreTrainedTokenizerFast] = None
        self.config = {}
        self.template = ''
        
        self.extra_files: Dict[str, bytes] = {}

        self.safe_escape = safe_escape
        self.safe_escape_id: int = None
        
        self.safe_rules: Dict[str, str] = safe_rules or {
            ']': '{safe}]',
            '[': '[{safe}',
            '|': '{safe}|{safe}'
        }

        if isinstance(tokenizer, str):
            self.load(tokenizer)
        elif isinstance(tokenizer, Tokenizer):
            self._tokenizer = tokenizer
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
            unk_token=self.config.get('unk_token'),
            pad_token=self.config.get('pad_token'),
            bos_token=self.config.get('bos_token'),
            eos_token=self.config.get('eos_token'),
            chat_template=self.template,
            clean_up_tokenization_spaces=False
        )
    
    def set_chat_template(self, template: str) -> 'PackedTokenizer':
        if not isinstance(template, str):
            raise TypeError(f'template must be str, got {type(template).__name__}')
        self.template = template
        self._update_fast_tokenizer()
        return self

    def set_eos(self, eos: str) -> bool:
        if self._tokenizer.token_to_id(eos) is not None:
            self.config['eos_token'] = eos
            return True
        return False

    @property
    def token_eos(self) -> Optional[str]:
        return self.config.get('eos_token', None)

    def set_pad(self, pad: str) -> bool:
        if self._tokenizer.token_to_id(pad) is not None:
            self.config['pad_token'] = pad
            return True
        return False

    @property
    def token_pad(self) -> Optional[str]:
        return self.config.get('pad_token', None)

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            raise ValueError("Tokenizer is not loaded.")
        return self._tokenizer
    
    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()
    
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

    def _apply_safe_rules(self, text: str) -> str:
        for old, new in self.safe_rules.items():
            text = text.replace(old, new.format(safe=self.safe_escape))
        return text
    
    def _sanitize_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return self._apply_safe_rules(content)
        elif isinstance(content, list):
            return [
                {**item, 'text': self._sanitize_content(item['text'])} if item.get('type') == 'text' else item 
                for item in content
            ]
        return content
        
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

    def encode(self, text: str, add_special_tokens: bool = False, apply_safe_rule: bool = True, **kwargs) -> List[int]:
        escape_id = self.ensure_escape()

        safe_text = self._apply_safe_rules(text) if apply_safe_rule else text
        
        raw_ids = self.fast_tokenizer.encode(safe_text, add_special_tokens=add_special_tokens, **kwargs)
        
        return [tid for tid in raw_ids if tid != escape_id]
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        return self.fast_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def add_file(self, name: str, data: Union[str, bytes]) -> 'PackedTokenizer':
        if isinstance(data, str):
            data = data.encode('utf-8')
        self.extra_files[name] = data
        return self

    def get_file(self, name: str) -> Optional[bytes]:
        return self.extra_files.get(name)

    def save(self, path: str) -> 'PackedTokenizer':
        if self._tokenizer is None:
            raise ValueError('No tokenizer to save.')

        reserved_files = {
            'tokenizer.json', 
            'tokenizer_config.json', 
            'chat_template.jinja', 
            'safe_rules.json', 
            'safe_escape.txt'
        }

        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
            # Save tokenizer.json
            z.writestr('tokenizer.json', self._tokenizer.to_str())
            
            # Save tokenizer_config.json
            z.writestr('tokenizer_config.json', json.dumps(self.config, indent=2))
            
            # Save chat_template.jinja
            z.writestr('chat_template.jinja', self.template)
            
            # Save safe rules and escape token
            z.writestr('safe_rules.json', json.dumps(self.safe_rules, indent=2))
            z.writestr('safe_escape.txt', self.safe_escape)

            # Save extra arbitrary files
            for name, data in self.extra_files.items():
                if name in reserved_files:
                    raise ValueError(f'Cannot use reserved file name for extra files: {name}')
                z.writestr(name, data)
            
        return self
    
    def load(self, path: str) -> 'PackedTokenizer':
        if not os.path.exists(path):
            raise FileNotFoundError(f'File not found: {path}')

        reserved_files = {
            'tokenizer.json', 
            'tokenizer_config.json', 
            'chat_template.jinja', 
            'safe_rules.json', 
            'safe_escape.txt'
        }

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

            # Load safe_rules.json
            rules_file = find_file('safe_rules.json')
            if rules_file:
                rules_json = z.read(rules_file).decode('utf-8')
                self.safe_rules = json.loads(rules_json)

            # Load safe_escape.txt
            escape_file = find_file('safe_escape.txt')
            if escape_file:
                self.safe_escape = z.read(escape_file).decode('utf-8')
                self.safe_escape_id = None  # Reset cache

            # Load extra arbitrary files
            self.extra_files = {}
            for f in file_list:
                base_name = f.split('/')[-1]
                if base_name and base_name not in reserved_files:
                    self.extra_files[f] = z.read(f)

        self._update_fast_tokenizer()
        return self