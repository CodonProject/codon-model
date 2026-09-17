import os
from typing import Optional


DIR_PATH = os.path.dirname(os.path.abspath(__file__))

EFFORT_LEVELS = ('low', 'high', 'max')
EFFORT_DEFAULT = 'high'
EFFORT_TOKEN = {level: f'<|effort_{level}|>' for level in EFFORT_LEVELS}

EFFORT_ALIASES = {
    'minimal': 'low',
    'low':     'low',
    'medium':  'high',
    'high':    'high',
    'xhigh':   'high',
    'max':     'max',
    'ultra':   'max',
}


def resolve_effort(effort: Optional[str] = None) -> str:
    alias = str(EFFORT_DEFAULT if effort is None else effort).strip().lower() or EFFORT_DEFAULT
    if alias not in EFFORT_ALIASES:
        raise ValueError(
            f'unknown effort level: {effort!r} (expected one of {tuple(EFFORT_ALIASES)})'
        )
    return EFFORT_ALIASES[alias]


LM = {
    'jinja': os.path.join(DIR_PATH, 'codon.j2'),
    'effort': EFFORT_TOKEN,
    'effort_default': EFFORT_DEFAULT,
    'effort_aliases': EFFORT_ALIASES,
    'spec_token': [
        # IM token
        '<|im_start|>', '<|im_end|>',
        # Role token
        '<|system|>', '<|model|>', '<|user|>', '<|tool_response|>',
        # CoT token
        '<|thought_start|>',
        '<|effort_low|>', '<|effort_high|>', '<|effort_max|>',
        '<|thought_end|>',
        # Tool Call token
        '<|tool_call_start|>', '<|tool_call_end|>', '<|tool_name_divider|>',
        # Multimodal token
        '<|modality_image_start|>', '<|modality_image_pad|>', '<|modality_image_end|>',
        '<|modality_audio_start|>', '<|modality_audio_pad|>', '<|modality_audio_end|>',
        '<|modality_video_start|>', '<|modality_video_pad|>', '<|modality_video_end|>',
        # FIM token
        '<|fim_prefix|>', '<|fim_suffix|>', '<|fim_middle|>',
        # Tokenizer safe escape
        '<|safe_escape|>',
        '<|pad|>', '<|unk|>', '<|sep|>'
    ]
}
