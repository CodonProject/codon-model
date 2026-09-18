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
        # Tokenizer safe escape
        '<|pad|>', '<|unk|>', '<|sep|>', '<|mask|>', '<|bos|>', '<|eos|>', '<|endoftext|>',
        '<|safe_escape|>',
        # IM token
        '<|im_start|>', '<|im_end|>',
        # Role token
        '<|system|>', '<|model|>', '<|user|>', '<|environment|>', '<|tool_response|>',
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
        '<|frame_start|>', '<|frame_end|>', '<|frame_sep|>',
        '<|timestamp_start|>', '<|timestamp_end|>',
        # Action token
        '<|action_start|>', '<|action_pad|>','<|action_end|>',
        '<|action_chunk_start|>', '<|action_sep|>', '<|action_chunk_end|>',
        '<|proprio_start|>', '<|proprio_pad|>', '<|proprio_end|>',
        '<|trajectory_start|>', '<|trajectory_end|>',
        '<|waypoint_start|>', '<|waypoint_end|>',
        '<|action_terminate|>', '<|action_continue|>',
        # Observation token
        '<|camera_start|>', '<|camera_end|>', '<|camera_id|>',
        '<|tactile_start|>', '<|tactile_end|>', '<|tactile_pad|>',
        '<|force_start|>', '<|force_end|>',
        # Grounding token
        '<|bbox_start|>', '<|bbox_end|>',
        '<|point_start|>', '<|point_end|>',
        '<|ref_start|>', '<|ref_end|>',
        '<|grasp_start|>', '<|grasp_end|>',
        # FIM token
        '<|fim_prefix|>', '<|fim_suffix|>', '<|fim_middle|>',
    ]
}

_spec = LM['spec_token']
assert len(_spec) == len(set(_spec)), f'Error: {sorted({t for t in _spec if _spec.count(t) > 1})}'