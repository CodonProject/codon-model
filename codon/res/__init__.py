import os

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

LM = {
    'jinja': os.path.join(DIR_PATH, 'codon.j2'),
    'spec_token': [
        # IM token
        '<|im_start|>', '<|im_end|>',
        # Role token
        '<|system|>', '<|model|>', '<|user|>', '<|tool_response|>',
        # CoT token
        '<|thought_start|>', '<|thought_end|>',
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
