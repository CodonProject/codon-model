from codon.config import configclass


@configclass
class MotifChordConfig:
    # 语言主干
    vocab_size: int = 2**14
    model_dim: int = 768
    num_layers: int = 16
    num_heads: int = 8
    num_kv_heads: int = 2
    dropout: float = 0.1
    tie_weights: bool = True

    # 混合架构：16 层 = 12 GDN + 4 Gated GQA
    layer_pattern: tuple = ('gdn', 'gdn', 'gdn', 'gqa') * 4

    # GDN 专用
    gdn_head_k_dim: int = 64
    gdn_expand_v: float = 2.0

    # Gated GQA 专用
    use_attn_gate: bool = True

    # 视觉（DINOv3 ViT-B/16）
    vision_dim: int = 768
    vision_image_size: int = 224
    vision_num_tokens: int = 196

    # 音频（Whisper Tiny，30s mel = [80, 3000]）
    audio_dim: int = 384
    audio_num_mel_bins: int = 80
    audio_pool_stride: int = 8