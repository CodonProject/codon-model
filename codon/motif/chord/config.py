from codon.config import configclass


@configclass
class MotifChordConfig:
    # ===== 语言主干 =====
    vocab_size: int = 2**14
    model_dim: int = 768
    num_layers: int = 20
    num_heads: int = 8
    num_kv_heads: int = 2
    dropout: float = 0.1
    tie_weights: bool = True

    # ===== GDN 专用 =====
    gdn_head_k_dim: int = 64
    gdn_expand_v: float = 2.0

    # ===== Gated GQA 专用 =====
    use_attn_gate: bool = True

    # ===== 视觉（DINOv3 ViT-B/16） =====
    vision_dim: int = 768
    vision_patch_size: int = 16
    vision_image_size: int = 224
    vision_num_tokens: int = 196   # (224/16)^2

    # ===== 音频（Whisper Tiny） =====
    audio_dim: int = 384
    audio_num_mel_bins: int = 80
    audio_pool_stride: int = 8     # 30s → 187 token

    # ===== 连续向量模态（codon.j2 的 action / proprio / tactile 占位符） =====
    # 维度必须与数据一致：Session 按张量行数展开 pad，这里只决定投影塔的输入宽度。
    action_dim: int = 7            # 常见：6-DoF 位姿增量 + 1 维夹爪
    proprio_dim: int = 7           # 关节角 / 末端位姿
    tactile_dim: int = 32          # 触觉阵列读数
    vector_hidden_dim: int = 0     # 0 表示用 model_dim