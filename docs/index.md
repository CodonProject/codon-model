# Codon Model Documentation

Codon is a PyTorch toolbox designed for research-driven prototyping and experimentation. It focuses on rapidly transforming algorithmic ideas into trainable end-to-end modules while maintaining a balance between engineering rigor and reproducibility.

## Overview

Codon is **not** a general-purpose large model training framework. Instead, it prioritizes the following three core workflows:

### 1. Prototyping
Compose complete Transformer / MoE / multimodal models from composable `block/` components and low-level `ops/` primitives, and run end-to-end training on a single GPU.

- Quickly implement and validate novel attention, normalization, or gating mechanisms within a standard Transformer or ConvNet backbone;
- Use modular blocks like `TransformerDenseDecoder`, `RMSNorm`, and `SwiGLU` to assemble and benchmark new architectures with minimal boilerplate.

### 2. Rapid Modeling
Leverage the pre-defined model families in `motif/` and the `BasicModel` base class to focus your efforts on core algorithms rather than engineering details like weight loading, gradient checkpointing, or parameter statistics.

- Start from ready-to-use causal language models (e.g., `MotifA1` with GQA) or vision autoencoders (`MotifV1` with LFQ);
- Utilize `BasicModel` for unified gradient checkpointing, human-readable parameter counting, and `.safetensors` persistence with fine-grained filtering rules.

### 3. Experimental Research
Conduct frontier algorithmic exploration using non-standard operators such as `ops/bio.py` (Hebbian/BCM/STDP learning rules), `ops/manifold.py` (vMF Riemannian manifold convolution with Triton acceleration), and `ops/fourier.py` (causal spectral mixing), without pulling in heavy external dependencies.

- Study biological-plausible learning rules directly within a deep learning pipeline;
- Experiment with hyperbolic or spherical projections via vMF concentration parameters and gravity-based attraction on manifolds;
- Benchmark $O(L \log L)$ spectral mixing as a replacement for traditional quadratic attention.

### Additional Capabilities

Codon also provides first-class support for:
- **Multimodal (Text + Image) experiments**: Build small-scale models for pretraining and fine-tuning across modalities using `session.py`, `fusion.py`, and `data/` utilities;
- **Advanced techniques**: Run controlled experiments on sparse activation (MoE), low-rank adaptation (LoRA), lookup-free quantization (LFQ), and positional encodings (RoPE / Sinusoidal);
- **Long-context training**: Use `utils/plan.py` (Chinchilla-based scheduler), `utils/session.py` (dynamic masked loss), and `utils/data/` (chunked streaming datasets) to manage complex training pipelines.

## Module Structure

```
codon/
├── block/          # Neural network building blocks
│   ├── attention.py     # MultiHeadAttention / MultiHeadFourier
│   ├── transformer.py   # Transformer Dense & MoE Decoder
│   ├── mlp.py           # MLP (with SwiGLU)
│   ├── conv.py          # CausalConv1d, DepthwiseSeparableConv
│   ├── embedding.py     # Sinusoidal / Rotary
│   ├── moe.py           # Mixture of Experts
│   ├── lora.py          # Low-Rank Adaptation (Linear / Conv / Embedding)
│   ├── film.py          # Feature-wise Linear Modulation
│   ├── fusion.py        # Multimodal fusion
│   ├── codebook.py      # Lookup-free quantization
│   ├── pixelshuffle.py  # PixelShuffle up/down-sampling
│   ├── adanorm.py       # Condition-modulated normalization
│   └── norm.py          # RMSNorm / Zero-Centered RMSNorm
├── motif/          # Pre-built model families
│   ├── motif_a1.py      # MotifA1 Causal Language Model
│   ├── motif_v1.py      # MotifV1 Vision Autoencoder
│   └── base.py          # CausalLanguageModel / AutoencoderVisionModel
├── ops/            # Low-level operators and experimental features
│   ├── attention.py     # apply_attention
│   ├── bio.py           # Hebbian / Oja / BCM / STDP learning rules
│   ├── fourier.py       # Causal Spectral Mixing
│   ├── pixelshuffle.py  # Space-to-channel reversible transforms
│   └── manifold/        # Riemannian manifold operators (vMF + Gravity)
├── model/          # Additional model families
│   ├── resnet.py        # ResNet implementations
│   ├── tcn.py           # Temporal Convolutional Network
│   └── patch_disc.py    # PatchGAN Discriminator
├── kit/            # Training and deployment utilities
│   └── train/           # Training loops and context scheduling
└── utils/          # Infrastructure and experimental helpers
    ├── tokens.py        # PackedTokenizer for text packing
    ├── seed.py          # Random seed management
    ├── info.py          # Runtime environment detection
    ├── generate.py      # Sampling / streaming generation
    ├── service.py       # OpenAI-compatible FastAPI inference service
    ├── session.py       # Multimodal dialogue Session with dynamic masking
    ├── plan.py          # Chinchilla context training planner
    ├── theta.py         # RoPE theta validation
    ├── mask.py          # Attention and loss masking
    ├── lifecycle.py     # Callback and resource lifecycle management
    ├── data/            # CodonDataset / ChunkedTokenStream / FlatData / Image
    └── eval/            # ConfusionMap / TSNEMap / RSA / GradCAM
```

## Documentation Contents

- [Base Classes](base.md)
  - [BasicModel](base.md) - Universal base class with gradient checkpointing, parameter statistics, and safetensors loading/filtering
  - [RemoteResourceMixin](base.md) - Automatic HuggingFace / ModelScope resource caching with best-node selection

- [Block Modules](block/)
  - [Attention](block/attention.md) - MultiHeadAttention / MultiHeadAttentionKEV / MultiHeadFourier
  - [Transformer](block/transformer.md) - TransformerDenseDecoder / TransformerMoEDecoder
  - [MLP](block/mlp.md) - MLP and SwiGLU
  - [Convolution](block/conv.md) - CausalConv1d, ConvBlock, DepthwiseSeparableConv
  - [Pixel Shuffle](block/pixelshuffle.md) - PixelShuffleUpSample / UnPixelShuffleDownSample
  - [Embedding](block/embedding.md) - SinusoidalEmbedding / RotaryEmbedding
  - [Normalization](block/norm.md) - RMSNorm / ZCRMSNorm
  - [Adaptive Norm](block/adanorm.md) - AdaLayerNorm
  - [MoE](block/moe.md) - Mixture of Experts
  - [LoRA](block/lora.md) - LinearLoRA / ConvLoRA / EmbeddingLoRA
  - [FiLM](block/film.md) - Feature-wise Linear Modulation
  - [Fusion](block/fusion.md) - Multimodal fusion
  - [Codebook](block/codebook.md) - LookupFreeQuantization

- [Motif Models](motif/)
  - [MotifA1](motif/motif_a1.md) - Causal language model with GQA and remote loading
  - [MotifV1](motif/motif_v1.md) - Vision autoencoder with LFQ
  - [Base Classes](motif/base.md) - CausalLanguageModel / AutoencoderVisionModel

- [Operations](ops/)
  - [Attention Operations](ops/attention.md) - apply_attention
  - [Bio-inspired Operations](ops/bio.md) - Hebbian / Oja / BCM / STDP
  - [Fourier Operations](ops/fourier.md) - apply_fourier_mixing
  - [Pixel Shuffle Operations](ops/pixelshuffle.md) - pixel_shuffle / unpixel_shuffle
  - [Manifold Operations](ops/manifold.md) - Riemannian manifold operators (vMF + Gravity + Triton)

- [Utilities](utils/)
  - [Tokenization](utils/tokens.md) - PackedTokenizer
  - [Seed Management](utils/seed.md) - seed_everything
  - [System Information](utils/info.md) - SystemEnvironment
  - [Generation](utils/generate.md) - Streaming generation
  - [Service](utils/service.md) - OpenAI-compatible FastAPI service
  - [Session](utils/session.md) - Multimodal dialogue Session with dynamic masking
  - [Training Plan](utils/plan.md) - Context training planner (Foundation / Expansion / Stabilization)
  - [Utility Functions](utils/utils.md) - RoPE theta validation / attention masking / lifecycle callbacks
  - [Datasets](utils/data.md) - CodonDataset / FlatDataset / ImageDataset / ChunkedTokenStream
  - [Evaluation](utils/eval.md) - ConfusionMap / TSNEMap / RSAMap / GradCAM

- [Model Zoo](model/)
  - [ResNet](model/resnet.md)
  - [TCN](model/tcn.md)
  - [Patch Discriminator](model/patch_disc.md) - PatchGAN Discriminator

- [Training Kit](kit/)
  - [Language Training](kit/language.md)
  - [Vision Training](kit/vision.md)

- [Developer Tools](dev/cli.md)
  - [CLI](dev/cli.md) - `codon hash`, `codon clear`

## Installation

```bash
pip install codon-model
```

Or install from source:

```bash
git clone https://github.com/CodonProject/codon-model.git
cd codon-model
pip install -e .
```

## License

Codon is licensed under the **Apache License 2.0**.
