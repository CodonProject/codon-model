# Codon

[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/Pytorch-2.0%2B-orange?style=flat&logo=pytorch)](https://pytorch.org)
[![PyPI version](https://img.shields.io/pypi/v/codon-model?style=flat&logoColor=Pypi&label=PyPI&color=green)](https://pypi.org/project/codon-model)

[![HuggingFace](https://img.shields.io/badge/HuggingFace-CodonProject-white?logo=huggingface)](https://huggingface.co/CodonProject)
[![Modelscope](https://img.shields.io/badge/Modelscope-CodonProject-white?logo=modelscope)](https://www.modelscope.cn/organization/CodonProject)


**Codon** is a lightweight PyTorch toolbox designed for research-driven prototyping and rapid architecture exploration. It aims to bridge the gap between abstract algorithmic ideas and trainable deep learning modules, minimizing engineering boilerplate while preserving flexibility.

---

## 🌟 Key Workflows

### 1. Rapid Architecture Prototyping
Assemble custom models (Transformers, MoEs, Multimodals) using clean, composable building blocks:
*   **Attention Variants**: Grouped Query Attention (GQA), QK Normalization, K=V Attention (KEV), and Gated Attention.
*   **Flexible Normalization**: RMSNorm, Zero-Centered RMSNorm (ZCRMSNorm), and Conditional Adaptive LayerNorm (AdaLayerNorm).
*   **Modular Layers**: SwiGLU MLP, Low-Rank Adaptation (LoRA/DoRA), and Space-to-Depth PixelShuffle.

### 2. Experimental Research Operators
Explore non-standard, biology-inspired and physics-aligned mathematical primitives:
*   **Bio-inspired Learning (`ops/bio.py`)**: Biological learning rules including Hebbian, Oja, BCM, and Spike-Timing-Dependent Plasticity (STDP).
*   **Manifold Operations (`ops/manifold/`)**: Riemannian manifold projection using von Mises-Fisher (vMF) concentration and gravity attraction (accelerated with Triton).
*   **Spectral Mixing (`ops/fourier.py`)**: Causal frequency-domain token mixing with $O(L \log L)$ complexity.

### 3. Built-in Model Families (Motifs)
Jumpstart your modeling with pre-configured models that support one-line remote weight loading with automatic Hugging Face/ModelScope platform selection:
*   **MotifA1**: A Causal Language Model featuring GQA, SwiGLU, and KV caching.
*   **MotifV1**: A Vision Autoencoder powered by Lookup-Free Quantization (LFQ) and 2D Rotary Embeddings.

---

## 🛠️ Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/CodonProject/codon-model.git
cd codon-model
pip install -e .
```

---

## 🚀 Quick Start

### 1. Build a Custom Model with `BasicModel`
Inherit from `BasicModel` to instantly gain features like gradient checkpointing, precise parameter counting, and strict `.safetensors` serialization rules.

```python
from codon.base  import *
from codon.block import TransformerDenseDecoder

class CustomCognitiveModel(BasicModel):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        # Composable Transformer Block with SwiGLU
        self.layer = TransformerDenseDecoder(
            model_dim=d_model,
            num_heads=n_heads,
            use_swiglu=True
        )
        self.head = nn.Linear(d_model, 10)

    def forward(self, x):
        # Activation checkpointing is supported natively
        x = self.checkpoint(self.layer, x).output
        return self.head(x)

# Instantiate and inspect
model = CustomCognitiveModel()
print(f"Trainable parameters: {model.count_params(trainable_only=True, human_readable=True)}")

# Freeze base layers easily for probe-tuning
model.layer.freeze()
print(f"After freezing base: {model.count_params(trainable_only=True, human_readable=True)}")
```

### 2. Auto-load Pretrained Models
Download and cache weights from either ModelScope or Hugging Face automatically based on your network latency:

```python
from codon.motif import MotifA1, MotifA1Tokenizer

# The best mirror platform is selected automatically
model = MotifA1().from_remote()
tokenizer = MotifA1Tokenizer().from_remote()

model.eval()
input_ids = tokenizer.encode("Consciousness is defined as", return_tensors='pt')

# Autoregressive generation
output_ids = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output_ids[0]))
```

---

## 📂 Repository Structure

```text
codon/
├── block/          # Neural network building blocks (Attention, MoE, LoRA, Norm, etc.)
├── motif/          # Pre-built model families (MotifA1 Language Model, MotifV1 Autoencoder)
├── ops/            # Experimental operators (Bio-inspired, Manifold, Fourier)
├── model/          # Classic model baselines (ResNet, TCN, PatchGAN)
├── kit/            # Training loop frameworks (Language/Vision)
└── utils/          # Core utilities (PackedTokenizers, Session masking, Eval tools)
```

---

## 📜 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.
