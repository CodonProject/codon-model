# Evaluation Library Documentation

## Overview

Analysis and visualization tools for model evaluation.

## Base Classes

### BaseAnalyzer

Base class for all analyzers.

```python
class BaseAnalyzer:
    def __init__(
        self,
        class_info: Union[int, list[str]],
        lang: str = None
    )
```

### AnalysisResult

Result container for analysis.

```python
@dataclass
class AnalysisResult:
    fig: plt.Figure
    ax: plt.Axes
    data: Any
    plot_func: Callable
```

---

## Analyzers

### ConfusionMap

Confusion matrix visualization.

```python
from codon.utils.eval import ConfusionMap

confusion = ConfusionMap(
    class_info=['cat', 'dog', 'bird'],
    val_loader=val_loader
)

result = confusion.analyse(model, name='Epoch 10')
print(f"Confusion matrix:\n{result.data}")
result.fig.savefig('confusion.png')
```

---

### TSNEMap

t-SNE feature visualization.

```python
from codon.utils.eval import TSNEMap

tsne = TSNEMap(
    class_info=10,
    val_loader=val_loader
)

result = tsne.analyse(
    feature_extractor=model.backbone,
    max_samples=2000,
    perplexity=30.0
)
result.fig.savefig('tsne.png')
```

---

### RSAMap

Representational Similarity Analysis.

```python
from codon.utils.eval import RSAMap

rsa = RSAMap(
    class_info=10,
    val_loader=val_loader
)

result = rsa.analyse(model)
```

---

### GradCAMMap

Gradient-weighted Class Activation Mapping.

```python
from codon.utils.eval import GradCAMMap

gradcam = GradCAMMap(
    model=model,
    target_layer=model.layer4[-1]
)

result = gradcam.analyse(input_image, target_class=0)
```

---

### ActivationDistribution

Activation distribution analysis.

```python
from codon.utils.eval import ActivationDistribution

act_dist = ActivationDistribution(
    model=model,
    layer_name='layer3'
)

result = act_dist.analyse(val_loader)
```

---

### LayerRSAMap

Layer-wise RSA comparison.

```python
from codon.utils.eval import LayerRSAMap

layer_rsa = LayerRSAMap(
    model=model,
    layers=['layer1', 'layer2', 'layer3', 'layer4']
)

result = layer_rsa.analyse(val_loader)
```

---

### DecisionBoundaryMap

Decision boundary visualization.

```python
from codon.utils.eval import DecisionBoundaryMap

boundary = DecisionBoundaryMap(
    model=model,
    feature_extractor=backbone
)

result = boundary.analyse(val_loader)
```

---

### CKAMap

Centered Kernel Alignment.

```python
from codon.utils.eval import CKAMap

cka = CKAMap(
    model1=model1,
    model2=model2
)

result = cka.analyse(val_loader)
```

---

### NeuronSelectivity

Neuron selectivity analysis.

```python
from codon.utils.eval import NeuronSelectivity

selectivity = NeuronSelectivity(
    model=model,
    layer_name='layer3'
)

result = selectivity.analyse(val_loader)
```

---

## Usage Patterns

### Full Evaluation Pipeline

```python
from codon.utils.eval import (
    ConfusionMap, TSNEMap, RSAMap,
    GradCAMMap, ActivationDistribution
)

# Confusion matrix
confusion = ConfusionMap(class_names, val_loader)
conf_result = confusion.analyse(model)

# Feature visualization
tsne = TSNEMap(class_names, val_loader)
tsne_result = tsne.analyse(model.backbone)

# Activation analysis
act = ActivationDistribution(model, 'layer3')
act_result = act.analyse(val_loader)

# Save all results
conf_result.fig.savefig('confusion.png')
tsne_result.fig.savefig('tsne.png')
act_result.fig.savefig('activation.png')
```

---

## Notes

1. **Language Support**: Supports 'en' and 'zh' for labels.
2. **Batch Processing**: All analyzers work with DataLoader.
3. **Plot Customization**: Access `fig` and `ax` for further customization.