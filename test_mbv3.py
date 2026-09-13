from codon import *
from codon.impl import MobileNetV3

backend = 'large'
image_dim = 960 if backend == 'large' else 576

model = MobileNetV3(return_features=True, backend=backend).from_remote()
model.eval()

batch_size = 2
dummy_input = torch.randn(batch_size, 3, 224, 224)

with torch.no_grad():
    features = model(dummy_input).permute(0, 2, 3, 1)

print(features.shape)

model_dim = 768

proj = nn.Linear(image_dim, model_dim)

features = proj(features)

print(features.shape) # torch.Size([2, 7, 7, 768])

_, h, w, _ = features.shape

from codon.block.embedding import InterleavedFourierRotaryEmbedding

rope = InterleavedFourierRotaryEmbedding(
    model_dim=model_dim,
    max_len=8192,
    num_axes=2
)

row = torch.arange(h).view(-1, 1).expand(h, w).flatten()
col = torch.arange(w).view(1, -1).expand(h, w).flatten()
positions = torch.stack([row, col], dim=-1).unsqueeze(0).expand(batch_size, -1, -1)

tokens = features.flatten(1, 2)

tokens_with_pos = rope(tokens, positions=positions)

print(tokens_with_pos.shape)