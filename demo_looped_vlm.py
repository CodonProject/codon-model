import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from exp_loop import LoopedConfig, LoopedVLM

torch.manual_seed(0)

cfg = LoopedConfig(model_dim=128, num_heads=4, recurrence=1,
                   prelude_attn_types=['mha'], body_attn_types=['mha', 'gdn'],
                   coda_attn_types=['mha'])
vlm = LoopedVLM(cfg, vocab_size=256, backend='small').eval()
print('params:', vlm.count_params(human_readable=True))

B, P = 1, 49                  # 224x224 / 32 -> 7x7 = 49 patch
L = 3 + P + 5                 # [文本×3][图像 49][文本×5]
input_ids = torch.randint(4, 256, (B, L))
input_ids[0, 3:3 + P] = 0     # 占位符 token id
patch_idx = torch.arange(3, 3 + P).unsqueeze(0)
images = [torch.randn(3, 224, 224)]

with torch.no_grad():
    out = vlm(input_ids=input_ids, images=images, image_patch_indices=patch_idx)
print('seq_len:', L, '| logits:', tuple(out.logits.shape))

# 1) 视觉特征与位置
with torch.no_grad():
    feats = vlm.encode_image(images[0].unsqueeze(0)).squeeze(0)      # [49, 128]
    pos = vlm._build_vlm_positions(B, L, patch_idx, [(7, 7)], [P], 0, input_ids.device)
print('vision feats   :', tuple(feats.shape))
print('patch positions:', pos[0, 3].tolist(), '...', pos[0, 3 + 48].tolist(), '(row,col) 0..6')
print('text positions :', pos[0, 0].tolist(), pos[0, L - 1].tolist(), '(t,t)')

# 2) 换图 -> logits 变化；同图 -> 可复现
with torch.no_grad():
    out2 = vlm(input_ids=input_ids, images=[torch.randn(3, 224, 224)], image_patch_indices=patch_idx)
    out3 = vlm(input_ids=input_ids, images=images, image_patch_indices=patch_idx)
print('different image changes logits:', not torch.allclose(out.logits, out2.logits))
print('same image reproducible        :', torch.allclose(out.logits, out3.logits, atol=1e-6))

# 3) 占位符数量不匹配 -> 报错
try:
    with torch.no_grad():
        vlm(input_ids=input_ids, images=images,
            image_patch_indices=torch.arange(3, 3 + P - 1).unsqueeze(0))
    print('ERROR: mismatch not rejected')
except ValueError as e:
    print('mismatch rejected:', str(e)[:50], '...')

# 4) 占位符越界 -> 报错
try:
    with torch.no_grad():
        vlm(input_ids=input_ids[:, :20], images=images,
            image_patch_indices=torch.arange(3, 3 + P).unsqueeze(0))
    print('ERROR: out-of-range not rejected')
except ValueError as e:
    print('out-of-range rejected:', str(e)[:50], '...')

# 5) 纯文本仍可用
with torch.no_grad():
    txt = vlm(input_ids=torch.randint(4, 256, (1, 8)))
print('text-only logits:', tuple(txt.logits.shape))

# 6) prefill（含图）+ decode 与一次性前向一致
ids = torch.randint(4, 256, (1, 3 + P + 4))
ids[0, 2:2 + P] = 0
p_idx = torch.arange(2, 2 + P).unsqueeze(0)
pre = 3 + P
with torch.no_grad():
    full = vlm(input_ids=ids, images=images, image_patch_indices=p_idx).logits
    cache = vlm.create_cache()
    vlm(input_ids=ids[:, :pre], images=images, image_patch_indices=p_idx, past_key_values=cache)
    step = vlm(input_ids=ids[:, pre:pre + 1], start_pos=pre, past_key_values=cache).logits
print('prefill+decode vs one-shot diff:', (step[:, -1] - full[:, pre]).abs().max().item())
print('OK')
