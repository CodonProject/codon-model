from vl import MotifA1_VL

import torch

model = MotifA1_VL('dead_codes.json')

model.language.freeze()
model.vision.freeze()

optimizer = torch.optim.AdamW(model.trainable_params, lr=5e-4, weight_decay=0.05)

