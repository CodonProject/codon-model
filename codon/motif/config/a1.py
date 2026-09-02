import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def build_config(total_steps: int) -> dict:
    if total_steps >= 20000:
        warmup_steps = max(2000, min(10000, int(total_steps * 0.05)))
    else:
        warmup_steps = max(1, int(total_steps * 0.08))

    return {
        'lr_peak': 6e-4,
        'lr_min': 6e-5,
        'weight_decay': 0.1,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        
        'total_steps': total_steps,
        'warmup_steps': warmup_steps,
        
        'grad_clip_norm': 1.0
    }

def build_optim_and_scheduler(model: nn.Module, config: dict) -> tuple[AdamW, SequentialLR]:
    optimizer = AdamW(
        model.parameters(),
        lr=config['lr_peak'],
        betas=config['betas'],
        eps=config['eps'],
        weight_decay=config['weight_decay']
    )
    
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=config['warmup_steps']
    )
    
    decay_steps = config['total_steps'] - config['warmup_steps']
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=max(1, decay_steps),
        eta_min=config['lr_min']
    )
    
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[config['warmup_steps']]
    )
    
    return optimizer, scheduler