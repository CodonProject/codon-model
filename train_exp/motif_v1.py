import os
import torch
import lpips
from tqdm import tqdm

from codon.utils.seed import seed_everything
seed_everything(42)

from codon.motif.motif_v1 import MotifV1
from codon.utils.data import ImageDataset, ImageDatasetItem
from codon.block.model.patch_disc import PatchDiscriminator
from codon.utils.transforms import _vision_transform

from codon.kit.train.vision import auto_vision_train

import torch.optim as optim

# config
LR = 1e-4
EPOCHS = 100
RESUME = True
PATH = {
    'data': './dataset',
    'ckpt': './checkpoint'
}
GRAD = 1
BATCH = 4

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# model
model = MotifV1().to(device)
discriminator = PatchDiscriminator.auto_build(
    in_channels=3,
    hidden_dim=256,
    image_size=12
).to(device)

# perceptual loss
perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device).eval()

# opt
optimizer_g = optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.9))
optimizer_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.9))

# dataset
dataset = ImageDataset(path=PATH['data'], transforms=_vision_transform())
dataloader = dataset.compose().loader(
    batch_size=BATCH,
    shuffle=True,
    drop_last=True
)

# create checkpoint directory
os.makedirs(PATH['ckpt'], exist_ok=True)

# resume training
start_epoch = 0
if RESUME:
    ckpts = [f for f in os.listdir(PATH['ckpt']) if f.endswith('.pth') and 'motif_v1_epoch_' in f]
    if ckpts:
        # Sort by epoch number
        ckpts.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_ckpt = os.path.join(PATH['ckpt'], ckpts[-1])
        print(f'Resuming from checkpoint: {latest_ckpt}')
        
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        start_epoch = checkpoint['epoch']

# training loop
for epoch in range(start_epoch, EPOCHS):
    model.train()
    discriminator.train()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}')
    
    for step, batch in enumerate(pbar):
        batch: ImageDatasetItem = batch
        images: torch.Tensor = batch.image.to(device)
        batch_size = images.size(0)
        
        batch_loss_g = 0.0
        batch_loss_d = 0.0
        batch_usage = 0.0
        
        for i in range(batch_size):
            single_image = images[i]
            
            result = auto_vision_train(
                model=model,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                image=single_image,
                patch_size=12,
                recon_loss_type='l1',
                recon_weight=1.0,
                perceptual_loss_fn=perceptual_loss_fn,
                perceptual_weight=1.0,
                adv_weight=0.1,
                quant_weight=1.0
            )
            
            batch_loss_g += result.loss_g
            batch_loss_d += result.loss_d
            batch_usage += result.codebook_usage_rate
            
        avg_loss_g = batch_loss_g / batch_size
        avg_loss_d = batch_loss_d / batch_size
        avg_usage = batch_usage / batch_size
        
        pbar.set_postfix({
            'loss_g': f'{avg_loss_g:.4f}',
            'loss_d': f'{avg_loss_d:.4f}',
            'usage': f'{avg_usage:.2f}'
        })
        
    ckpt_path = os.path.join(PATH['ckpt'], f'motif_v1_epoch_{epoch+1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
    }, ckpt_path)
    print(f'Checkpoint saved to {ckpt_path}')
