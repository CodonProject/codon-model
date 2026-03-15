import os
import sys

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Dynamically add the project root to the beginning of sys.path to prioritize local imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.optim as optim
from PIL import Image

from codon.model.motif.motif_v1 import MotifV1
from codon.model.patch_disc import PatchDiscriminator
from codon.kit.auto_vision_train import auto_train_motif_vision


def main() -> None:
    '''
    Main function to test the auto_train_motif_vision training step.
    It initializes the models, optimizers, and runs a single training iteration.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('Initializing MotifV1...')
    model = MotifV1(
        in_features=3,
        out_features=3,
        patch_size=12,
        latent_dim=256,
        num_heads=4,
        num_kv_heads=4,
        codebook_dim=18     # 2^18 = 262144
    ).to(device)

    print('Initializing PatchDiscriminator...')
    discriminator = PatchDiscriminator.auto_build(
        in_channels=3,
        hidden_dim=64,
        image_size=12
    ).to(device)

    # Setup optimizers
    lr = 1e-4
    optimizer_g = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    # Prepare input image
    image_path = os.path.join(project_root, 'test.jpg')
    
    if os.path.exists(image_path):
        print(f'Loading image from: {image_path}')
        try:
            image = Image.open(image_path).convert('RGB')
            # Resize image to prevent massive sequence length and CUDA timeout
            image = image.resize((256, 256))
        except Exception as e:
            print(f'Failed to load image: {e}. Falling back to random tensor.')
            # [C, H, W] expected for split_image if tensor is used directly
            image = torch.rand(3, 256, 256) 
    else:
        print(f'Image not found at {image_path}. Using a random dummy tensor.')
        # Provide a dummy image tensor of size [3, H, W]
        image = torch.rand(3, 256, 256)

    print('Starting auto_train_motif_vision step...')
    try:
        output = auto_train_motif_vision(
            model=model,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            image=image,
            patch_size=12,
            recon_loss_type='l1',
            recon_weight=1.0,
            perceptual_loss_fn=None,  # Not using LPIPS for basic test
            perceptual_weight=1.0,
            adv_weight=0.1,
            quant_weight=1.0,
            device=device
        )

        print('\n--- Training Step Successful ---')
        print(f'Loss G:          {output.loss_g:.4f}')
        print(f'Loss D:          {output.loss_d:.4f}')
        print(f'Recon Loss:      {output.loss_recon:.4f}')
        print(f'Quant Loss:      {output.loss_quant:.4f}')
        print(f'Adversarial Loss:{output.loss_adv:.4f}')
        print(f'Codebook Usage:  {output.codebook_usage_rate * 100:.2f}%')
        print(f'Perplexity:      {output.perplexity:.4f}')
        
        if output.real_patches is not None and output.fake_patches is not None:
            print(f'Real Patches Shape: {output.real_patches.shape}')
            print(f'Fake Patches Shape: {output.fake_patches.shape}')

    except Exception as e:
        print(f'\nTraining step failed with error:\n{e}')
        raise


if __name__ == '__main__':
    main()
