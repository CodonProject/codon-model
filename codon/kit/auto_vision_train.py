import torch
import torch.nn as nn
import numpy as np

from dataclasses import dataclass
from typing import Union, Optional, Literal, Callable
from PIL import Image

from codon.model.motif.motif_v1 import MotifV1, MotifV1Output
from codon.model.patch_disc import PatchDiscriminator
from codon.utils.split import split_image, SplitedImage


@dataclass
class AutoTrainMotifVisionOutput:
    '''
    Dataclass to hold the outputs and metrics from a single auto_train step.

    Attributes:
        loss_g (float): Total generator loss.
        loss_d (float): Total discriminator loss.
        loss_recon (float): Reconstruction loss (L1 or MSE).
        loss_perceptual (float): Perceptual loss from LPIPS. Returns 0.0 if not used.
        loss_quant (float): Quantization loss from the codebook.
        loss_adv (float): Generator's adversarial loss from PatchGAN.
        codebook_usage_rate (float): Percentage of the codebook utilized in this step (0.0 to 1.0).
        perplexity (float): Perplexity of the quantization process.
        real_patches (torch.Tensor, optional): The original splited image patches for visualization.
        fake_patches (torch.Tensor, optional): The reconstructed image patches for visualization.
    '''
    loss_g: float
    loss_d: float
    loss_recon: float
    loss_perceptual: float
    loss_quant: float
    loss_adv: float
    codebook_usage_rate: float
    perplexity: float
    real_patches: Optional[torch.Tensor] = None
    fake_patches: Optional[torch.Tensor] = None


def auto_train_motif_vision(
    model: MotifV1,
    discriminator: PatchDiscriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    image: Union[torch.Tensor, str, Image.Image, np.ndarray],
    patch_size: int = 12,
    recon_loss_type: Literal['l1', 'mse'] = 'l1',
    recon_weight: float = 1.0,
    perceptual_loss_fn: Optional[Callable] = None,
    perceptual_weight: float = 1.0,
    adv_weight: float = 0.1,
    quant_weight: float = 1.0,
    codebook_size: int = 2**18,
    device: Union[str, torch.device] = 'cpu'
) -> AutoTrainMotifVisionOutput:
    '''
    Executes a single end-to-end training step for the MotifV1 autoencoder.

    This function handles image splitting, forward passes for both the generator (MotifV1) 
    and the discriminator (PatchDiscriminator), loss calculations (including GAN, LPIPS, 
    L1/MSE, and Quantization), and backpropagation.

    Args:
        model (MotifV1): The MotifV1 autoencoder model.
        discriminator (PatchDiscriminator): The PatchGAN discriminator.
        optimizer_g (torch.optim.Optimizer): Optimizer for the MotifV1 model.
        optimizer_d (torch.optim.Optimizer): Optimizer for the discriminator.
        image (Union[torch.Tensor, str, Image.Image, np.ndarray]): The input image.
        patch_size (int): The patch size used by the MotifV1 model. Defaults to 12.
        recon_loss_type (Literal['l1', 'mse']): Type of reconstruction loss. Defaults to 'l1'.
        recon_weight (float): Weight for the reconstruction loss. Defaults to 1.0.
        perceptual_loss_fn (Callable, optional): Initialized LPIPS or other perceptual loss function. Defaults to None.
        perceptual_weight (float): Weight for the perceptual loss. Defaults to 1.0.
        adv_weight (float): Weight for the generator's adversarial GAN loss. Defaults to 0.1.
        quant_weight (float): Weight for the lookup-free quantization loss. Defaults to 1.0.
        codebook_size (int): The total capacity of the codebook. Defaults to 2^18 = 262144.
        device (Union[str, torch.device]): Device to perform computations on. Defaults to 'cpu'.

    Returns:
        AutoTrainMotifVisionOutput: Dataclass containing all the calculated losses and metrics.
    '''
    # 1. Process and split the input image
    splited: SplitedImage = split_image(
        image=image,
        patch_size=patch_size,
        padding=True
    )
    
    real_patches = splited.patches.to(device)
    grid_shape = splited.grid_shape

    model.train()
    discriminator.train()

    # Define simple loss functions
    mse_criterion = nn.MSELoss()
    if recon_loss_type == 'l1':
        recon_criterion = nn.L1Loss()
    else:
        recon_criterion = mse_criterion

    # Forward pass through MotifV1 once, reusing outputs for both discriminator and generator
    motif_out: MotifV1Output = model(real_patches, grid_shape)
    fake_patches = motif_out.reconstructed_image

    optimizer_d.zero_grad()

    # Forward discriminator on real patches
    d_out_real = discriminator(real_patches)
    loss_d_real = mse_criterion(d_out_real, torch.ones_like(d_out_real))

    # Forward discriminator on fake patches (detached to avoid backprop to generator)
    d_out_fake = discriminator(fake_patches.detach())
    loss_d_fake = mse_criterion(d_out_fake, torch.zeros_like(d_out_fake))

    # Total discriminator loss and backprop
    loss_d = 0.5 * (loss_d_real + loss_d_fake)
    loss_d.backward()
    optimizer_d.step()

    optimizer_g.zero_grad()

    # 2.1 Reconstruction Loss (L1 or MSE)
    loss_recon = recon_criterion(fake_patches, real_patches)

    # 2.2 Perceptual Loss (LPIPS)
    loss_perceptual_val = torch.tensor(0.0, device=device)
    if perceptual_loss_fn is not None:
        # LPIPS expects input in range [-1, 1], Motif uses [0, 1]
        p_real = real_patches * 2.0 - 1.0
        p_fake = fake_patches * 2.0 - 1.0
        loss_perceptual_val = perceptual_loss_fn(p_real, p_fake).mean()

    # 2.3 Quantization Loss
    loss_quant = motif_out.quantization_loss

    # 2.4 Generator Adversarial Loss
    d_out_fake_g = discriminator(fake_patches)
    loss_adv = mse_criterion(d_out_fake_g, torch.ones_like(d_out_fake_g))

    # 2.5 Total Generator Loss
    loss_g = (
        recon_weight * loss_recon +
        perceptual_weight * loss_perceptual_val +
        quant_weight * loss_quant +
        adv_weight * loss_adv
    )

    loss_g.backward()
    optimizer_g.step()
    
    # Calculate codebook utilization
    indices = motif_out.indices
    unique_indices = torch.unique(indices)
    usage_rate = unique_indices.numel() / codebook_size

    return AutoTrainMotifVisionOutput(
        loss_g=loss_g.item(),
        loss_d=loss_d.item(),
        loss_recon=loss_recon.item(),
        loss_perceptual=loss_perceptual_val.item(),
        loss_quant=loss_quant.item(),
        loss_adv=loss_adv.item(),
        codebook_usage_rate=float(usage_rate),
        perplexity=motif_out.perplexity.item(),
        real_patches=real_patches,
        fake_patches=fake_patches
    )
