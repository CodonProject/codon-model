import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from dataclasses import dataclass
from typing import Union, Optional, Literal, Callable

from codon.model import PatchDiscriminator
from codon.model.motif import (
    AutoencoderVisionModel,
    AutoVisionEncoderOutput,
    AutoVisionDecoderOutput
)
from codon.utils.split import split_image, SplitedImage


@dataclass
class AutoVisionTrainResult:
    '''
    Dataclass to hold the outputs and metrics from a single auto_vision_train step.

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


def _patches_to_image(patches: torch.Tensor, grid_shape: tuple) -> torch.Tensor:
    '''
    Helper function to reconstruct a full image tensor from a sequence of patches.
    This is used to supply a padded full image to the generic AutoencoderVisionModel.encode().

    Args:
        patches (torch.Tensor): Patches tensor with shape [num_patches_h * num_patches_w, channels, patch_size, patch_size].
        grid_shape (tuple): Grid shape as (num_patches_h, num_patches_w).

    Returns:
        torch.Tensor: Reconstructed full image tensor with shape [1, channels, height, width].
    '''
    num_patches_h, num_patches_w = grid_shape
    channels, patch_size = patches.shape[1], patches.shape[2]

    patches = patches.view(1, num_patches_h, num_patches_w, channels, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    patches = patches.view(1, channels, num_patches_h * patch_size, num_patches_w * patch_size)

    return patches


def auto_vision_train(
    model: AutoencoderVisionModel,
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
) -> AutoVisionTrainResult:
    '''
    Executes a single end-to-end training step for an AutoencoderVisionModel.

    This function handles image splitting (with necessary padding), forward passes for both 
    the generator (AutoencoderVisionModel) and the discriminator (PatchDiscriminator), 
    loss calculations (including GAN, LPIPS, L1/MSE, and Quantization), and backpropagation.

    Args:
        model (AutoencoderVisionModel): The autoencoder vision model.
        discriminator (PatchDiscriminator): The PatchGAN discriminator.
        optimizer_g (torch.optim.Optimizer): Optimizer for the autoencoder model.
        optimizer_d (torch.optim.Optimizer): Optimizer for the discriminator.
        image (Union[torch.Tensor, str, Image.Image, np.ndarray]): The input image.
        patch_size (int): The patch size used by the model. Defaults to 12.
        recon_loss_type (Literal['l1', 'mse']): Type of reconstruction loss. Defaults to 'l1'.
        recon_weight (float): Weight for the reconstruction loss. Defaults to 1.0.
        perceptual_loss_fn (Callable, optional): Initialized LPIPS or other perceptual loss function. Defaults to None.
        perceptual_weight (float): Weight for the perceptual loss. Defaults to 1.0.
        adv_weight (float): Weight for the generator's adversarial GAN loss. Defaults to 0.1.
        quant_weight (float): Weight for the lookup-free quantization loss. Defaults to 1.0.

    Returns:
        AutoVisionTrainResult: Dataclass containing all the calculated losses and metrics.
    '''
    # Fallback mechanisms to get device and codebook_size if they aren't explicitly properties
    device = getattr(model, 'device', next(model.parameters()).device)
    codebook_size = getattr(model, 'codebook_size', 2**18)

    # 1. Process and split the input image with padding to handle arbitrary sizes
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

    # 2. Forward pass through generator (AutoencoderVisionModel)
    # Reconstruct padded full image to feed into generic encode method
    padded_full_image = _patches_to_image(real_patches, grid_shape).to(device)

    encoder_out: AutoVisionEncoderOutput = model.encode(padded_full_image)
    decoder_out: AutoVisionDecoderOutput = model.decode(encoder_out)
    
    fake_patches = decoder_out.reconstructed

    # 3. Discriminator Training
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

    # 4. Generator Training
    optimizer_g.zero_grad()

    # 4.1 Reconstruction Loss (L1 or MSE)
    loss_recon = recon_criterion(fake_patches, real_patches)

    # 4.2 Perceptual Loss (LPIPS)
    loss_perceptual_val = torch.tensor(0.0, device=device)
    if perceptual_loss_fn is not None:
        # Expected image range handling: LPIPS usually expects [-1, 1], models might output [0, 1]
        p_real = real_patches * 2.0 - 1.0
        p_fake = fake_patches * 2.0 - 1.0
        loss_perceptual_val = perceptual_loss_fn(p_real, p_fake).mean()

    # 4.3 Quantization Loss
    # Fallback to 0.0 if the encoder output does not provide a quantization loss (e.g., standard AE)
    loss_quant_val = torch.tensor(0.0, device=device)
    if encoder_out.loss is not None:
        loss_quant_val = encoder_out.loss

    # 4.4 Generator Adversarial Loss
    d_out_fake_g = discriminator(fake_patches)
    loss_adv = mse_criterion(d_out_fake_g, torch.ones_like(d_out_fake_g))

    # 4.5 Total Generator Loss
    loss_g = (
        recon_weight * loss_recon +
        perceptual_weight * loss_perceptual_val +
        quant_weight * loss_quant_val +
        adv_weight * loss_adv
    )

    loss_g.backward()
    optimizer_g.step()
    
    # Calculate codebook utilization if applicable
    usage_rate = 0.0
    if encoder_out.indices is not None:
        indices = encoder_out.indices
        unique_indices = torch.unique(indices)
        usage_rate = unique_indices.numel() / codebook_size

    perplexity_val = 0.0
    if encoder_out.perplexity is not None:
        perplexity_val = encoder_out.perplexity.item()

    return AutoVisionTrainResult(
        loss_g=loss_g.item(),
        loss_d=loss_d.item(),
        loss_recon=loss_recon.item(),
        loss_perceptual=loss_perceptual_val.item(),
        loss_quant=loss_quant_val.item(),
        loss_adv=loss_adv.item(),
        codebook_usage_rate=float(usage_rate),
        perplexity=float(perplexity_val),
        real_patches=real_patches,
        fake_patches=fake_patches
    )
