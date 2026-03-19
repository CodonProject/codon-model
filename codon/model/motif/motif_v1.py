from codon.base import *

from codon.model import ResNet
from codon.block import (
    LookupFreeQuantization, LookupFreeQuantizationOutput,
    InterleavedRotaryEmbedding,
    MultiHeadAttention, AttentionOutput,
    ConvBlock, PixelShuffleUpSample
)

from .base import AutoencoderVisionModel, AutoVisionEncoderOutput, AutoVisionDecoderOutput

from typing import Tuple

import math


class MotifV1Encoder(BasicModel):
    def __init__(
        self,
        in_features: int = 3,
        patch_size: int = 12,
        latent_dim: int = 256,
        num_heads: int = 4,
        num_kv_heads: int = 4,
        codebook_dim: int = 18,
        entropy_weight: float = 0.1,
        commitment_weight: float = 0.25,
        diversity_gamma: float = 1.0,
        rope_emb: InterleavedRotaryEmbedding = None,
        use_attention: bool = True,
        depth_level: int = 1
    ):
        super().__init__()
        self.in_features = in_features
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.codebook_dim = codebook_dim
        self.entropy_weight = entropy_weight
        self.commitment_weight = commitment_weight
        self.diversity_gamma = diversity_gamma
        self.use_attention = use_attention
        self.depth_level = depth_level

        self.resnet = ResNet.auto_build(
            input_shape=(in_features, patch_size, patch_size),
            output_shape=(latent_dim,),
            depth_level=depth_level
        )

        self.attn = MultiHeadAttention(
            hidden_size=latent_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            use_qk_norm=True,
            use_gate=True,
            is_causal=False
        ) if self.use_attention else None

        self.codebook = LookupFreeQuantization(
            latent_dim=latent_dim,
            codebook_dim=codebook_dim,
            entropy_weight=entropy_weight,
            commitment_weight=commitment_weight,
            diversity_gamma=diversity_gamma
        )

        self.rope_emb = rope_emb if isinstance(rope_emb, InterleavedRotaryEmbedding) and rope_emb.num_axes >= 2 else InterleavedRotaryEmbedding(
            model_dim=latent_dim // num_heads,
            num_axes=2
        )

    def forward(
        self,
        splited_image: torch.Tensor,
        grid_shape: tuple,
        rope_emb: InterleavedRotaryEmbedding = None
    ) -> AutoVisionEncoderOutput:
        '''
        Forward pass of the MotifV1Encoder.

        Args:
            splited_image (torch.Tensor): Input tensor with shape [num_patches, channels, patch_size, patch_size].
            grid_shape (tuple): Grid shape as (num_patches_h, num_patches_w).
            rope_emb (InterleavedRotaryEmbedding, optional): 2D rotary positional embedding. Defaults to None.

        Returns:
            AutoVisionEncoderOutput: The output containing quantized latent, loss, indices, etc.
        '''
        num_patches_h, num_patches_w = grid_shape

        hidden_states = self.resnet(splited_image)

        hidden_states = hidden_states.unsqueeze(0)

        positions_h = torch.arange(num_patches_h, device=splited_image.device)
        positions_w = torch.arange(num_patches_w, device=splited_image.device)
        grid_h, grid_w = torch.meshgrid(positions_h, positions_w, indexing='ij')
        positions = torch.stack([grid_h.flatten(), grid_w.flatten()], dim=-1)
        positions = positions.unsqueeze(0).float()

        current_rope = rope_emb if rope_emb is not None else self.rope_emb

        if self.use_attention:
            attn_out: AttentionOutput = self.attn(
                hidden_states=hidden_states,
                position_emb=current_rope,
                embedding_pos=positions
            )
            z = attn_out.output.squeeze(0)
            hidden_states_out = attn_out.output.squeeze(0)
        else:
            hidden_states = current_rope(hidden_states, positions=positions)
            z = hidden_states.squeeze(0)
            hidden_states_out = hidden_states.squeeze(0)

        z = z.view(1, num_patches_h, num_patches_w, self.latent_dim)
        z = z.permute(0, 3, 1, 2)

        codebook_out: LookupFreeQuantizationOutput = self.codebook(z)

        z_q = codebook_out.z_q.permute(0, 2, 3, 1).reshape(-1, self.latent_dim)

        return AutoVisionEncoderOutput(
            z_q=z_q,
            loss=codebook_out.loss,
            indices=codebook_out.indices.reshape(-1),
            entropy=codebook_out.entropy,
            perplexity=codebook_out.perplexity,
            hidden_states=hidden_states_out,
            grid_shape=grid_shape
        )


class MotifV1Decoder(BasicModel):
    def __init__(
        self,
        latent_dim: int = 256,
        out_features: int = 3,
        patch_size: int = 12,
        num_heads: int = 4,
        num_kv_heads: int = 4,
        base_channels: int = 64,
        initial_size: int = None,
        rope_emb: InterleavedRotaryEmbedding = None,
        norm: str = 'batch',
        activation: str = 'relu',
        use_attention: bool = True,
        depth_level: int = 1
    ) -> None:
        '''
        Initializes the MotifV1Decoder module.

        Args:
            latent_dim (int): The dimensionality of the latent representation. Defaults to 256.
            out_features (int): Number of output features (e.g., RGB channels). Defaults to 3.
            patch_size (int): The spatial size of the output patch. Defaults to 12.
            num_heads (int): Number of attention heads. Defaults to 4.
            num_kv_heads (int): Number of Key/Value attention heads. Defaults to 4.
            base_channels (int): Base channel width for the upsampling module. Defaults to 64.
            initial_size (int, optional): The spatial size of the feature map before upsampling.
                                          If None, automatically calculated to match the Encoder's downsampling. Defaults to None.
            rope_emb (InterleavedRotaryEmbedding, optional): 2D rotary positional embedding. Defaults to None.
            norm (str): Normalization type for convolution blocks. Defaults to 'batch'.
            activation (str): Activation function type. Defaults to 'relu'.
            use_attention (bool): Whether to use attention layer. Defaults to True.
            depth_level (int): Level of network depth multiplier. Defaults to 1.
        '''
        super().__init__()

        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.out_features = out_features
        self.use_attention = use_attention
        self.depth_level = depth_level

        self.attn = MultiHeadAttention(
            hidden_size=latent_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            use_qk_norm=True,
            use_gate=True,
            is_causal=False
        ) if self.use_attention else None

        self.rope_emb = rope_emb if isinstance(rope_emb, InterleavedRotaryEmbedding) and rope_emb.num_axes >= 2 else InterleavedRotaryEmbedding(
            model_dim=latent_dim // num_heads,
            num_axes=2
        )

        if initial_size is None:
            num_stages = max(1, int(math.log2(patch_size)))
            downsample_factor = 2 ** (num_stages - 1)
            self.initial_size = max(1, patch_size // downsample_factor)
        else:
            self.initial_size = initial_size

        self.initial_channels = base_channels * 4

        self.linear = nn.Linear(latent_dim, self.initial_channels * self.initial_size * self.initial_size)

        self.up_blocks = PixelShuffleUpSample.auto_build(
            input_shape=(self.initial_channels, self.initial_size, self.initial_size),
            output_shape=(base_channels, patch_size, patch_size),
            norm=norm,
            activation=activation,
            depth_level=depth_level
        )

        self.final_conv = ConvBlock(
            in_channels=base_channels,
            out_channels=out_features,
            kernel_size=3,
            padding=1,
            norm=None,
            activation=None
        )

    def forward(
        self,
        z_q: torch.Tensor,
        grid_shape: tuple,
        rope_emb: InterleavedRotaryEmbedding = None
    ) -> AutoVisionDecoderOutput:
        '''
        Forward pass of the MotifV1Decoder.

        Args:
            z_q (torch.Tensor): Input quantized latent tensor with shape [num_patches, latent_dim].
            grid_shape (tuple): Grid shape as (num_patches_h, num_patches_w).
            rope_emb (InterleavedRotaryEmbedding, optional): 2D rotary positional embedding. Defaults to None.

        Returns:
            AutoVisionDecoderOutput: The output containing reconstructed image patches and hidden states.
        '''
        num_patches_h, num_patches_w = grid_shape

        hidden_states = z_q.unsqueeze(0)

        positions_h = torch.arange(num_patches_h, device=z_q.device)
        positions_w = torch.arange(num_patches_w, device=z_q.device)
        grid_h, grid_w = torch.meshgrid(positions_h, positions_w, indexing='ij')
        positions = torch.stack([grid_h.flatten(), grid_w.flatten()], dim=-1)
        positions = positions.unsqueeze(0).float()

        current_rope = rope_emb if rope_emb is not None else self.rope_emb

        if self.use_attention:
            attn_out: AttentionOutput = self.attn(
                hidden_states=hidden_states,
                position_emb=current_rope,
                embedding_pos=positions
            )
            z_hidden = attn_out.output.squeeze(0)
        else:
            hidden_states = current_rope(hidden_states, positions=positions)
            z_hidden = hidden_states.squeeze(0)

        z = self.linear(z_hidden)
        z = z.view(-1, self.initial_channels, self.initial_size, self.initial_size)

        z = self.up_blocks(z)
        output = self.final_conv(z)

        return AutoVisionDecoderOutput(
            reconstructed=output,
            grid_shape=grid_shape,
            hidden_states=z_hidden
        )


class MotifV1(AutoencoderVisionModel):
    '''
    MotifV1 autoencoder model combining MotifV1Encoder and MotifV1Decoder.

    Attributes:
        encoder (MotifV1Encoder): The encoder part of the model.
        decoder (MotifV1Decoder): The decoder part of the model.
        rope_emb (InterleavedRotaryEmbedding): Shared rotary positional embedding for both encoder and decoder.
    '''

    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 3,
        patch_size: int = 12,
        latent_dim: int = 256,
        num_heads: int = 4,
        num_kv_heads: int = 4,
        codebook_dim: int = 18,
        entropy_weight: float = 0.1,
        commitment_weight: float = 0.25,
        diversity_gamma: float = 1.0,
        base_channels: int = 128,
        initial_size: int = None,
        rope_emb: InterleavedRotaryEmbedding = None,
        norm: str = 'batch',
        activation: str = 'relu',
        encoder_use_attention: bool = True,
        decoder_use_attention: bool = True,
        encoder_depth_level: int = 6,
        decoder_depth_level: int = 6
    ) -> None:
        '''
        Initializes the MotifV1 model.

        Args:
            in_features (int): Number of input features (channels). Defaults to 3.
            out_features (int): Number of output features (channels). Defaults to 3.
            patch_size (int): The spatial size of the patch. Defaults to 12.
            latent_dim (int): The dimensionality of the latent representation. Defaults to 256.
            num_heads (int): Number of attention heads. Defaults to 4.
            num_kv_heads (int): Number of Key/Value attention heads. Defaults to 4.
            codebook_dim (int): Dimension of the codebook. Defaults to 18.
            entropy_weight (float): Weight for the entropy loss. Defaults to 0.1.
            commitment_weight (float): Weight for the commitment loss. Defaults to 0.25.
            diversity_gamma (float): Gamma value for diversity loss. Defaults to 1.0.
            base_channels (int): Base channel width for the upsampling module in decoder. Defaults to 128.
            initial_size (int, optional): The spatial size of the feature map before upsampling. Defaults to None.
            rope_emb (InterleavedRotaryEmbedding, optional): 2D rotary positional embedding. Defaults to None.
            norm (str): Normalization type for convolution blocks. Defaults to 'batch'.
            activation (str): Activation function type. Defaults to 'relu'.
            encoder_use_attention (bool): Whether to use attention in encoder. Defaults to True.
            decoder_use_attention (bool): Whether to use attention in decoder. Defaults to True.
            encoder_depth_level (int): Level of network depth multiplier for encoder. Defaults to 6.
            decoder_depth_level (int): Level of network depth multiplier for decoder. Defaults to 6.
        '''
        super().__init__()
        self.codebook_size = 2 ** codebook_dim

        self.rope_emb = rope_emb if isinstance(rope_emb, InterleavedRotaryEmbedding) and rope_emb.num_axes >= 2 else InterleavedRotaryEmbedding(
            model_dim=latent_dim // num_heads,
            num_axes=2
        )

        self.encoder = MotifV1Encoder(
            in_features=in_features,
            patch_size=patch_size,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            codebook_dim=codebook_dim,
            entropy_weight=entropy_weight,
            commitment_weight=commitment_weight,
            diversity_gamma=diversity_gamma,
            rope_emb=self.rope_emb,
            use_attention=encoder_use_attention,
            depth_level=encoder_depth_level
        )

        self.decoder = MotifV1Decoder(
            latent_dim=latent_dim,
            out_features=out_features,
            patch_size=patch_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            base_channels=base_channels,
            initial_size=initial_size,
            rope_emb=self.rope_emb,
            norm=norm,
            activation=activation,
            use_attention=decoder_use_attention,
            depth_level=decoder_depth_level
        )

    def forward(self, splited_image: torch.Tensor, grid_shape: tuple) -> AutoVisionEncoderOutput:
        '''
        Forward pass of the MotifV1 model.

        Args:
            splited_image (torch.Tensor): Input tensor with shape [num_patches, channels, patch_size, patch_size].
            grid_shape (tuple): Grid shape as (num_patches_h, num_patches_w).

        Returns:
            AutoVisionEncoderOutput: Output containing reconstructed image and latent details.
        '''
        encoder_out = self.encoder(splited_image, grid_shape)
        decoder_out = self.decoder(encoder_out.z_q, grid_shape)

        return AutoVisionEncoderOutput(
            z_q=encoder_out.z_q,
            loss=encoder_out.loss,
            indices=encoder_out.indices,
            grid_shape=grid_shape,
            entropy=encoder_out.entropy,
            perplexity=encoder_out.perplexity,
            hidden_states=encoder_out.hidden_states
        )

    def _encode(self, x: torch.Tensor) -> AutoVisionEncoderOutput:
        '''
        Internal encoding method for AutoencoderVisionModel.

        Args:
            x (torch.Tensor): Input image tensor with shape [batch, channels, height, width].

        Returns:
            AutoVisionEncoderOutput: Output containing latent representation and grid_shape.
        '''
        splited_image, grid_shape = self._split_image(x)
        return self.encoder(splited_image, grid_shape)

    def _decode(self, encoder_output: AutoVisionEncoderOutput) -> AutoVisionDecoderOutput:
        '''
        Internal decoding method for AutoencoderVisionModel.

        Args:
            encoder_output (AutoVisionEncoderOutput): Output from encode method containing
                                                      latent representation and grid_shape.

        Returns:
            AutoVisionDecoderOutput: Output containing reconstructed image patches.
        '''
        return self.decoder(encoder_output.z_q, encoder_output.grid_shape)

    def encode(self, x: torch.Tensor) -> AutoVisionEncoderOutput:
        '''
        Encode a full image to latent representation.

        Args:
            x (torch.Tensor): Input image tensor with shape [batch, channels, height, width].

        Returns:
            AutoVisionEncoderOutput: Output containing quantized latent, loss, indices, etc.
        '''
        return self._encode(x)

    def _split_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, tuple]:
        '''
        Split a full image into patches.

        Args:
            image (torch.Tensor): Full image tensor with shape [batch, channels, height, width].

        Returns:
            Tuple[torch.Tensor, tuple]: Tuple of (splited_image tensor, grid_shape tuple).
        '''
        batch_size, channels, height, width = image.shape
        patch_size = self.encoder.patch_size

        num_patches_h = height // patch_size
        num_patches_w = width // patch_size

        grid_shape = (num_patches_h, num_patches_w)

        splited = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        splited = splited.permute(0, 2, 3, 1, 4, 5).contiguous()
        splited = splited.view(batch_size * num_patches_h * num_patches_w, channels, patch_size, patch_size)

        return splited, grid_shape

    def _reconstruct_image(self, patches: torch.Tensor, grid_shape: tuple) -> torch.Tensor:
        '''
        Reconstruct a full image from patches.

        Args:
            patches (torch.Tensor): Patches tensor with shape [num_patches, channels, patch_size, patch_size].
            grid_shape (tuple): Grid shape as (num_patches_h, num_patches_w).

        Returns:
            torch.Tensor: Reconstructed image tensor with shape [batch, channels, height, width].
        '''
        num_patches_h, num_patches_w = grid_shape
        channels, patch_size = patches.shape[1], patches.shape[2]

        batch_size = 1
        patches_per_batch = num_patches_h * num_patches_w
        if patches.shape[0] > patches_per_batch:
            batch_size = patches.shape[0] // patches_per_batch

        patches = patches.view(batch_size, num_patches_h, num_patches_w, channels, patch_size, patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        patches = patches.view(batch_size, channels, num_patches_h * patch_size, num_patches_w * patch_size)

        return patches
