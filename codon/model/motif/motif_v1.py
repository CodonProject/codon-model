from codon.base import *

import math

from codon.model.resnet import ResNet

from codon.block import (
    LookupFreeQuantization, LookupFreeQuantizationOutput,
    InterleavedRotaryEmbedding,
    MultiHeadAttention, AttentionOutput,
    ConvBlock, PixelShuffleUpSample
)

from dataclasses import dataclass

@dataclass
class MotifV1EncoderOutput:
    '''
    Output of the MotifV1Encoder.

    Attributes:
        z_q (torch.Tensor): Quantized latent tensor with shape [num_patches, latent_dim].
        loss (torch.Tensor): Total quantization loss from codebook.
        indices (torch.Tensor): Integer indices of quantized bits with shape [num_patches].
        entropy (torch.Tensor): Average bit-wise entropy from codebook.
        perplexity (torch.Tensor): Perplexity calculated as 2^entropy.
        hidden_states (torch.Tensor): Hidden states before quantization with shape [num_patches, latent_dim].
    '''
    z_q: torch.Tensor
    loss: torch.Tensor
    indices: torch.Tensor
    entropy: torch.Tensor
    perplexity: torch.Tensor
    hidden_states: torch.Tensor
    grid_shape: tuple


@dataclass
class MotifV1DecoderOutput:
    '''
    Output of the MotifV1Decoder.

    Attributes:
        reconstructed_image (torch.Tensor): Reconstructed output image patches with shape [num_patches, out_features, patch_size, patch_size].
        hidden_states (torch.Tensor): Hidden states after attention but before upsampling with shape [num_patches, latent_dim].
    '''
    reconstructed_image: torch.Tensor
    hidden_states: torch.Tensor


@dataclass
class MotifV1Output:
    '''
    Output of the MotifV1 model.

    Attributes:
        reconstructed_image (torch.Tensor): Reconstructed output image patches with shape [num_patches, out_features, patch_size, patch_size].
        z_q (torch.Tensor): Quantized latent tensor with shape [num_patches, latent_dim].
        quantization_loss (torch.Tensor): Total quantization loss from codebook.
        indices (torch.Tensor): Integer indices of quantized bits with shape [num_patches].
        entropy (torch.Tensor): Average bit-wise entropy from codebook.
        perplexity (torch.Tensor): Perplexity calculated as 2^entropy.
        encoder_hidden_states (torch.Tensor): Hidden states from encoder before quantization.
        decoder_hidden_states (torch.Tensor): Hidden states from decoder after attention.
    '''
    reconstructed_image: torch.Tensor
    z_q: torch.Tensor
    quantization_loss: torch.Tensor
    indices: torch.Tensor
    entropy: torch.Tensor
    perplexity: torch.Tensor
    encoder_hidden_states: torch.Tensor
    decoder_hidden_states: torch.Tensor


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
    ) -> MotifV1EncoderOutput:
        '''
        Forward pass of the MotifV1Encoder.

        Args:
            splited_image (torch.Tensor): Input tensor with shape [num_patches, channels, patch_size, patch_size].
            grid_shape (tuple): Grid shape as (num_patches_h, num_patches_w).
            rope_emb (InterleavedRotaryEmbedding, optional): 2D rotary positional embedding. Defaults to None.

        Returns:
            MotifV1EncoderOutput: The output containing quantized latent, loss, indices, etc.
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

        return MotifV1EncoderOutput(
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
    ) -> MotifV1DecoderOutput:
        '''
        Forward pass of the MotifV1Decoder.

        Args:
            z_q (torch.Tensor): Input quantized latent tensor with shape [num_patches, latent_dim].
            grid_shape (tuple): Grid shape as (num_patches_h, num_patches_w).
            rope_emb (InterleavedRotaryEmbedding, optional): 2D rotary positional embedding. Defaults to None.

        Returns:
            MotifV1DecoderOutput: The output containing reconstructed image patches and hidden states.
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

        return MotifV1DecoderOutput(
            reconstructed_image=output,
            hidden_states=z_hidden
        )


class MotifV1(BasicModel):
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
        
    def forward(self, splited_image: torch.Tensor, grid_shape: tuple) -> MotifV1Output:
        '''
        Forward pass of the MotifV1 model.

        Args:
            splited_image (torch.Tensor): Input tensor with shape [num_patches, channels, patch_size, patch_size].
            grid_shape (tuple): Grid shape as (num_patches_h, num_patches_w).

        Returns:
            MotifV1Output: Output dataclass containing reconstructed image and latent details.
        '''
        encoder_out: MotifV1EncoderOutput = self.encoder(splited_image, grid_shape)
        decoder_out: MotifV1DecoderOutput = self.decoder(encoder_out.z_q, grid_shape)
        
        return MotifV1Output(
            reconstructed_image=decoder_out.reconstructed_image,
            z_q=encoder_out.z_q,
            quantization_loss=encoder_out.loss,
            indices=encoder_out.indices,
            entropy=encoder_out.entropy,
            perplexity=encoder_out.perplexity,
            encoder_hidden_states=encoder_out.hidden_states,
            decoder_hidden_states=decoder_out.hidden_states
        )
