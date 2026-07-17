from codon import *
from codon.block.conv import ConvBlock


class PatchDiscriminator(BasicModel):
    '''
    PatchGAN discriminator.

    The output is not a scalar, but an N x N matrix, where each point represents
    whether the corresponding patch is real or fake.

    Attributes:
        main (nn.Sequential): The main sequential model.
    '''

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 3,
        norm: str = 'batch',
        activation: str = 'leaky_relu',
        leaky_relu: float = 0.2
    ) -> None:
        '''
        Initialize the PatchDiscriminator.

        Args:
            in_channels (int): Number of input channels. Defaults to 3.
            hidden_dim (int): Base number of filters (channels) in the discriminator. Defaults to 64.
            num_layers (int): Number of layers in the discriminator. Defaults to 3.
            norm (str, optional): Normalization type. Defaults to 'batch'.
            activation (str, optional): Activation function type. Defaults to 'leaky_relu'.
            leaky_relu (float, optional): Negative slope for LeakyReLU. Defaults to 0.2.
        '''
        super().__init__()

        sequence = [
            ConvBlock(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                dim=2,
                norm=None,
                activation=activation,
                leaky_relu=leaky_relu,
                bias=True
            )
        ]

        channel_mult = 1
        channel_mult_prev = 1
        for n in range(1, num_layers):
            channel_mult_prev = channel_mult
            channel_mult = min(2 ** n, 8)
            sequence += [
                ConvBlock(
                    in_channels=hidden_dim * channel_mult_prev,
                    out_channels=hidden_dim * channel_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    dim=2,
                    norm=norm,
                    activation=activation,
                    leaky_relu=leaky_relu,
                    bias=False
                )
            ]

        channel_mult_prev = channel_mult
        channel_mult = min(2 ** num_layers, 8)

        sequence += [
            ConvBlock(
                in_channels=hidden_dim * channel_mult_prev,
                out_channels=hidden_dim * channel_mult,
                kernel_size=4,
                stride=1,
                padding=1,
                dim=2,
                norm=norm,
                activation=activation,
                leaky_relu=leaky_relu,
                bias=False
            ),
            ConvBlock(
                in_channels=hidden_dim * channel_mult,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
                dim=2,
                norm=None,
                activation=None,
                bias=True
            )
        ]

        self.main = nn.Sequential(*sequence)

    @staticmethod
    def auto_build(
        in_channels: int,
        hidden_dim: int,
        image_size: int,
        norm: str = 'batch',
        activation: str = 'leaky_relu',
        leaky_relu: float = 0.2
    ) -> 'PatchDiscriminator':
        '''
        Automatically builds a PatchDiscriminator based on the image size.

        Args:
            in_channels (int): Number of input channels.
            hidden_dim (int): Base number of filters (channels).
            image_size (int): Size of the input image.
            norm (str, optional): Normalization type. Defaults to 'batch'.
            activation (str, optional): Activation function type. Defaults to 'leaky_relu'.
            leaky_relu (float, optional): Negative slope for LeakyReLU. Defaults to 0.2.

        Returns:
            PatchDiscriminator: The constructed PatchDiscriminator.
        '''
        num_layers = int(math.log2(image_size / 32))
        num_layers = max(1, num_layers)

        return PatchDiscriminator(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            norm=norm,
            activation=activation,
            leaky_relu=leaky_relu
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Defines the computation performed at every call.

        Args:
            input_tensor (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the discriminator.
        '''
        return self.main(input_tensor)
