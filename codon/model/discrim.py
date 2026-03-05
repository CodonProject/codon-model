from codon.base import *


class PatchDiscriminator(BasicModel):
    '''
    PatchGAN discriminator.

    The output is not a scalar, but an N x N matrix, where each point represents
    whether the corresponding patch is real or fake.

    Attributes:
        main (nn.Sequential): The main sequential model.
    '''

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, num_layers: int = 3) -> None:
        '''
        Initialize the PatchDiscriminator.

        Args:
            in_channels (int): Number of input channels. Defaults to 3.
            hidden_dim (int): Base number of filters (channels) in the discriminator. Defaults to 64.
            num_layers (int): Number of layers in the discriminator. Defaults to 3.
        '''
        super().__init__()

        sequence = [
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        channel_mult = 1
        channel_mult_prev = 1
        for n in range(1, num_layers):
            channel_mult_prev = channel_mult
            channel_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(hidden_dim * channel_mult_prev, hidden_dim * channel_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim * channel_mult),
                nn.LeakyReLU(0.2, True)
            ]

        channel_mult_prev = channel_mult
        channel_mult = min(2 ** num_layers, 8)

        sequence += [
            nn.Conv2d(hidden_dim * channel_mult_prev, hidden_dim * channel_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * channel_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(hidden_dim * channel_mult, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.main = nn.Sequential(*sequence)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Defines the computation performed at every call.

        Args:
            input_tensor (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the discriminator.
        '''
        return self.main(input_tensor)
