from codon import *
from codon.block.activation import LeakyReLU

from codon.impl.yolo.darknet import Darknet


class YOLOv1(BasicModel):
    '''
    YOLOv1 detector: a Darknet backbone followed by a fully connected detection head.

    The overall flow (default input side 448, backbone downsampling by 32 -> 14x14 grid)::

        image [B, 3, 448, 448]
          -> Darknet backbone (num_classes=0)    -> feature map [B, 1024, 14, 14]
          -> one 2x2 stride=2 pooling            -> [B, 1024, 7, 7]   = S x S
          -> Flatten                             -> [B, 50176]
          -> FC(50176 -> 4096) -> leaky ReLU -> Dropout
          -> FC(4096 -> 1470)                    -> [B, 1470]
          -> reshape                             -> [B, 7, 7, 30] = S x S x (B_boxes * 5 + C)

    Meaning of the 30 numbers in the last dimension (as defined by this implementation):
        [ 0:20]  20 class conditional probabilities Pr(Class_i | Object), one set per cell
        [20:25]  x, y, w, h, confidence of box0
        [25:30]  x, y, w, h, confidence of box1
    where (x, y) are the offsets relative to the top-left corner of the cell and w, h are the
    sizes relative to the whole image.

    Three deviations from the paper (all unavoidable once the backbone becomes Darknet-19):

    - The paper's Darknet backbone downsamples by 64 (448 -> 7x7) whereas Darknet-19 downsamples
      by 32 (448 -> 14x14), so one extra 2x2 stride=2 pooling is appended after the backbone to
      squeeze the grid down to S x S and match the paper's 7x7; pass ``S=13`` for a 13x13 grid
      (as in YOLOv2), in which case that single extra pooling is all that is added.
    - The paper's classification head uses average pooling, whereas this implementation keeps the
      fully connected path (Flatten + FC); the parameter count is of the same order as the paper.
    - The number of backbone output channels depends on ``variant``, and the detection head input
      dimension adapts to ``out_channels * s_h * s_w`` with no hard-coding.

    Attributes:
        backbone (Darknet): Feature extractor backbone (num_classes=0, outputs the grid feature map).
        grid_pool (nn.MaxPool2d, optional): Downsamples the backbone output once more to align with
            S x S; None when no extra pooling is needed.
        head (nn.Sequential): Detection head made of two fully connected layers, the last one
            linear.
        S (int): Grid side length.
        B (int): Number of boxes predicted per cell.
        C (int): Number of classes.
        grid_size (int): Side length of the backbone output feature map.
        sub_grid_size (int): Half of grid_size, i.e. the grid side length after the extra pooling.
    '''

    def __init__(
        self,
        in_channels: int = 3,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        dropout: float = 0.5,
        fc_dim: int = 4096,
        norm: str = 'batch',
        variant: str = '19',
        size: int = 448
    ):
        '''
        Initializes YOLOv1.

        Args:
            in_channels (int, optional): Number of input image channels. Defaults to 3.
            S (int, optional): Grid side length; the paper uses 7 on VOC. Defaults to 7.
            B (int, optional): Number of boxes predicted per cell; the paper uses 2. Defaults to 2.
            C (int, optional): Number of classes; 20 on VOC. Defaults to 20.
            dropout (float, optional): Dropout probability after the first fully connected layer;
                the paper uses 0.5. Defaults to 0.5.
            fc_dim (int, optional): Hidden dimension of the detection head; the paper uses 4096.
                Reducing it shrinks the parameter count substantially. Defaults to 4096.
            norm (str, optional): Normalization type of the backbone; the official Darknet
                configuration uses 'batch'. Defaults to 'batch'.
            variant (str, optional): Backbone variant, either '19' (the Darknet-19 of YOLOv2) or
                '53' (the Darknet-53 of YOLOv3). Defaults to '19'.
            size (int, optional): Input image side length; the paper uses 448. Must be a multiple
                of 32. Defaults to 448.

        Raises:
            ValueError: If ``S`` cannot be obtained by downsampling the backbone feature map, or
                if ``size`` is not a multiple of 32.
        '''
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.dropout_p = dropout
        self.fc_dim = fc_dim
        self.variant = variant
        self.size = size

        self.backbone = Darknet(
            variant=variant,
            in_channels=in_channels,
            num_classes=0,
            size=size,
            norm=norm
        )
        self.grid_size = self.backbone.grid_size

        # The backbone downsamples by only 32, so the paper's 448 -> 7x7 still needs one more
        # halving; nothing is appended when the grid is already S.
        if self.grid_size == S:
            self.grid_pool = None
            s_h = s_w = S
        elif self.grid_size % 2 == 0 and self.grid_size // 2 == S:
            self.grid_pool = nn.MaxPool2d(kernel_size=2, stride=2)
            s_h = s_w = S
        else:
            raise ValueError(
                f'the backbone outputs a {self.grid_size}x{self.grid_size} feature map at '
                f'size={size}, and one extra 2x2 pooling only yields '
                f'{self.grid_size // 2}x{self.grid_size // 2}, which cannot give a grid with '
                f'S={S}. Set S to {self.grid_size} or {self.grid_size // 2}, or adjust size.'
            )
        self.sub_grid_size = s_h

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.backbone.coda_channels * s_h * s_w, fc_dim),
            LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, S * S * (B * 5 + C)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): Input image of shape [Batch, in_channels, size, size].

        Returns:
            torch.Tensor: Detection predictions of shape [Batch, S, S, B * 5 + C], which is
                [Batch, 7, 7, 30] by default.
        '''
        x = self.backbone(x)
        if self.grid_pool is not None:
            x = self.grid_pool(x)
        x = self.head(x)
        return x.reshape(-1, self.S, self.S, self.B * 5 + self.C)

    @staticmethod
    def auto_build(
        input_shape: Tuple[int, ...],
        S: int = 7,
        B: int = 2,
        C: int = 20,
        dropout: float = 0.5,
        fc_dim: int = 4096,
        norm: str = 'batch',
        variant: str = '19'
    ) -> 'YOLOv1':
        '''
        Builds a YOLOv1 automatically from the input shape.

        Args:
            input_shape (Tuple[int, ...]): Input shape without the batch dimension, e.g.
                (3, 448, 448).
            S (int, optional): Grid side length. Defaults to 7.
            B (int, optional): Number of boxes predicted per cell. Defaults to 2.
            C (int, optional): Number of classes. Defaults to 20.
            dropout (float, optional): Dropout probability. Defaults to 0.5.
            fc_dim (int, optional): Hidden dimension of the detection head. Defaults to 4096.
            norm (str, optional): Normalization type of the backbone. Defaults to 'batch'.
            variant (str, optional): Backbone variant, '19' or '53'. Defaults to '19'.

        Returns:
            YOLOv1: The constructed model instance.
        '''
        return YOLOv1(
            in_channels=input_shape[0],
            S=S,
            B=B,
            C=C,
            dropout=dropout,
            fc_dim=fc_dim,
            norm=norm,
            variant=variant,
            size=input_shape[-1]
        )


if __name__ == '__main__':
    from codon.block.activation import LeakyReLU as CodonLeakyReLU

    model = YOLOv1()
    x = torch.randn(2, 3, 448, 448)
    with torch.no_grad():
        y = model(x)
    print('out:', tuple(y.shape))
    print('flat:', tuple(y.reshape(2, -1).shape))
    print('act slopes:', {m.negative_slope for m in model.modules() if isinstance(m, CodonLeakyReLU)})
    print('params(M):', round(sum(p.numel() for p in model.parameters()) / 1e6, 2))
    print('has negative:', bool((y < 0).any()), '| range:', float(y.min()), float(y.max()))
    print(model.count_params(human_readable=True))

    # Darknet-53 backbone + a 13x13 grid (YOLOv2-style grid)
    model53 = YOLOv1(variant='53', S=13, fc_dim=1024)
    with torch.no_grad():
        y53 = model53(torch.randn(1, 3, 416, 416))
    print('v53 out:', tuple(y53.shape), '| params', model53.count_params(human_readable=True))
