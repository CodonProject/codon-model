from codon import *
from codon.block.activation import LeakyReLU

from codon.impl.yolo.darknet import Darknet


class YOLOv1(BasicModel):
    '''
    YOLOv1 检测网络:Darknet 主干 + 全连接检测头。

    整体流程(默认输入边长 448、主干 32 倍下采样 -> 14x14 网格)::

        图像 [B, 3, 448, 448]
          -> Darknet 主干(num_classes=0)        -> 特征图 [B, 1024, 14, 14]
          -> 一次 2x2 stride=2 池化               -> [B, 1024, 7, 7]   = S x S
          -> Flatten                              -> [B, 50176]
          -> FC(50176 -> 4096) -> leaky ReLU -> Dropout
          -> FC(4096 -> 1470)                     -> [B, 1470]
          -> reshape                              -> [B, 7, 7, 30] = S x S x (B_boxes * 5 + C)

    最后一维 30 个数的语义(本实现约定):
        [ 0:20]  20 个类别条件概率 Pr(Class_i | Object),每个格子只有一套
        [20:25]  box0 的 x, y, w, h, confidence
        [25:30]  box1 的 x, y, w, h, confidence
    其中 (x, y) 是相对格子左上角的偏移比例,w, h 是相对整张图的比例。

    与论文的三点出入(都是把主干换成 Darknet-19 之后的必然结果):

    - 论文主干 Darknet 是 64 倍下采样(448 -> 7x7),Darknet-19 是 32 倍(448 -> 14x14),
      所以这里在主干之后补一次 2x2 stride=2 池化把网格压到 S x S,与论文的 7x7 对齐;
      想要 13x13 网格(与 YOLOv2 一致)时传 ``S=13``,此时仍然只补一次池化。
    - 论文分类头用平均池化,本实现沿用全连接路径(Flatten + FC),参数量与论文一致量级。
    - 主干输出通道数取决于 ``variant``,检测头的输入维度按 ``out_channels * s_h * s_w`` 自适应,
      不做任何硬编码。

    Attributes:
        backbone (Darknet): 特征提取主干(num_classes=0,输出网格特征图)。
        grid_pool (nn.MaxPool2d, optional): 把主干输出再下采样一次以对齐 S x S;不需要时为 None。
        head (nn.Sequential): 两层全连接构成的检测头,最后一层为线性激活。
        S (int): 网格边长。
        B (int): 每个格子预测的框数量。
        C (int): 类别数。
        grid_size (int): 主干输出特征图边长。
        sub_grid_size (int): grid_size 的一半,即补池化之后的网格边长。
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
        初始化 YOLOv1。

        Args:
            in_channels (int, optional): 输入图像通道数。Defaults to 3.
            S (int, optional): 网格边长,VOC 上论文用 7。Defaults to 7.
            B (int, optional): 每个格子预测的框数量,论文为 2。Defaults to 2.
            C (int, optional): 类别数,VOC 上为 20。Defaults to 20.
            dropout (float, optional): 第一个全连接层后的 dropout 概率,论文为 0.5。Defaults to 0.5.
            fc_dim (int, optional): 检测头中间层维度,论文为 4096。减小它可以大幅压缩参数量。
                Defaults to 4096.
            norm (str, optional): 主干归一化类型,官方 Darknet 配置为 'batch'。Defaults to 'batch'.
            variant (str, optional): 主干变体,'19'(YOLOv2 的 Darknet-19)或 '53'(YOLOv3 的
                Darknet-53)。Defaults to '19'.
            size (int, optional): 输入图像边长,论文为 448。必须是 32 的倍数。Defaults to 448.

        Raises:
            ValueError: 当 ``S`` 不能由主干特征图下采样得到,或 ``size`` 不是 32 的倍数时。
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

        # 主干只下采样 32 倍,论文的 448 -> 7x7 还需要再翻一倍;网格已经是 S 时就不补。
        if self.grid_size == S:
            self.grid_pool = None
            s_h = s_w = S
        elif self.grid_size % 2 == 0 and self.grid_size // 2 == S:
            self.grid_pool = nn.MaxPool2d(kernel_size=2, stride=2)
            s_h = s_w = S
        else:
            raise ValueError(
                f'主干在 size={size} 下输出 {self.grid_size}x{self.grid_size} 的特征图,'
                f'再补一次 2x2 池化只能得到 {self.grid_size // 2}x{self.grid_size // 2},'
                f'无法得到 S={S} 的网格。请把 S 设为 {self.grid_size} 或 {self.grid_size // 2},'
                f'或调整 size。'
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
            x (torch.Tensor): 输入图像。形状 [Batch, in_channels, size, size]

        Returns:
            torch.Tensor: 检测预测。形状 [Batch, S, S, B * 5 + C],默认即 [Batch, 7, 7, 30]
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
        根据输入形状自动构建 YOLOv1。

        Args:
            input_shape (Tuple[int, ...]): 输入形状(不含 batch),例如 (3, 448, 448)。
            S (int, optional): 网格边长。Defaults to 7.
            B (int, optional): 每个格子预测的框数量。Defaults to 2.
            C (int, optional): 类别数。Defaults to 20.
            dropout (float, optional): dropout 概率。Defaults to 0.5.
            fc_dim (int, optional): 检测头中间层维度。Defaults to 4096.
            norm (str, optional): 主干归一化类型。Defaults to 'batch'.
            variant (str, optional): 主干变体 '19' 或 '53'。Defaults to '19'.

        Returns:
            YOLOv1: 构建好的模型实例。
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

    # 用 Darknet-53 主干 + 13x13 网格(YOLOv2 风格的网格)
    model53 = YOLOv1(variant='53', S=13, fc_dim=1024)
    with torch.no_grad():
        y53 = model53(torch.randn(1, 3, 416, 416))
    print('v53 out:', tuple(y53.shape), '| params', model53.count_params(human_readable=True))
