from codon import *
from codon.block.activation import LeakyReLU

from codon.impl.yolo.darknet import Darknet


class YOLOv1(BasicModel):
    '''
    YOLOv1 检测网络:Darknet 主干 + 全连接检测头。

    整体流程:
        图像 [B, 3, 448, 448]
          -> Darknet 主干      -> 特征图 [B, 1024, 7, 7]
          -> Flatten           -> [B, 50176]
          -> FC(50176 -> 4096) -> [B, 4096]
          -> FC(4096 -> 1470)  -> [B, 1470]
          -> reshape           -> [B, 7, 7, 30]  = S x S x (B_boxes * 5 + C)

    最后一维 30 个数的语义(本实现约定):
        [ 0:20]  20 个类别条件概率 Pr(Class_i | Object),每个格子只有一套
        [20:25]  box0 的 x, y, w, h, confidence
        [25:30]  box1 的 x, y, w, h, confidence
    其中 (x, y) 是相对格子左上角的偏移比例,w, h 是相对整张图的比例。

    Attributes:
        backbone (Darknet): 特征提取主干。
        head (nn.Sequential): 两层全连接构成的检测头,最后一层为线性激活。
        S (int): 网格边长。
        B (int): 每个格子预测的框数量。
        C (int): 类别数。
    '''

    def __init__(
        self,
        in_channels: int = 3,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        dropout: float = 0.5,
        fc_dim: int = 4096,
        norm: str = None
    ):
        '''
        初始化 YOLOv1。

        Args:
            in_channels (int, optional): 输入图像通道数。Defaults to 3.
            S (int, optional): 网格边长,VOC 上为 7。Defaults to 7.
            B (int, optional): 每个格子预测的框数量。Defaults to 2.
            C (int, optional): 类别数,VOC 上为 20。Defaults to 20.
            dropout (float, optional): 第一个全连接层后的 dropout 概率,论文为 0.5。Defaults to 0.5.
            fc_dim (int, optional): 检测头中间层维度,论文为 4096。减小它可以大幅压缩参数量。Defaults to 4096.
            norm (str, optional): 主干归一化类型,None 为论文原版。Defaults to None.
        '''
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.dropout_p = dropout
        self.fc_dim = fc_dim

        self.backbone = Darknet(
            in_channels=in_channels,
            norm=norm
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels * S * S, fc_dim),
            LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, S * S * (B * 5 + C)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): 输入图像。形状 [Batch, in_channels, 448, 448]

        Returns:
            torch.Tensor: 检测预测。形状 [Batch, S, S, B * 5 + C],即 [Batch, 7, 7, 30]
        '''
        x = self.backbone(x)
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
        norm: str = None
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
            norm (str, optional): 主干归一化类型。Defaults to None.

        Returns:
            YOLOv1: 构建好的模型实例。
        '''
        in_channels = input_shape[0]
        return YOLOv1(
            in_channels=in_channels,
            S=S,
            B=B,
            C=C,
            dropout=dropout,
            fc_dim=fc_dim,
            norm=norm
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
