from codon import *
from codon.block.conv import ConvBlock
from codon.block.activation import LeakyReLU


#: YOLOv1 论文公式 (2):所有非输出层统一使用斜率 0.1 的 leaky ReLU。
#: 注意 codon.block.activation.LeakyReLU 的默认斜率是 0.01,必须显式传 0.1。
#: 该激活无参数、无状态,全网络共享同一个实例。
act = LeakyReLU(0.1)


class Repetition(BasicModel):
    '''
    YOLOv1 主干中的重复块:1x1 reduction + 3x3 convolution。

    论文用 1x1 卷积压缩通道,再用 3x3 卷积提取空间特征,替代 GoogLeNet 的 Inception 模块。
    两个卷积都用 padding=1(same padding)保持空间尺寸,下采样只发生在池化和 stride=2 的卷积处。

    Attributes:
        features (nn.Sequential): [ConvBlock(1x1), ConvBlock(3x3)] 的序列。
    '''

    def __init__(
        self,
        outer_dim: int,
        inner_dim: int,
        norm: str = None,
        activation: Union[str, BasicModel] = act
    ):
        '''
        初始化 Repetition 块。

        Args:
            outer_dim (int): 块的输入/输出通道数。
            inner_dim (int): 1x1 reduction 后的中间通道数。
            norm (str, optional): 归一化类型,None 为论文原版(无归一化)。Defaults to None.
            activation (Union[str, BasicModel], optional): 激活函数,字符串或已实例化的模块。Defaults to act.
        '''
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(
                in_channels=outer_dim,
                out_channels=inner_dim,
                kernel_size=1,
                norm=norm,
                activation=activation
            ),
            ConvBlock(
                in_channels=inner_dim,
                out_channels=outer_dim,
                kernel_size=3,
                padding=1,
                norm=norm,
                activation=activation
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): 输入特征图。形状 [Batch, outer_dim, H, W]

        Returns:
            torch.Tensor: 输出特征图。形状 [Batch, outer_dim, H, W](空间尺寸不变)
        '''
        return self.features(x)


class Darknet(BasicModel):
    '''
    YOLOv1 的主干特征提取网络(23 个卷积层 + 4 个最大池化)。

    结构固定为 **6 次下采样 = 1 次 stride=2 卷积(stem)+ 4 次 maxpool + 1 次 stride=2 卷积(coda)**,
    只把输入边长 ``size`` 参数化::

        splitter : conv 7x7 s2 p3 -> pool -> conv 3x3 p1 -> pool
        prelude  : conv 1x1 -> conv 3x3 -> conv 1x1 -> conv 3x3 -> pool
        body     : Repetition x4 -> conv 1x1 -> conv 3x3 -> pool
        coda     : Repetition x2 -> conv 3x3 s2 -> conv 3x3 -> conv 3x3

    两个阶段的用法(论文 2.1 / 2.2 节):

    ============  ==================  ===================  ==========================
    阶段           输入                特征图               说明
    ============  ==================  ===================  ==========================
    检测(微调)     448 x 448           7 x 7 x 1024         与论文一致,64 倍下采样整除
    预训练         224 x 224           4 x 4 x 1024         224/64 = 3.5,向上取整到 4
    ============  ==================  ===================  ==========================

    关于"论文图三把 224 阶段的特征图也标成 7x7":那是取整效应造成的偏差。224 想走到 7x7
    需要 5 次池化(112/32 = 3.5,必须 ceil 到 4 再池化),而 maxpool 在 3x3 上仍会生效,
    会吃掉一层卷积的空间上下文。本实现选择 **结构对齐、特征图边长按算术实际计算**,
    而不是硬凑 7x7;分类头用 ``1024 * grid_size ** 2`` 自适应即可。

    Attributes:
        splitter (nn.Sequential): conv7x7 s2 + pool + conv3x3 + pool。
        prelude (nn.Sequential): 4 个卷积(1x1/3x3 交替)+ pool。
        body (nn.Sequential): 4 个 Repetition + 2 个卷积 + pool。
        coda (nn.Sequential): 2 个 Repetition + 1 个 stride=2 卷积 + 2 个卷积。
        out_channels (int): 输出特征图通道数,固定 1024。
        grid_size (int): 输出特征图的空间边长,由 size 算术推导。
        coda_stride (int): coda 下采样卷积的步长,固定 2。
        num_pools (int): maxpool 数量,固定 4。
    '''

    #: 主干中 maxpool 的数量(固定)。
    NUM_POOLS = 4
    #: coda 中下采样卷积的步长(固定)。
    CODA_STRIDE = 2

    def __init__(
        self,
        in_channels: int = 3,
        size: int = 448,
        grid_size: Optional[int] = None,
        norm: str = None,
        activation: Union[str, BasicModel] = act
    ):
        '''
        初始化 Darknet 主干。

        Args:
            in_channels (int, optional): 输入图像通道数。Defaults to 3.
            size (int, optional): 输入图像边长。448 用于检测,224 用于 ImageNet 预训练。Defaults to 448.
            grid_size (Optional[int], optional): 期望的输出特征图边长,仅用于一致性校验。
                None 时由 ``size`` 算术推导。Defaults to None.
            norm (str, optional): 归一化类型。论文原版为 None(无归一化)。Defaults to None.
            activation (Union[str, BasicModel], optional): 激活函数。Defaults to act(斜率 0.1 的 leaky ReLU)。

        Raises:
            ValueError: 当 ``size`` 不是偶数,或推导出的特征图与 ``grid_size`` 不符时。
        '''
        super().__init__()
        if size % 2 != 0:
            raise ValueError(f'输入边长必须是偶数(主干含多次 2 倍下采样),收到 {size}')

        self.in_channels = in_channels
        self.size = size
        self.norm = norm
        self.activation = activation
        self.out_channels = 1024

        self.num_pools, self.coda_stride, self.grid_size = self.resolve_pools(size, grid_size)

        self.splitter = self._build_splitter()
        self.prelude = self._build_prelude()
        self.body = self._build_body()
        self.coda = self._build_coda()

    # ------------------------------------------------------------------
    # 结构推导
    # ------------------------------------------------------------------

    @classmethod
    def resolve_pools(
        cls,
        size: int,
        grid_size: Optional[int] = None
    ) -> Tuple[int, int, int]:
        '''
        推导给定输入边长下的 (池化次数, coda 步长, 特征图边长)。

        下采样算术(``ceil`` 精确建模 maxpool 对奇数尺寸向下取整的行为)::

            after_stem  = ceil(size / 2)                          # conv 7x7 s2
            after_pools = ceil(after_stem / 2 ** NUM_POOLS)       # 4 次 maxpool
            grid        = ceil(after_pools / CODA_STRIDE)         # coda conv 3x3 s2

        Args:
            size (int): 输入图像边长。
            grid_size (Optional[int]): 期望的特征图边长,给定时做一致性校验。

        Returns:
            Tuple[int, int, int]: (num_pools, coda_stride, grid_size)

        Raises:
            ValueError: 当推导结果与给定的 ``grid_size`` 不符时。
        '''
        after_stem = math.ceil(size / 2)
        grid = math.ceil(math.ceil(after_stem / 2 ** cls.NUM_POOLS) / cls.CODA_STRIDE)

        if grid_size is not None and grid_size != grid:
            achievable = sorted({
                math.ceil(math.ceil(after_stem / 2 ** p) / s)
                for p in range(6) for s in (1, 2)
                if math.ceil(math.ceil(after_stem / 2 ** p) / s) >= 3
            })
            raise ValueError(
                f'输入边长 {size} 在本结构下得到 {grid}x{grid} 的特征图,'
                f'与要求的 {grid_size}x{grid_size} 不符。'
                f'该尺寸通过调整池化次数与步长可达到的边长有: {achievable}。'
            )

        return cls.NUM_POOLS, cls.CODA_STRIDE, grid

    # ------------------------------------------------------------------
    # 各段构建
    # ------------------------------------------------------------------

    def _build_splitter(self) -> nn.Sequential:
        '''构建 splitter 段:conv 7x7 s2 p3 -> pool -> conv 3x3 p1 -> pool。'''
        return nn.Sequential(
            ConvBlock(
                in_channels=self.in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                activation=self.activation,
                norm=self.norm
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(
                in_channels=64,
                out_channels=192,
                padding=1,
                activation=self.activation,
                norm=self.norm
            ),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def _build_prelude(self) -> nn.Sequential:
        '''构建 prelude 段:conv 1x1 -> conv 3x3 -> conv 1x1 -> conv 3x3 -> pool。'''
        return nn.Sequential(
            ConvBlock(
                in_channels=192,
                out_channels=128,
                kernel_size=1,
                activation=self.activation,
                norm=self.norm
            ),
            ConvBlock(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
                activation=self.activation,
                norm=self.norm
            ),
            ConvBlock(
                in_channels=256,
                out_channels=256,
                kernel_size=1,
                activation=self.activation,
                norm=self.norm
            ),
            ConvBlock(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                padding=1,
                activation=self.activation,
                norm=self.norm
            ),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def _build_body(self) -> nn.Sequential:
        '''构建 body 段:4 个 Repetition + conv 1x1 + conv 3x3 -> pool。'''
        layers = [
            Repetition(
                outer_dim=512,
                inner_dim=256,
                norm=self.norm,
                activation=self.activation
            ) for _ in range(4)
        ]
        layers += [
            ConvBlock(
                in_channels=512,
                out_channels=512,
                kernel_size=1,
                activation=self.activation,
                norm=self.norm
            ),
            ConvBlock(
                in_channels=512,
                out_channels=1024,
                kernel_size=3,
                padding=1,
                activation=self.activation,
                norm=self.norm
            ),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def _build_coda(self) -> nn.Sequential:
        '''构建 coda 段:2 个 Repetition + conv 3x3 s2 + conv 3x3 + conv 3x3。'''
        layers = [
            Repetition(
                outer_dim=1024,
                inner_dim=512,
                norm=self.norm,
                activation=self.activation
            ) for _ in range(2)
        ]
        layers.append(
            ConvBlock(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                padding=1,
                stride=self.coda_stride,
                norm=self.norm,
                activation=self.activation
            )
        )
        layers += [
            ConvBlock(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                padding=1,
                norm=self.norm,
                activation=self.activation
            ) for _ in range(2)
        ]
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): 输入图像。形状 [Batch, in_channels, size, size]

        Returns:
            torch.Tensor: 特征图。形状 [Batch, 1024, grid_size, grid_size]
        '''
        x = self.splitter(x)
        x = self.prelude(x)
        x = self.body(x)
        x = self.coda(x)

        assert x.shape[-2:] == (self.grid_size, self.grid_size), (
            f'特征图空间尺寸应为 {self.grid_size}x{self.grid_size}, 实际 {tuple(x.shape[-2:])}。'
            f'请确认输入边长是否为 {self.size}(主干按 size 推导下采样次数)。'
        )
        return x


class DarknetClassifier(BasicModel):
    '''
    ImageNet 预训练用的 Darknet 分类网络(论文 2.2 节)。

    论文原文:"we use the first 20 convolutional layers from Figure 3 followed by an
    average-pooling layer and a fully connected layer."

    实现为论文图三的路径:Flatten -> FC(1024 * grid_size^2 -> fc_dim) -> leaky ReLU
    -> Dropout -> FC(fc_dim -> num_classes)。两个全连接的权重都随机初始化,用于训练 1000 类。

    Attributes:
        backbone (Darknet): 特征提取主干。
        head (nn.Sequential): 分类头。
        num_classes (int): 类别数。
        grid_size (int): 主干输出特征图边长。
    '''

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        size: int = 224,
        norm: str = None,
        activation: Union[str, BasicModel] = act,
        fc_dim: int = 4096,
        dropout: float = 0.5
    ):
        '''
        初始化 Darknet 分类网络。

        Args:
            in_channels (int, optional): 输入图像通道数。Defaults to 3.
            num_classes (int, optional): 分类类别数。Defaults to 1000.
            size (int, optional): 输入图像边长,预训练用 224。Defaults to 224.
            norm (str, optional): 归一化类型,None 为论文原版。Defaults to None.
            activation (Union[str, BasicModel], optional): 激活函数。Defaults to act.
            fc_dim (int, optional): 全连接层维度,论文为 4096。Defaults to 4096.
            dropout (float, optional): dropout 概率,论文为 0.5。Defaults to 0.5.
        '''
        super().__init__()
        self.num_classes = num_classes
        self.fc_dim = fc_dim
        self.size = size

        self.backbone = Darknet(
            in_channels=in_channels,
            size=size,
            norm=norm,
            activation=activation
        )
        self.grid_size = self.backbone.grid_size

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels * self.grid_size ** 2, fc_dim),
            LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): 输入图像。形状 [Batch, in_channels, size, size]

        Returns:
            torch.Tensor: 类别 logits。形状 [Batch, num_classes]
        '''
        x = self.backbone(x)
        return self.head(x)
