from codon import *
from codon.block.conv import ConvBlock
from codon.block.activation import LeakyReLU


#: YOLOv1 论文公式 (2) 与 Darknet 官方配置里的非输出层统一使用斜率 0.1 的 leaky ReLU。
#: 注意 codon.block.activation.LeakyReLU 的默认斜率是 0.01,必须显式传 0.1。
#: 该激活无参数、无状态,全网络共享同一个实例(``Darknet`` 会把它原样复用而不是按名字重建)。
act = LeakyReLU(0.1)

#: Darknet 主干下采样 2 ** 5 = 32 倍,因此输入边长应是 32 的倍数。
GLOBAL_DOWNSAMPLE = 32


# ----------------------------------------------------------------------
# 通用构件
# ----------------------------------------------------------------------


def darknet_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    norm: str = 'batch',
    activation: Union[str, BasicModel] = act,
    bias: bool = None
) -> ConvBlock:
    '''
    构建一个 Darknet 卷积单元,即 cfg 文件里的一个 ``[convolutional]`` 段。

    Darknet 的约定是 ``pad = size // 2``(same padding),所以 3x3 卷积用 padding=1、
    1x1 卷积用 padding=0,再由 stride 决定是否下采样。有归一化时卷积不需要 bias
    (bn 的 beta 已经承担了偏置),这一点与原版 darknet 的 ``batch_normalize`` 行为一致。

    注意 ``codon.block.conv.ConvBlock`` 的默认值是 ``norm='batch'`` 与
    ``activation='relu'``,与 Darknet 不同,所以这里所有参数都显式传递。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int): 卷积核边长。
        stride (int, optional): 步长,2 表示下采样一半。Defaults to 1.
        norm (str, optional): 归一化类型,Darknet 原版为 'batch'。Defaults to 'batch'.
        activation (Union[str, BasicModel], optional): 激活函数。Defaults to act(斜率 0.1)。
        bias (bool, optional): 是否使用卷积偏置。None 表示按 ``norm is None`` 自动决定。
            Defaults to None.

    Returns:
        ConvBlock: 卷积 + 归一化 + 激活 的组合单元。
    '''
    if bias is None:
        bias = norm is None

    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        bias=bias,
        dim=2,
        norm=norm,
        activation=activation,
        dropout=0.0,
        pre_norm=False
    )


def darknet_linear_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 1,
    stride: int = 1
) -> nn.Module:
    '''
    构建一个纯线性卷积(无归一化、无激活),用于残差块的相加前与分类头。

    ``codon.block.activation.get_activation`` 不接受 None,而 ``ConvBlock`` 在这种
    情况下没法表达「不加激活」,所以这里直接返回 ``nn.Conv2d``:残差块的 shortcut 相加
    与分类头的 logits 都需要线性输出。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int, optional): 卷积核边长。Defaults to 1.
        stride (int, optional): 步长。Defaults to 1.

    Returns:
        nn.Module: ``nn.Conv2d`` 实例(带偏置,因为后面没有归一层)。
    '''
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2
    )


def activation_name(activation: Union[str, None, BasicModel]) -> Optional[str]:
    '''
    把激活配置归一化成字符串描述。

    建模时用的是激活对象本身(见 ``Darknet.activation_module``),这个字符串只用于展示与日志,
    所以要保证同一份配置写两次得到同一个名字。

    Args:
        activation (Union[str, None, BasicModel]): 激活配置。

    Returns:
        Optional[str]: 字符串原样返回,``None`` 原样返回,模块取其类名小写(如 'leakyrelu')。
    '''
    if activation is None or isinstance(activation, str):
        return activation
    return type(activation).__name__.lower()


# ----------------------------------------------------------------------
# Darknet 主干(YOLOv2 / YOLOv3)
# ----------------------------------------------------------------------


class ResidualBlock(BasicModel):
    '''
    Darknet53 的瓶颈残差块:1x1 压缩 -> 3x3 扩展 -> shortcut。

    对应官方 cfg 里连续出现的 ``[convolutional] filters=C size=1``、
    ``[convolutional] filters=C size=3`` 与 ``[shortcut] activation=linear``:

        y = x + conv3x3( activation( conv1x1( activation(x) ) ) )

    与 ``codon.block.conv.ResBasicBlock`` 的差异:

    - shortcut 是 ``activation=linear``:相加之后**不**再统一激活,激活只在每个卷积之后;
      ResBasicBlock 的 'original' 变体则是相加后再统一激活。因此扩展卷积 ``conv2`` 是纯线性的。
    - Darknet53 的瓶颈比是 1:1(1x1 与 3x3 的输出通道相同,CIFAR 版 Darknet 才有 C/2 的压缩比),
      所以 1x1 与 3x3 的通道数分别给出。
    - 不需要 downsample 分支:下采样由每个块开头的 stride=2 卷积完成,残差块本身保持空间尺寸。

    Attributes:
        conv1 (ConvBlock): 1x1 压缩卷积(带激活)。
        conv2 (nn.Conv2d): 3x3 扩展卷积(线性输出,相加前不激活)。
        in_channels (int): 块的输入通道数。
        inner_channels (int): 1x1 压缩后的瓶颈通道数。
        out_channels (int): 块输出(shortcut 相加后)的通道数。
    '''

    def __init__(
        self,
        in_channels: int,
        inner_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        norm: str = 'batch',
        activation: Union[str, BasicModel] = act
    ):
        '''
        初始化瓶颈残差块。

        Args:
            in_channels (int): 块的输入通道数。
            inner_channels (Optional[int], optional): 1x1 压缩后的瓶颈通道数。
                省略时取 ``out_channels // 2``。Defaults to None.
            out_channels (Optional[int], optional): 3x3 扩展后的输出通道数,也就是残差相加后的
                通道数。省略时取 ``in_channels``(Darknet53 的 1:1 瓶颈)。Defaults to None.
            norm (str, optional): 归一化类型。Defaults to 'batch'.
            activation (Union[str, BasicModel], optional): 激活函数。Defaults to act。

        Raises:
            ValueError: 当任一通道数不是正整数,或 ``out_channels != in_channels``
                (shortcut 无法相加)时。
        '''
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels
        inner_channels = out_channels // 2 if inner_channels is None else inner_channels
        if inner_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f'瓶颈通道数必须是正整数,收到 inner={inner_channels}, out={out_channels}。'
            )
        if out_channels != in_channels:
            raise ValueError(
                f'shortcut 要求输入输出通道一致,收到 in={in_channels}, out={out_channels};'
                f'需要改通道数时请在块的开头用卷积完成。'
            )

        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.out_channels = out_channels

        self.conv1 = darknet_conv(
            in_channels=in_channels,
            out_channels=inner_channels,
            kernel_size=1,
            stride=1,
            norm=norm,
            activation=activation
        )
        self.conv2 = darknet_linear_conv(
            in_channels=inner_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): 输入特征图。形状 [Batch, in_channels, H, W]

        Returns:
            torch.Tensor: 输出特征图。形状 [Batch, out_channels, H, W](空间尺寸不变)
        '''
        return x + self.conv2(self.conv1(x))


# ----------------------------------------------------------------------
# Darknet 主干(YOLOv2 / YOLOv3)
# ----------------------------------------------------------------------

#: conv 块:块内每个元素是一个普通卷积。
CONV_BLOCK = 'conv'
#: residual 块:块首是下采样卷积,其后每两个元素是一个瓶颈残差块。
RESIDUAL_BLOCK = 'residual'

#: 每种块类型在「全整数写法」里默认使用的核边长序列(按元素循环套用)。
DEFAULT_KERNELS: Dict[str, Tuple[int, ...]] = {
    CONV_BLOCK: (3,),
    RESIDUAL_BLOCK: (3, 1, 3),
}

#: 单个段里允许的 2x2 stride=2 池化次数上限(官方配置最多两次)。
MAX_POOL_PER_PART = 2

#: 官方配置的骨干预卷积层数(不含末尾的 1x1 分类卷积):
#: Darknet19 = 18 个卷积 + 1 个全连接分类层;Darknet53 = 52 个卷积 + 1 个全连接分类层。
#: 本实现把分类层也写成 1x1 卷积,因此 ``num_classes > 0`` 时卷积总数分别正好是 19 和 53,
#: 与网络名字里的 "19" / "53" 一一对应。
EXPECTED_CONVS: Dict[str, int] = {'19': 18, '53': 52}


#: 两个官方配置的完整结构描述。
#:
#: 每个变体给出 ``stem`` 与 ``prelude / body / coda`` 四段,外加全局池化与分类头:
#:
#: - ``stem``:首卷积序列的 ``channels / kernels / strides``,``pool`` 表示其后是否接一次池化。
#: - ``prelude / body / coda``:``maxpool`` 是段头 2x2 stride=2 池化次数,``blocks`` 是若干块,
#:   每个块写成 ``(块类型, 通道规格)``。通道规格见 ``_expand_spec``:全整数时按
#:   ``DEFAULT_KERNELS`` 循环套用核边长;全 ``(通道数, 核边长)`` 元组时逐层显式指定。
#:
#: 段的划分不是随意的:它同时决定 ``stage='all'`` 返回的三个尺度,所以按「每个下采样段一个块」
#: 的粒度来切,让 prelude / body / coda 分别落在 8 / 16 / 32 倍下采样上(YOLOv3 的 FPN 尺度)。
DARKNET_SPECS: Dict[str, Dict[str, Any]] = {
    '19': {
        # 官方 cfg/darknet19.cfg:19 个权重层 = 18 个卷积 + 1 个全连接分类层,
        # 下采样全部由 5 次 2x2 maxpool 完成(32 倍)。
        'stem': {
            'channels': (32,),
            'kernels': (3,),
            'strides': (1,),
            'pool': False,
        },
        'prelude': {
            'maxpool': 1,
            'blocks': ((CONV_BLOCK, ((64, 3),)),),
        },
        'body': {
            'maxpool': 2,
            'blocks': (
                (CONV_BLOCK, ((128, 3), (64, 1), (128, 3))),
                (CONV_BLOCK, ((256, 3), (128, 1), (256, 3))),
            ),
        },
        'coda': {
            'maxpool': 2,
            'blocks': (
                (CONV_BLOCK, ((512, 3), (256, 1), (512, 3), (256, 1), (512, 3))),
                (CONV_BLOCK, ((1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3))),
            ),
        },
        # 官方 cfg:``[avgpool] -> conv 1x1 filters=1000 activation=linear -> [softmax]``
        'avgpool': True,
        'num_classes': 1000,
    },
    '53': {
        # 官方 cfg/darknet53.cfg:每个块以 stride=2 卷积下采样,块内是若干瓶颈残差块,
        # 23 个残差块 / 52 个卷积 + 1 个全连接分类层 = 53 个权重层。
        # 段划分按每段一个下采样卷积:prelude 是 8 倍下采样(52x52)、body 是 16 倍(26x26)、
        # coda 是 32 倍(13x13),正好是 YOLOv3 的三个预测尺度。
        'stem': {
            'channels': (32,),
            'kernels': (3,),
            'strides': (1,),
            'pool': False,
        },
        'prelude': {
            'maxpool': 0,
            'blocks': (
                # 下采样 c64  + 1 个残差块  -> 8 倍下采样
                (RESIDUAL_BLOCK, ((64, 3),) + ((64, 1), (64, 3)) * 1),
                # 下采样 c128 + 2 个残差块
                (RESIDUAL_BLOCK, ((128, 3),) + ((128, 1), (128, 3)) * 2),
            ),
        },
        'body': {
            'maxpool': 0,
            'blocks': (
                # 下采样 c256 + 8 个残差块      -> 16 倍下采样
                (RESIDUAL_BLOCK, ((256, 3),) + ((256, 1), (256, 3)) * 8),
                # 下采样 c512 + 8 个残差块
                (RESIDUAL_BLOCK, ((512, 3),) + ((512, 1), (512, 3)) * 8),
            ),
        },
        'coda': {
            'maxpool': 0,
            'blocks': (
                # 下采样 c1024 + 4 个残差块     -> 32 倍下采样(YOLOv3 的第三个预测尺度)
                (RESIDUAL_BLOCK, ((1024, 3),) + ((1024, 1), (1024, 3)) * 4),
            ),
        },
        'avgpool': True,
        'num_classes': 1000,
    },
}


class Darknet(BasicModel):
    '''
    Darknet 主干,按官方 cfg 复原 YOLOv2 的 Darknet-19 与 YOLOv3 的 Darknet-53。

    网络分四段装配,命名沿用 YOLOv1 的 ``Darknet``::

        stem     : 首卷积(检测网络取特征时就截到这里)
        prelude  : 段头若干次 2x2 stride=2 池化 + 若干块
        body     : 段头池化 + 中间若干块
        coda     : 段头池化 + 最后若干块,输出最高层语义特征

    其中每段由「段头池化次数 + 若干块」组成,块又分两种:

    - **conv 块**:若干普通卷积,块首卷积负责下采样与换通道,其余卷积按 3x3 升通道 / 1x1 压回来
      交替(Darknet19 的写法)。
    - **residual 块**:块首一个下采样卷积,其后每两个卷积组成一个 1x1 -> 3x3 的瓶颈残差块
      (Darknet53 的写法)。

    ``variant='19'``(输入 416;下采样全部由 5 次 maxpool 完成)::

        stem     : conv 3x3 s1 c32                                      -> 416 x 416
        prelude  : pool -> conv 3x3 c64                                 -> 208 x 208  (2x)
        body     : pool -> conv c128 -> c64 -> c128
                   pool -> conv c256 -> c128 -> c256                    ->  52 x  52  (8x)
        coda     : pool -> conv c512  -> c256 -> c512  -> c256 -> c512
                   pool -> conv c1024 -> c512 -> c1024 -> c512 -> c1024 ->  13 x  13  (32x)
        avgpool -> conv 1x1 c1000 -> [B, 1000]

    ``variant='53'``(输入 416;下采样由每个块开头的 stride=2 卷积完成)::

        stem     : conv 3x3 s1 c32                                      -> 416 x 416
        prelude  : conv 3x3 s2 c64   -> Residual c64            x1      -> 208 x 208
                   conv 3x3 s2 c128  -> Residual c128           x2      -> 104 x 104  (8x)
        body     : conv 3x3 s2 c256  -> Residual c256           x8      ->  52 x  52
                   conv 3x3 s2 c512  -> Residual c512           x8      ->  26 x  26  (16x)
        coda     : conv 3x3 s2 c1024 -> Residual c1024          x4      ->  13 x  13  (32x)
        avgpool -> conv 1x1 c1000 -> [B, 1000]

    命名与官方 cfg 的对应关系:stem 对应第一个 ``[convolutional]``;prelude/body/coda 对应被
    ``[maxpool]``(Darknet19)或 ``# Downsample``(Darknet53)切开的各段;分类头对应 cfg 末尾的
    ``[avgpool] -> conv 1x1 filters=1000``。下采样只发生在两处 —— 段头的 2x2 stride=2 池化,
    或块首的 stride=2 卷积,同一段内两者互斥,因此总倍数恒为 32。

    三种用法:

    - **ImageNet 分类**(默认 ``num_classes=1000``):末尾接全局平均池化与 1x1 分类卷积,
      与官方 cfg 完全一致;``conv_count`` 分别是 19 与 53,正是两个网络名字的由来。
    - **检测主干**(``num_classes=0``):去掉分类头,``forward`` 直接返回 32 倍下采样的网格特征图
      (416 输入时为 13x13),交给检测头。
    - **多尺度特征**(``forward(x, stage='all')``):返回 ``prelude / body / coda`` 三个尺度的特征图。
      ``variant='53'`` 下是 104x104(8x)/ 26x26(16x)/ 13x13(32x),正是 YOLOv3 的 FPN 尺度;
      ``variant='19'`` 下是 208x208(2x)/ 52x52(8x)/ 13x13(32x)。

    **尺寸与多尺度训练**

    主干是「全卷积 + 全局平均池化 + 1x1 分类卷积」,没有 Flatten 全连接,所以**任意 32 的倍数边长
    都能前向**,``size`` 只决定 ``grid_size`` 这个记录值,不影响任何权重形状(参数量恒定)。因此同一个
    模型实例可以在训练中不断换尺寸 —— 这正是消除"训练 256、推理 416"分布失配的正规做法:

    1. 每个 batch(或每几个 epoch)用 ``sample_train_size`` 抽一档边长,把图 resize/random-crop 到
       该边长再前向。batch norm 的 running stats 由前向自动累积,不需要额外操作。
    2. 目标部署尺寸要占足够比例(用 ``sample_train_size(focus=(416,))`` 保证),否则统计量仍会被
       其它尺寸主导。官方 ``darknet19.cfg`` 的 ``min_crop=128, max_crop=448`` 就是这个思路。
    3. 训练完直接在目标尺寸上评估,不要拿训练尺寸的数字外推。

    想要严格复现单尺寸实验时,``forward`` 传 ``keep_size=True`` 可以恢复"边长必须等于构造时 size"的
    旧校验。注意 batch norm 的统计量与分辨率是绑定的:如果只用一档尺寸训练,换尺寸推理就会掉点,
    此时要么按上面做多尺度训练,要么事后在目标尺寸上重估统计量。

    Attributes:
        variant (str): '19' 或 '53',决定深度与通道规格。
        num_classes (int): 类别数,0 表示已去掉分类头。
        size (int): 构造时声明的参考输入边长(也是 ``verify_shapes`` 与 ``grid_size`` 用的那一档);
            它不限制前向:任何 32 的倍数的边长都能直接吃。
        downsample (int): 全局下采样倍数,固定 32。
        grid_size (int): ``size`` 下的输出特征图边长。
        stem (nn.Sequential): 首卷积序列。
        prelude (nn.Sequential): 第一段。
        body (nn.Sequential): 中间段。
        coda (nn.Sequential): 最后一段。
        avgpool (nn.AdaptiveAvgPool2d, optional): 全局平均池化,``num_classes=0`` 时为 None。
        classifier (nn.Conv2d, optional): 1x1 分类卷积,``num_classes=0`` 时为 None。
        block_channels (Tuple[int, ...]): 逐块输入通道数。
        stage_channels (Tuple[int, int, int]): prelude / body / coda 的输出通道数。
        conv_specs (Tuple[Tuple[int, int], ...]): 逐卷积的 ``(输出通道, 核边长)`` 表(含分类卷积)。
        conv_count (int): 卷积层总数(``num_classes>0`` 时为 19 或 53,否则各减 1)。
        residual_count (int): 残差块总数(Darknet19 为 0,Darknet53 为 23)。
    '''

    #: 官方配置表,键为变体名。
    SPECS = DARKNET_SPECS

    def __setattr__(self, name: str, value: Any) -> None:
        '''
        拦截 ``activation`` 赋值,避免把共享的激活模块注册成子模块。

        传入的激活是 ``BasicModel`` 子类(例如 ``LeakyReLU(0.1)``)时,直接赋值会让
        ``nn.Module`` 把它登记进 ``_modules``,于是 ``state_dict`` 里会多出
        ``activation.negative_slope`` 这类条目,破坏 ``from_remote`` / ``load_state_dict``
        的键匹配。这里把激活拆成两部分:``activation`` 只给出稳定的字符串描述,原始对象放在
        普通属性 ``_activation_value`` 里(不进 ``_modules``),建模时用 ``activation_module``
        取回 —— 这样 ``LeakyReLU(0.1)`` 的斜率能原样生效。

        Args:
            name (str): 属性名。
            value (Any): 属性值。

        Raises:
            ValueError: 当 ``activation`` 收到带可学习参数的模块,或类型不受支持时。
        '''
        if name == 'activation':
            if isinstance(value, BasicModel):
                params = [n for n, _ in value.named_parameters()]
                if params:
                    raise ValueError(
                        f'激活模块 {type(value).__name__} 带可学习参数 {params},'
                        f'不支持按共享实例传入,请改传激活名称字符串。'
                    )
            elif value is not None and not isinstance(value, str):
                raise ValueError(
                    f'activation 需要是字符串、None 或无参数激活模块,收到 {type(value).__name__}。'
                )
            # 两个属性都直接写实例字典:绕开 nn.Module 的子模块登记,state_dict 才干净。
            object.__setattr__(self, '_activation_value', value)
            object.__setattr__(self, '_activation_name', activation_name(value))
            return

        super().__setattr__(name, value)

    @property
    def activation(self) -> Optional[str]:
        '''
        激活配置的字符串描述:'leakyrelu'、None,或传入实例时的类名(例如 'leakyrelu')。
        '''
        return getattr(self, '_activation_name', None)

    @property
    def activation_module(self) -> Union[str, None, nn.Module]:
        '''
        返回实际参与建模的激活对象:传入模块实例时是该实例本身,否则是字符串名称。

        建模时一律用它而不是 ``activation`` 字符串,这样 ``LeakyReLU(0.1)`` 这种自定义斜率
        才能原样生效 —— 字符串 'leakyrelu' 走 ``get_activation`` 会退回 codon 的默认斜率 0.01。
        '''
        return getattr(self, '_activation_value', None)

    def __init__(
        self,
        variant: str = '19',
        in_channels: int = 3,
        num_classes: int = 1000,
        size: int = 416,
        norm: str = 'batch',
        activation: Union[str, BasicModel] = act
    ):
        '''
        初始化 Darknet 主干。

        Args:
            variant (str, optional): '19'(YOLOv2)或 '53'(YOLOv3)。Defaults to '19'.
            in_channels (int, optional): 输入图像通道数。Defaults to 3.
            num_classes (int, optional): 类别数;0 表示去掉全局池化与分类卷积,只输出特征图。
                Defaults to 1000.
            size (int, optional): 输入图像边长,必须是 32 的倍数。416 得到 13x13 网格,
                608 得到 19x19,ImageNet 预训练常用 256。这个值只作为参考档位(决定 ``grid_size``
                与构造期形状自检用的尺寸),前向可以换任意 32 的倍数的边长。Defaults to 416.
            norm (str, optional): 归一化类型,官方配置为 'batch';传 None 可关掉归一化。
                Defaults to 'batch'.
            activation (Union[str, BasicModel], optional): 激活函数,默认斜率 0.1 的 leaky ReLU。
                Defaults to act.

        Raises:
            ValueError: 当 ``variant`` 不在配置表里、``size`` 不是 32 的倍数,
                或规格里的通道/核边长自相矛盾时。
        '''
        super().__init__()

        variant = str(variant)
        if variant not in self.SPECS:
            raise ValueError(
                f'不支持的 Darknet 变体 {variant!r},可选:{sorted(self.SPECS)}。'
            )

        self.variant = variant
        self.in_channels = in_channels
        self.num_classes = int(num_classes)
        self.size = size
        self.norm = norm
        self.activation = activation

        self.downsample = GLOBAL_DOWNSAMPLE
        self.grid_size = self.resolve_grid(size, self.downsample)

        self.conv_specs = self.resolve_conv_specs()
        self._count_convolutions(self.conv_specs)
        self._block_strides = self.resolve_block_strides()
        self._part_inputs = self.resolve_part_inputs()
        self.block_channels = tuple(channel for _, channel, _ in self._block_strides)

        self.stem_channels = self._build_stem()
        self.prelude = self._build_part('prelude')
        self.body = self._build_part('body')
        self.coda = self._build_part('coda')

        spec = self.SPECS[self.variant]
        if self.num_classes:
            self.avgpool = nn.AdaptiveAvgPool2d(1) if spec['avgpool'] else None
            self.classifier = darknet_linear_conv(
                in_channels=self.coda_channels,
                out_channels=self.num_classes,
                kernel_size=1
            )
        else:
            self.avgpool = None
            self.classifier = None

        self.stage_channels = (self.prelude_channels, self.body_channels, self.coda_channels)
        self.residual_count = sum(
            1 for module in self.modules() if isinstance(module, ResidualBlock)
        )

        # 构造期的最后一步:用一块假输入真的跑一遍,让「通道接不上」这类错误在实例化时就暴露,
        # 而不是等到训练/推理时才发现。见 verify_shapes。
        self.verify_shapes()

    # ------------------------------------------------------------------
    # 配置推导
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_grid(size: int, downsample: int = GLOBAL_DOWNSAMPLE) -> int:
        '''
        校验输入边长并推导输出特征图边长。

        Darknet 的下采样由 maxpool 与 stride=2 卷积组成,两者对偶数边长都是精确的 2 倍缩小,
        所以总倍数固定为 ``downsample``(Darknet19/53 都是 32)。

        Args:
            size (int): 输入图像边长。
            downsample (int, optional): 全局下采样倍数。Defaults to 32.

        Returns:
            int: 输出特征图边长,即 ``size // downsample``。

        Raises:
            ValueError: 当 ``size`` 不是 ``downsample`` 的倍数时。
        '''
        if size % downsample != 0:
            raise ValueError(
                f'输入边长 {size} 不是 {downsample} 的整数倍,{size}x{size} 无法整除下采样。'
                f'最接近的合法边长是 {Darknet.nearest_valid_size(size, downsample)}。'
            )
        return size // downsample

    @staticmethod
    def nearest_valid_size(size: int, downsample: int = GLOBAL_DOWNSAMPLE) -> int:
        '''
        求最接近的合法输入边长(``downsample`` 的倍数)。

        Args:
            size (int): 期望的输入边长。
            downsample (int, optional): 全局下采样倍数。Defaults to 32.

        Returns:
            int: 最接近的 ``downsample`` 倍数,最小为 ``downsample``。
        '''
        return max(downsample, int(round(size / downsample)) * downsample)

    def _stem_schema(self) -> Dict[str, Any]:
        '''返回当前变体的 stem 规格,并把 ``channels/kernels/strides`` 逐项校验成整数元组。'''
        schema = self.SPECS[self.variant]['stem']
        channels = tuple(int(c) for c in schema['channels'])
        kernels = tuple(int(k) for k in schema['kernels'])
        strides = tuple(int(s) for s in schema['strides'])
        if not (len(channels) == len(kernels) == len(strides)) or not channels:
            raise ValueError(
                f'stem 的 channels/kernels/strides 必须等长且非空,'
                f'收到 {channels} / {kernels} / {strides}。'
            )
        if any(c <= 0 for c in channels) or any(s not in (1, 2) for s in strides):
            raise ValueError(f'stem 的通道数须为正、步长须为 1 或 2,收到 {channels} / {strides}。')
        if any(k <= 0 or k % 2 == 0 for k in kernels):
            raise ValueError(f'stem 的核边长必须是正奇数,收到 {kernels}。')
        return {'channels': channels, 'kernels': kernels, 'strides': strides,
                'pool': bool(schema['pool'])}

    @staticmethod
    def _expand_spec(spec: Sequence[Any], block_type: str) -> Tuple[Tuple[int, int], ...]:
        '''
        把块的通道规格展开成逐卷积的 ``(out_channels, kernel_size)`` 表。

        两种等价写法(与官方 cfg 的行序一一对应):

        - **全整数**:每个整数是一个卷积的输出通道数,核边长按 ``DEFAULT_KERNELS[block_type]``
          循环套用(residual 块为 3x3 / 1x1 / 3x3 循环,正好是「下采样卷积 + (1x1, 3x3) 对」)。
        - **全 (通道数, 核边长) 元组**:逐层显式指定核边长,用于默认模式表达不了的地方
          (例如 residual 块里压缩比不是 1:1)。

        Args:
            spec (Sequence[Any]): 该块的通道规格。
            block_type (str): ``CONV_BLOCK`` 或 ``RESIDUAL_BLOCK``。

        Returns:
            Tuple[Tuple[int, int], ...]: 逐卷积的 (输出通道, 核边长)。

        Raises:
            ValueError: 当规格为空、两种写法混用,或通道/核边长非法时。
        '''
        if block_type not in DEFAULT_KERNELS:
            raise ValueError(f'不支持的块类型 {block_type!r},可选:{sorted(DEFAULT_KERNELS)}。')
        if not spec:
            raise ValueError(f'块规格不能为空,收到 {spec!r}。')

        if all(isinstance(item, int) for item in spec):
            kernels = DEFAULT_KERNELS[block_type]
            if len(spec) % len(kernels) != 0:
                raise ValueError(
                    f'{block_type} 块的整数写法长度必须是 {len(kernels)} 的整数倍,'
                    f'收到 {len(spec)} 个通道 {tuple(spec)};'
                    f'需要逐层指定核边长时请写成 (通道数, 核边长) 元组。'
                )
            expanded = tuple(
                (int(channel), kernels[index % len(kernels)])
                for index, channel in enumerate(spec)
            )
        elif all(isinstance(item, (tuple, list)) and len(item) == 2 for item in spec):
            expanded = tuple((int(channel), int(kernel)) for channel, kernel in spec)
        else:
            raise ValueError(
                f'{block_type} 块的规格要么全是整数(按默认核边长),要么全是 (通道数, 核边长) 元组,'
                f'不能混用,收到 {spec!r}。'
            )

        for channel, kernel in expanded:
            if channel <= 0 or kernel <= 0 or kernel % 2 == 0:
                raise ValueError(
                    f'通道数必须是正整数、核边长必须是正奇数,收到 (通道={channel}, 核={kernel})。'
                )
        return expanded

    def _part_schema(self, part: str) -> Dict[str, Any]:
        '''返回某一段的规格,并校验 ``maxpool`` 与块的写法。'''
        if part not in ('prelude', 'body', 'coda'):
            raise ValueError(f'不支持的段名 {part!r}。')
        schema = self.SPECS[self.variant][part]
        pools = int(schema['maxpool'])
        if not 0 <= pools <= MAX_POOL_PER_PART:
            raise ValueError(
                f'{self.variant} 的 {part} 段声明了 {pools} 次池化,'
                f'必须在 0..{MAX_POOL_PER_PART} 之间。'
            )
        if not schema['blocks']:
            raise ValueError(f'{self.variant} 的 {part} 段至少需要一个块。')
        return {'maxpool': pools, 'blocks': tuple(schema['blocks'])}

    def _walk_block(
        self,
        expanded: Sequence[Tuple[int, int]],
        block_type: str,
        block_name: str
    ) -> int:
        '''
        沿一个块的卷积序列推导通道流,并校验配置内部自洽。

        规则与官方 cfg 的写法一一对应:

        - 把序列按核边长切成若干「连续同核」的段。同一段里的层依次吃同一张特征图,因此通道数
          必须完全一致 —— 这既解释了 3x3 升通道之后为什么必须紧跟 1x1 压回来,也解释了
          Darknet53 残差块里 ``filters=C`` 连续重复的写法。
        - 跨段的通道变化是合法的:1x1 压缩(Darknet19 的 128 -> 64)、3x3 投影
          (Darknet53 的 32 -> 64)都靠换核来换通道。
        - 块的入口通道由上游决定,所以**首层允许任意改通道**(它就是本块的下采样/投影卷积);
          其余各层的输入必须由块内自己对齐,不一致时在这里直接报错。

        Args:
            expanded (Sequence[Tuple[int, int]]): ``_expand_spec`` 展开的逐卷积 (通道, 核)。
            block_type (str): ``CONV_BLOCK`` 或 ``RESIDUAL_BLOCK``。
            block_name (str): 报错信息里使用的块名字。

        Returns:
            int: 块的输出通道数。

        Raises:
            ValueError: 同核段内通道不一致,或残差规格不合法时。
        '''
        if block_type == RESIDUAL_BLOCK:
            if len(expanded) < 3 or len(expanded) % 2 == 0:
                raise ValueError(
                    f'{block_name} 的残差规格必须是 1 + 2n 个卷积'
                    f'(下采样卷积 + n 个 (1x1, 3x3) 对),收到 {expanded}。'
                )
            for index in range(1, len(expanded) - 1, 2):
                inner, outer = expanded[index][0], expanded[index + 1][0]
                if inner != outer:
                    raise ValueError(
                        f'{block_name} 第 {(index + 1) // 2} 个残差块的 1x1 与 3x3 输出通道必须相同'
                        f'(shortcut 按元素相加),收到 {inner} 与 {outer}。'
                    )

        start = 0
        while start < len(expanded):
            kernel = expanded[start][1]
            end = start
            while end + 1 < len(expanded) and expanded[end + 1][1] == kernel:
                end += 1
            group = expanded[start:end + 1]

            if len(group) > 1 and start:
                # 连续同核的若干层叠加在同一张特征图上,通道数必须一致;段首那组由上游决定输入,
                # 所以只要求组内一致,不与上游比较。
                first_channel = group[0][0]
                for offset, (channel, _) in enumerate(group[1:], start=1):
                    if channel != first_channel:
                        raise ValueError(
                            f'{block_name} 第 {start + offset + 1} 层的通道衔接断裂:'
                            f'连续的 {kernel}x{kernel} 卷积共享同一张特征图,'
                            f'但前面是 {first_channel} 通道、这里写的是 {channel} 通道。'
                        )
            start = end + 1

        return expanded[-1][0]

    def resolve_conv_specs(self) -> Tuple[Tuple[int, int], ...]:
        '''
        按官方 cfg 的顺序展开全部卷积:stem -> prelude -> body -> coda -> 分类卷积。

        同时做两件事:

        1. 逐块校验通道衔接(见 ``_walk_block``),让配置错误在构造期暴露,而不是等到前向时报
           ``mat1 and mat2 shapes cannot be multiplied`` 这类难以定位的错误。
        2. 收集逐卷积的 ``(输出通道, 核边长)`` 表,供下游(YOLO 检测头、形状检查)直接读取。

        Returns:
            Tuple[Tuple[int, int], ...]: 逐卷积的 ``(输出通道, 核边长)`` 表(含分类卷积)。

        Raises:
            ValueError: 当块内通道衔接断裂,或残差规格不合法时。
        '''
        stem = self._stem_schema()
        specs: List[Tuple[int, int]] = list(zip(stem['channels'], stem['kernels']))

        for part in ('prelude', 'body', 'coda'):
            schema = self._part_schema(part)
            for index, (block_type, spec) in enumerate(schema['blocks']):
                expanded = self._expand_spec(spec, block_type)
                self._walk_block(expanded, block_type, f'{part} 段第 {index + 1} 个块')
                specs.extend(expanded)

        if self.num_classes:
            specs.append((self.num_classes, 1))
        return tuple(specs)

    def resolve_block_strides(self) -> Tuple[Tuple[str, int, int], ...]:
        '''
        推导逐块的输入通道与块首卷积的步长。

        步长规则与官方 cfg 的 ``[maxpool]`` / ``# Downsample`` 完全对应:

        - 段内有池化时:池化完成本段的下采样,所以**第一个块**的块首卷积步长为 1,
          其余块各用 stride=2 卷积再下采样一次。
        - 段内没有池化时:每个块的块首卷积都是 stride=2。

        Returns:
            Tuple[Tuple[str, int, int], ...]: 逐块的 ``(块名, 输入通道, 块首卷积步长)``。

        Raises:
            ValueError: 当推导出的下采样次数与 ``log2(downsample)`` 不符时。
        '''
        stem = self._stem_schema()
        blocks: List[Tuple[str, int, int]] = []
        current = stem['channels'][-1]
        downsample_points = sum(1 for stride in stem['strides'] if stride == 2) + int(stem['pool'])

        for part in ('prelude', 'body', 'coda'):
            schema = self._part_schema(part)
            pools = schema['maxpool']
            downsample_points += pools
            for index, (block_type, spec) in enumerate(schema['blocks']):
                # 段头池化已完成本段的下采样,所以只有它后面的块(含段内第一个块之后的块)
                # 才用 stride=2 卷积继续下采样。
                stride = 2 if index >= pools else 1
                blocks.append((f'{part}[{index}]', current, stride))
                downsample_points += 1 if stride == 2 else 0
                current = self._expand_spec(spec, block_type)[-1][0]

        expect = int(math.log2(self.downsample))
        if downsample_points != expect:
            raise ValueError(
                f'Darknet-{self.variant} 共有 {downsample_points} 处 2 倍下采样,'
                f'合计 {2 ** downsample_points} 倍,与 downsample={self.downsample} 不符。'
            )
        return tuple(blocks)

    def resolve_part_inputs(self) -> Dict[str, int]:
        '''
        推导 ``prelude / body / coda`` 三段的输入通道。

        段间衔接是显式的:prelude 接 stem 的输出,body 接 prelude 最后一个块的输出,
        coda 接 body 最后一个块的输出;相邻两段之间没有卷积,只有池化,所以通道数不变。

        Returns:
            Dict[str, int]: ``{'prelude': c0, 'body': c1, 'coda': c2}``。
        '''
        stem = self._stem_schema()
        current = stem['channels'][-1]
        inputs: Dict[str, int] = {}
        for part in ('prelude', 'body', 'coda'):
            inputs[part] = current
            schema = self._part_schema(part)
            for block_type, spec in schema['blocks']:
                current = self._expand_spec(spec, block_type)[-1][0]
        return inputs

    def _count_convolutions(self, conv_specs: Sequence[Tuple[int, int]]) -> None:
        '''
        校验卷积层总数与官方配置一致,并写入 ``conv_count``。

        Darknet 的名字直接来自权重层数:Darknet19 是 18 个卷积 + 1 个全连接分类层,
        Darknet53 是 52 个卷积 + 1 个全连接分类层;本实现把分类层也写成 1x1 卷积,
        所以 ``conv_count`` 分别等于 19 和 53。

        Args:
            conv_specs (Sequence[Tuple[int, int]]): ``resolve_conv_specs`` 的结果。

        Raises:
            ValueError: 当展开结果与 ``EXPECTED_CONVS`` 不符时。
        '''
        self.conv_count = len(conv_specs)
        expected = EXPECTED_CONVS.get(self.variant)
        if expected is not None:
            expected += 1 if self.num_classes else 0
            if self.conv_count != expected:
                raise ValueError(
                    f'Darknet-{self.variant} 的配置展开出 {self.conv_count} 个卷积层,'
                    f'与官方配置的 {expected} 个不符(numb_classes={self.num_classes}),'
                    f'请检查 SPECS[{self.variant!r}]。'
                )

    # ------------------------------------------------------------------
    # 各段构建
    # ------------------------------------------------------------------

    def _build_stem(self) -> int:
        '''
        构建 ``stem``:首卷积序列。

        两个变体的 stem 都是单个 stride=1 的 3x3 卷积(下采样交给 prelude 的池化或块首的
        stride=2 卷积);规格表里保留完整的 ``channels / kernels / strides`` 写法,便于扩展变体。

        Returns:
            int: stem 的输出通道数。
        '''
        schema = self._stem_schema()
        layers: List[nn.Module] = []
        current = self.in_channels
        for channel, kernel, stride in zip(schema['channels'], schema['kernels'], schema['strides']):
            layers.append(self._build_conv(current, channel, kernel, stride))
            current = channel
        if schema['pool']:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.stem = nn.Sequential(*layers)
        return current

    def _build_conv(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1
    ) -> ConvBlock:
        '''
        构建一个带归一化与激活的 Darknet 卷积(``pad = kernel // 2``)。

        激活用 ``activation_module`` 而不是 ``activation`` 字符串:传入 ``LeakyReLU(0.1)`` 时
        能原样复用该实例,避免 ``get_activation('leakyrelu')`` 退回 codon 的默认斜率 0.01。
        '''
        return darknet_conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm=self.norm,
            activation=self.activation_module
        )

    def _build_block(
        self,
        in_channels: int,
        block_type: str,
        spec: Sequence[Any],
        stride: int = 2
    ) -> nn.Sequential:
        '''
        构建一个块:块首卷积负责下采样,其后是若干同通道的卷积或瓶颈残差块。

        Args:
            in_channels (int): 块输入通道数。
            block_type (str): ``CONV_BLOCK`` 或 ``RESIDUAL_BLOCK``。
            spec (Sequence[Any]): 通道规格,语义见 ``_expand_spec``。
            stride (int, optional): 块首卷积的步长。Defaults to 2.

        Returns:
            nn.Sequential: 该块的层序列。
        '''
        expanded = self._expand_spec(spec, block_type)
        layers: List[nn.Module] = []
        current = in_channels
        index = 0
        while index < len(expanded):
            if block_type == RESIDUAL_BLOCK and index:
                # residual 块:从第 2 个元素起,每两个元素 (1x1 压缩, 3x3 扩展) 组成一个残差块。
                # 残差块自带这两层,所以这里必须整体跳过两个元素,不能只跳一个。
                inner = expanded[index][0]
                outer = expanded[index + 1][0]
                layers.append(ResidualBlock(
                    in_channels=outer,
                    inner_channels=inner,
                    out_channels=outer,
                    norm=self.norm,
                    activation=self.activation_module
                ))
                current = outer
                index += 2
                continue

            channel, kernel = expanded[index]
            layers.append(self._build_conv(
                current,
                channel,
                kernel,
                stride if index == 0 else 1
            ))
            current = channel
            index += 1
        return nn.Sequential(*layers)

    def _build_part(self, part: str) -> nn.Sequential:
        '''
        构建 ``prelude / body / coda`` 中的一段:段头池化 + 若干块。

        块的输入通道与块首步长都取自 ``resolve_block_strides``,因此这一段既是「按官方 cfg 装配」,
        也是「配置自洽性的运行期验证」:任何通道接不上的组合在构造阶段就会被排除。

        Args:
            part (str): 'prelude'、'body' 或 'coda'。

        Returns:
            nn.Sequential: 该段的层序列(池化 + 块)。

        Raises:
            ValueError: 当段名不受支持时。
        '''
        schema = self._part_schema(part)
        strides = {name: stride for name, _, stride in self._block_strides}

        layers: List[nn.Module] = [
            nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(schema['maxpool'])
        ]
        current = self._part_inputs[part]
        for index, (block_type, spec) in enumerate(schema['blocks']):
            block = self._build_block(current, block_type, spec, strides[f'{part}[{index}]'])
            layers.append(block)
            current = self._expand_spec(spec, block_type)[-1][0]

        setattr(self, f'{part}_channels', current)
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------

    def verify_shapes(self) -> Tuple[int, int, int]:
        '''
        用一块假输入跑一遍前向,校验通道与空间尺寸真的能接上。

        静态检查(spec 表的自洽性)只能覆盖「同核段内通道一致」这类结构性约束,真正决定特征图能否
        接上的是每个卷积的 in/out 通道,而 in_channels 依赖上游 —— 所以这里直接做一次真实前向:
        任何通道对不上都会在实例化 ``Darknet`` 时抛 ``RuntimeError``,而不是等到训练时才炸。

        Returns:
            Tuple[int, int, int]: prelude / body / coda 输出特征图的通道数。

        Raises:
            RuntimeError: 前向过程中出现形状不匹配(通道数或空间尺寸接不上)。
        '''
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                probe = torch.zeros(1, self.in_channels, self.size, self.size)
                channels = tuple(feature.shape[1] for feature in self.stage_features(probe))
        except RuntimeError as exc:  # pragma: no cover - 只在配置写错时触发
            raise RuntimeError(
                f'Darknet-{self.variant} 的通道/尺寸衔接校验失败(size={self.size}):'
                f'{exc}。请检查 SPECS[{self.variant!r}] 里的通道规格。'
            ) from exc
        finally:
            self.train(was_training)
        return channels

    def stage_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        '''
        返回三个尺度的特征图,对应 ``prelude / body / coda`` 三段的输出。

        416 输入时,``variant='53'`` 得到 104x104(8 倍下采样)、26x26(16 倍)、13x13(32 倍),
        正是 YOLOv3 的 FPN 尺度;``variant='19'`` 得到 208x208(2 倍)、52x52(8 倍)、13x13(32 倍)。

        Args:
            x (torch.Tensor): 输入图像。形状 [Batch, in_channels, size, size]

        Returns:
            List[torch.Tensor]: 长度 3 的特征图列表,空间尺寸依次减半。
        '''
        x = self.stem(x)
        features: List[torch.Tensor] = []
        for part in (self.prelude, self.body, self.coda):
            x = part(x)
            features.append(x)
        return features

    def forward(
        self,
        x: torch.Tensor,
        stage: Literal['coda', 'all'] = 'coda',
        keep_size: bool = False
    ) -> torch.Tensor:
        '''
        主干是「全卷积 + 全局平均池化 + 1x1 分类卷积」的结构,所以**任意 32 的倍数边长都能直接
        前向**,不必等于构造时的 ``size``:``size`` 只用来推导并记录 ``grid_size``,卷积核的形状
        与它无关。多尺度训练就是这样用的 —— 同一个模型实例在每个 batch 换一档边长,让 batch norm
        的统计量覆盖各档分辨率,从而消除"训练 256、推理 416"的分布失配。

        需要留意的两点:

        - 换尺寸会**改变 batch norm 的统计量**(训练模式下 running stats 会被新分辨率的分布拉走),
          这正是多尺度训练想要的;但只想要某一档的话就别混着喂,或者事后按该档重估统计量。
        - ``keep_size=True`` 时恢复旧行为(边长必须等于构造时的 ``size``),用于需要严格复现单尺寸
          实验的场景。

        Args:
            x (torch.Tensor): 输入图像。形状 [Batch, in_channels, side, side],``side`` 必须是 32 的
                倍数(224 / 256 / 320 / 416 / 448 等);``keep_size=True`` 时还必须等于构造时的 ``size``。
            stage (Literal['coda', 'all'], optional): 取哪一层输出。'coda' 返回最高层特征图
                (有分类头时是类别 logits);'all' 返回 ``stage_features`` 的三尺度特征图列表。
                Defaults to 'coda'.
            keep_size (bool, optional): 是否强制输入边长与构造时的 ``size`` 一致。Defaults to False.

        Returns:
            torch.Tensor: ``stage='coda'`` 时,若 ``num_classes > 0`` 返回类别 logits
                ``[Batch, num_classes]``,否则返回网格特征图 ``[Batch, C, side // 32, side // 32]``;
                ``stage='all'`` 时返回长度 3 的特征图列表。

        Raises:
            ValueError: 当输入不是 4 维/不是正方形/不是 32 的倍数(或 ``keep_size=True`` 时边长与
                ``size`` 不符),或 ``stage`` 取值非法时。
        '''
        if x.ndim != 4:
            raise ValueError(f'Darknet 主干期望 4 维输入 [B, C, H, W],收到 {tuple(x.shape)}。')

        height, width = x.shape[-2:]
        if height != width:
            raise ValueError(
                f'Darknet 主干是方形下采样结构,期望正方形输入,收到 {height}x{width}。'
            )
        if height % self.downsample != 0:
            raise ValueError(
                f'输入边长 {height} 不是 {self.downsample} 的整数倍(主干有 '
                f'{int(math.log2(self.downsample))} 次 2 倍下采样),'
                f'最接近的合法边长是 {self.nearest_valid_size(height, self.downsample)}。'
            )
        if keep_size and height != self.size:
            raise ValueError(
                f'keep_size=True 时输入边长必须等于构造时的 size={self.size},收到 {height};'
                f'该主干的输出网格固定为 {self.grid_size}x{self.grid_size}。'
            )

        if stage == 'all':
            return self.stage_features(x)
        if stage != 'coda':
            raise ValueError(f"stage 只支持 'coda' 或 'all',收到 {stage!r}。")

        x = self.coda(self.body(self.prelude(self.stem(x))))
        if self.classifier is None:
            return x
        if self.avgpool is not None:
            x = self.avgpool(x)
        return self.classifier(x).flatten(1)

    @staticmethod
    def valid_sizes(
        min_size: int = 256,
        max_size: int = 448,
        step: int = GLOBAL_DOWNSAMPLE
    ) -> Tuple[int, ...]:
        '''
        列出 ``[min_size, max_size]`` 区间内所有合法的输入边长(``step`` 的倍数)。

        多尺度训练时用它一次性拿到候选档位,例如 ``(256, 320, 352, 416, 448)``。

        Args:
            min_size (int, optional): 区间下界(含),会被向上取到 ``step`` 的倍数。
                Defaults to 256.
            max_size (int, optional): 区间上界(含),会被向下取到 ``step`` 的倍数。
                Defaults to 448.
            step (int, optional): 步长,必须等于主干的下采样倍数。Defaults to 32.

        Returns:
            Tuple[int, ...]: 合法边长元组,升序;区间内没有合法值时抛异常。

        Raises:
            ValueError: 当 ``min_size``/``max_size`` 非法,或区间内没有 ``step`` 的倍数时。
        '''
        if step <= 0 or min_size <= 0 or max_size < min_size:
            raise ValueError(
                f'尺寸区间非法:min_size={min_size}, max_size={max_size}, step={step}。'
            )
        low = math.ceil(min_size / step) * step
        high = math.floor(max_size / step) * step
        if low > high:
            raise ValueError(
                f'[{min_size}, {max_size}] 内没有 {step} 的倍数,请放宽区间。'
            )
        return tuple(range(low, high + 1, step))

    @classmethod
    def sample_train_size(
        cls,
        min_size: int = 256,
        max_size: int = 448,
        focus: Optional[Sequence[int]] = None,
        focus_ratio: float = 0.5,
        step: int = GLOBAL_DOWNSAMPLE,
        generator: Optional[torch.Generator] = None
    ) -> int:
        '''
        为多尺度训练随机抽一个输入边长。

        默认在 ``[min_size, max_size]`` 的合法档位里**均匀**取值。给 ``focus`` 时可以偏向某些尺寸
        (例如部署在 416,就传 ``focus=(416,)``) —— 此时有 ``focus_ratio`` 的概率从 ``focus`` 里取,
        其余概率在区间内均匀取。这样既不丢掉多尺度带来的 batch norm 覆盖,又能保证目标尺寸的统计量
        占足够权重(只偶尔喂一次 416 是没用的,统计量仍由 256 主导)。

        用法是在每个 batch(或每个 epoch)取一次,把图 resize/random-crop 到该边长再前向;
        统计量随之更新,因为是模型自己前向时累积的,不需要额外操作。

        Args:
            min_size (int, optional): 区间下界(含)。Defaults to 256.
            max_size (int, optional): 区间上界(含)。Defaults to 448.
            focus (Optional[Sequence[int]], optional): 需要重点覆盖的尺寸列表,会先对齐到 ``step``
                的倍数,非法项被忽略。Defaults to None.
            focus_ratio (float, optional): 从 ``focus`` 里取值的概率,取值 0..1。Defaults to 0.5.
            step (int, optional): 合法边长的步长。Defaults to 32.
            generator (Optional[torch.Generator], optional): 随机数发生器,便于复现。Defaults to None.

        Returns:
            int: 本次训练使用的输入边长。

        Raises:
            ValueError: 当区间非法、``focus_ratio`` 越界,或 ``focus`` 里没有任何合法尺寸时。
        '''
        candidates = cls.valid_sizes(min_size, max_size, step)
        if not 0.0 <= focus_ratio <= 1.0:
            raise ValueError(f'focus_ratio 必须落在 [0, 1],收到 {focus_ratio}。')

        focused: Tuple[int, ...] = ()
        if focus:
            focused = tuple(sorted({
                int(size) for size in focus
                if int(size) % step == 0 and min_size <= int(size) <= max_size
            }))
            if not focused:
                raise ValueError(
                    f'focus={tuple(focus)} 里没有落在 [{min_size}, {max_size}] 且为 {step} 的倍数的尺寸。'
                )

        if focused and focus_ratio > 0:
            # 只用一次均匀抽样同时决定「走 focus 还是走区间」和「取哪个元素」,
            # 这样比例精确等于 focus_ratio,也不依赖随机流的初始状态。
            draw = torch.rand((), generator=generator).item()
            if draw < focus_ratio:
                index = int(draw / focus_ratio * len(focused))
                return focused[min(index, len(focused) - 1)]
            index = int((draw - focus_ratio) / (1.0 - focus_ratio) * len(candidates))
            return candidates[min(index, len(candidates) - 1)]

        index = int(torch.rand((), generator=generator).item() * len(candidates))
        return candidates[min(index, len(candidates) - 1)]

    @staticmethod
    def preprocess(
        x: torch.Tensor,
        downsample: int = GLOBAL_DOWNSAMPLE
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        '''
        把输入补零到 ``downsample`` 的整数倍(只在右下方向补)。

        数据尺寸不规则时(例如任意分辨率的图片),先把边长补到 32 的倍数再送进主干,可以避免
        尺寸校验报错。补零不改变 batch norm 的统计量,只是把多出来的区域变成常数。

        Args:
            x (torch.Tensor): 输入张量。形状 [B, C, H, W]
            downsample (int, optional): 目标下采样倍数。Defaults to 32.

        Returns:
            Tuple[torch.Tensor, Tuple[int, int, int, int]]: 补齐后的张量与
                ``(left, right, top, bottom)`` 填充量,便于把预测坐标映射回原图。
        '''
        height, width = x.shape[-2:]
        target_h = math.ceil(height / downsample) * downsample
        target_w = math.ceil(width / downsample) * downsample
        pad = (0, target_w - width, 0, target_h - height)

        if pad == (0, 0, 0, 0):
            return x, pad
        return F.pad(x, pad), pad

    @staticmethod
    def auto_build(
        input_shape: Tuple[int, ...],
        output_shape: Optional[Tuple[int, ...]] = None,
        variant: str = '19',
        norm: str = 'batch',
        activation: Union[str, BasicModel] = act
    ) -> 'Darknet':
        '''
        按输入/输出形状自动构建。

        Args:
            input_shape (Tuple[int, ...]): 输入形状(不含 batch),例如 (3, 416, 416)。
            output_shape (Optional[Tuple[int, ...]], optional): 输出形状。给定时取最后一维作为
                类别数并保留分类头;为 None 时去掉分类头(检测主干)。Defaults to None.
            variant (str, optional): '19' 或 '53'。Defaults to '19'.
            norm (str, optional): 归一化类型。Defaults to 'batch'.
            activation (Union[str, BasicModel], optional): 激活函数。Defaults to act。

        Returns:
            Darknet: 构建好的主干。
        '''
        return Darknet(
            variant=variant,
            in_channels=input_shape[0],
            num_classes=0 if output_shape is None else int(output_shape[-1]),
            size=input_shape[-1],
            norm=norm,
            activation=activation
        )


if __name__ == '__main__':
    '''自检:打印两个变体的输出形状、三尺度特征、卷积层数与参数量。'''
    for variant, expect in (('19', 19), ('53', 53)):
        model = Darknet(variant=variant, size=416)
        with torch.no_grad():
            logits = model(torch.randn(1, 3, 416, 416))
            features = model(torch.randn(1, 3, 416, 416), stage='all')
        print(f'Darknet-{variant} 416 -> {tuple(logits.shape)}')
        print('  三尺度特征:', [tuple(f.shape) for f in features])
        print('  卷积层数  :', model.conv_count, '| expect', expect,
              '| 残差块', model.residual_count)
        print('  通道表    :', model.block_channels, '/', model.stage_channels)
        print('  参数量    :', model.count_params(human_readable=True))

    backbone = Darknet(variant='53', num_classes=0, size=416)
    with torch.no_grad():
        feat = backbone(torch.randn(1, 3, 416, 416))
    print('检测主干 Darknet-53 ->', tuple(feat.shape),
          '| params', backbone.count_params(human_readable=True))

    for size in (224, 256, 416, 608):
        model = Darknet(variant='19', size=size)
        with torch.no_grad():
            out = model(torch.randn(1, 3, size, size))
        print(f'Darknet-19 {size} -> {tuple(out.shape)}')
