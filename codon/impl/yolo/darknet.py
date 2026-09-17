from codon import *
from codon.block.conv import ConvBlock
from codon.block.activation import LeakyReLU


#: YOLOv1 paper equation (2) and the official Darknet configurations use a leaky ReLU with
#: slope 0.1 for every non-output layer. Note that codon.block.activation.LeakyReLU defaults to
#: slope 0.01, so 0.1 must be passed explicitly.
#: The activation has no parameters and no state, so the whole network shares one instance
#: (``Darknet`` reuses it as is instead of rebuilding it from a name).
act = LeakyReLU(0.1)

#: The Darknet backbone downsamples by 2 ** 5 = 32, so the input side must be a multiple of 32.
GLOBAL_DOWNSAMPLE = 32


# ----------------------------------------------------------------------
# Shared building blocks
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
    Builds a Darknet convolution unit, i.e. one ``[convolutional]`` section of a cfg file.

    The Darknet convention is ``pad = size // 2`` (same padding), so a 3x3 convolution uses
    padding=1 and a 1x1 convolution uses padding=0, while whether the layer downsamples is
    decided by stride. With normalization the convolution needs no bias (the beta of the batch
    norm already acts as the offset), which matches the ``batch_normalize`` behaviour of the
    original darknet.

    Note that ``codon.block.conv.ConvBlock`` defaults to ``norm='batch'`` and
    ``activation='relu'``, which differs from Darknet, so every argument is passed explicitly
    here.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size of the convolution.
        stride (int, optional): Stride; 2 halves the spatial size. Defaults to 1.
        norm (str, optional): Normalization type; the original Darknet uses 'batch'.
            Defaults to 'batch'.
        activation (Union[str, BasicModel], optional): Activation function. Defaults to act
            (slope 0.1).
        bias (bool, optional): Whether to use a convolution bias. None decides automatically
            from ``norm is None``. Defaults to None.

    Returns:
        ConvBlock: A unit combining convolution, normalization and activation.
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
    Builds a purely linear convolution (no normalization, no activation), used before the
    addition of a residual block and in the classification head.

    ``codon.block.activation.get_activation`` does not accept None and ``ConvBlock`` cannot
    express "no activation" in that case, so an ``nn.Conv2d`` is returned directly: the shortcut
    addition of a residual block and the logits of the classification head both need a linear
    output.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Kernel size of the convolution. Defaults to 1.
        stride (int, optional): Stride. Defaults to 1.

    Returns:
        nn.Module: An ``nn.Conv2d`` instance (with bias, since no normalization follows).
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
    Normalizes an activation configuration into a string description.

    The activation object itself is what gets used when building the model (see
    ``Darknet.activation_module``); this string is only for display and logging, so the same
    configuration must always yield the same name.

    Args:
        activation (Union[str, None, BasicModel]): The activation configuration.

    Returns:
        Optional[str]: Strings and ``None`` are returned unchanged; a module yields the lowercase
            name of its class (for example 'leakyrelu').
    '''
    if activation is None or isinstance(activation, str):
        return activation
    return type(activation).__name__.lower()


# ----------------------------------------------------------------------
# Darknet backbone (YOLOv2 / YOLOv3)
# ----------------------------------------------------------------------


class ResidualBlock(BasicModel):
    '''
    The bottleneck residual block of Darknet53: 1x1 squeeze -> 3x3 expand -> shortcut.

    It corresponds to the consecutive ``[convolutional] filters=C size=1``,
    ``[convolutional] filters=C size=3`` and ``[shortcut] activation=linear`` entries of the
    official cfg:

        y = x + conv3x3( activation( conv1x1( activation(x) ) ) )

    Differences from ``codon.block.conv.ResBasicBlock``:

    - The shortcut is ``activation=linear``: no activation is applied after the addition, the
      activation only follows each convolution; the 'original' variant of ResBasicBlock instead
      activates once after the addition. The expanding convolution ``conv2`` is therefore purely
      linear.
    - The bottleneck ratio of Darknet53 is 1:1 (the 1x1 and 3x3 layers have the same number of
      output channels; only the CIFAR version of Darknet uses a C/2 compression ratio), so the
      channel counts of the 1x1 and 3x3 convolutions are given separately.
    - No downsample branch is needed: downsampling is done by the stride=2 convolution at the
      start of each block, and the residual block itself preserves the spatial size.

    Attributes:
        conv1 (ConvBlock): 1x1 squeeze convolution (with activation).
        conv2 (nn.Conv2d): 3x3 expand convolution (linear output, no activation before the
            addition).
        in_channels (int): Number of input channels of the block.
        inner_channels (int): Number of bottleneck channels after the 1x1 squeeze.
        out_channels (int): Number of output channels of the block (after the shortcut addition).
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
        Initializes the bottleneck residual block.

        Args:
            in_channels (int): Number of input channels of the block.
            inner_channels (Optional[int], optional): Number of bottleneck channels after the 1x1
                squeeze. Defaults to ``out_channels // 2`` when omitted. Defaults to None.
            out_channels (Optional[int], optional): Number of output channels of the 3x3 expand,
                i.e. of the residual addition. Defaults to ``in_channels`` (the 1:1 bottleneck of
                Darknet53) when omitted. Defaults to None.
            norm (str, optional): Normalization type. Defaults to 'batch'.
            activation (Union[str, BasicModel], optional): Activation function. Defaults to act.

        Raises:
            ValueError: If any channel count is not a positive integer, or if
                ``out_channels != in_channels`` (the shortcut cannot be added).
        '''
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels
        inner_channels = out_channels // 2 if inner_channels is None else inner_channels
        if inner_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f'bottleneck channel counts must be positive integers, got '
                f'inner={inner_channels}, out={out_channels}'
            )
        if out_channels != in_channels:
            raise ValueError(
                f'the shortcut requires identical input and output channels, got '
                f'in={in_channels}, out={out_channels}; change the channel count with a '
                f'convolution at the start of the block instead'
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
            x (torch.Tensor): Input feature map of shape [Batch, in_channels, H, W].

        Returns:
            torch.Tensor: Output feature map of shape [Batch, out_channels, H, W] (the spatial
                size is unchanged).
        '''
        return x + self.conv2(self.conv1(x))


# ----------------------------------------------------------------------
# Darknet backbone (YOLOv2 / YOLOv3)
# ----------------------------------------------------------------------

#: conv block: every element inside the block is a plain convolution.
CONV_BLOCK = 'conv'
#: residual block: the block starts with a downsampling convolution and every two elements
#: after it form one bottleneck residual block.
RESIDUAL_BLOCK = 'residual'

#: Default kernel-size sequence used by each block type in the "all integers" notation
#: (cycled over the elements).
DEFAULT_KERNELS: Dict[str, Tuple[int, ...]] = {
    CONV_BLOCK: (3,),
    RESIDUAL_BLOCK: (3, 1, 3),
}

#: Upper bound on the number of 2x2 stride=2 poolings inside one stage (the official
#: configurations use at most two).
MAX_POOL_PER_PART = 2

#: Number of backbone convolutions in the official configurations (excluding the trailing 1x1
#: classification convolution):
#: Darknet19 = 18 convolutions + 1 fully connected classification layer; Darknet53 = 52
#: convolutions + 1 fully connected classification layer.
#: This implementation writes the classification layer as a 1x1 convolution too, so with
#: ``num_classes > 0`` the total number of convolutions is exactly 19 and 53, matching the
#: "19" / "53" in the network names one to one.
EXPECTED_CONVS: Dict[str, int] = {'19': 18, '53': 52}


#: Complete structural description of the two official configurations.
#:
#: Every variant provides the four parts ``stem`` and ``prelude / body / coda``, plus global
#: pooling and the classification head:
#:
#: - ``stem``: the ``channels / kernels / strides`` of the leading convolution sequence, where
#:   ``pool`` says whether a pooling follows it.
#: - ``prelude / body / coda``: ``maxpool`` is the number of leading 2x2 stride=2 poolings and
#:   ``blocks`` are the blocks, each written as ``(block type, channel spec)``. See
#:   ``_expand_spec`` for the channel spec: with all integers the kernel sizes of
#:   ``DEFAULT_KERNELS`` are cycled; with all ``(channels, kernel size)`` tuples every layer is
#:   given explicitly.
#:
#: The split into stages is not arbitrary: it also determines the three scales returned by
#: ``stage='all'``, so it is cut at the granularity of "one block per downsampling stage" so
#: that prelude / body / coda land on 8 / 16 / 32 times downsampling (the FPN scales of YOLOv3).
DARKNET_SPECS: Dict[str, Dict[str, Any]] = {
    '19': {
        # Official cfg/darknet19.cfg: 19 weight layers = 18 convolutions + 1 fully connected
        # classification layer, and all downsampling is done by 5 2x2 maxpools (32x).
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
        # Official cfg: ``[avgpool] -> conv 1x1 filters=1000 activation=linear -> [softmax]``
        'avgpool': True,
        'num_classes': 1000,
    },
    '53': {
        # Official cfg/darknet53.cfg: every block downsamples with a stride=2 convolution and
        # contains several bottleneck residual blocks: 23 residual blocks / 52 convolutions + 1
        # fully connected classification layer = 53 weight layers.
        # The stages are split as one downsampling convolution each: prelude is 8x downsampling
        # (52x52), body is 16x (26x26) and coda is 32x (13x13), exactly the three prediction
        # scales of YOLOv3.
        'stem': {
            'channels': (32,),
            'kernels': (3,),
            'strides': (1,),
            'pool': False,
        },
        'prelude': {
            'maxpool': 0,
            'blocks': (
                # downsample c64  + 1 residual block  -> 8x downsampling
                (RESIDUAL_BLOCK, ((64, 3),) + ((64, 1), (64, 3)) * 1),
                # downsample c128 + 2 residual blocks
                (RESIDUAL_BLOCK, ((128, 3),) + ((128, 1), (128, 3)) * 2),
            ),
        },
        'body': {
            'maxpool': 0,
            'blocks': (
                # downsample c256 + 8 residual blocks     -> 16x downsampling
                (RESIDUAL_BLOCK, ((256, 3),) + ((256, 1), (256, 3)) * 8),
                # downsample c512 + 8 residual blocks
                (RESIDUAL_BLOCK, ((512, 3),) + ((512, 1), (512, 3)) * 8),
            ),
        },
        'coda': {
            'maxpool': 0,
            'blocks': (
                # downsample c1024 + 4 residual blocks    -> 32x downsampling (YOLOv3's third scale)
                (RESIDUAL_BLOCK, ((1024, 3),) + ((1024, 1), (1024, 3)) * 4),
            ),
        },
        'avgpool': True,
        'num_classes': 1000,
    },
}


class Darknet(BasicModel):
    '''
    Darknet backbone, reproducing the Darknet-19 of YOLOv2 and the Darknet-53 of YOLOv3 from
    the official cfg files.

    The network is assembled from four stages, keeping the ``Darknet`` naming of YOLOv1::

        stem     : leading convolution (feature extraction for detection stops here)
        prelude  : several leading 2x2 stride=2 poolings + several blocks
        body     : leading pooling + several blocks in the middle
        coda     : leading pooling + the last blocks, outputs the highest-level semantic features

    Every stage consists of "a number of leading poolings + several blocks", and there are two
    kinds of block:

    - **conv block**: several plain convolutions, where the leading convolution downsamples and
      changes the number of channels and the remaining ones alternate 3x3 up-channel / 1x1
      down-channel (the Darknet19 style).
    - **residual block**: one downsampling convolution at the start, after which every two
      convolutions form a 1x1 -> 3x3 bottleneck residual block (the Darknet53 style).

    ``variant='19'`` (input 416; all downsampling is done by 5 maxpools)::

        stem     : conv 3x3 s1 c32                                      -> 416 x 416
        prelude  : pool -> conv 3x3 c64                                 -> 208 x 208  (2x)
        body     : pool -> conv c128 -> c64 -> c128
                   pool -> conv c256 -> c128 -> c256                    ->  52 x  52  (8x)
        coda     : pool -> conv c512  -> c256 -> c512  -> c256 -> c512
                   pool -> conv c1024 -> c512 -> c1024 -> c512 -> c1024 ->  13 x  13  (32x)
        avgpool -> conv 1x1 c1000 -> [B, 1000]

    ``variant='53'`` (input 416; downsampling is done by the stride=2 convolution at the start of
    every block)::

        stem     : conv 3x3 s1 c32                                      -> 416 x 416
        prelude  : conv 3x3 s2 c64   -> Residual c64            x1      -> 208 x 208
                   conv 3x3 s2 c128  -> Residual c128           x2      -> 104 x 104  (8x)
        body     : conv 3x3 s2 c256  -> Residual c256           x8      ->  52 x  52
                   conv 3x3 s2 c512  -> Residual c512           x8      ->  26 x  26  (16x)
        coda     : conv 3x3 s2 c1024 -> Residual c1024          x4      ->  13 x  13  (32x)
        avgpool -> conv 1x1 c1000 -> [B, 1000]

    How the naming maps onto the official cfg: stem is the first ``[convolutional]``; prelude /
    body / coda are the stages cut apart by ``[maxpool]`` (Darknet19) or ``# Downsample``
    (Darknet53); the classification head is the trailing
    ``[avgpool] -> conv 1x1 filters=1000``. Downsampling only happens in two places -- the
    leading 2x2 stride=2 pooling of a stage, or the stride=2 convolution at the start of a block
    -- and the two are mutually exclusive inside one stage, so the total factor is always 32.

    Three ways to use it:

    - **ImageNet classification** (the default ``num_classes=1000``): global average pooling and
      a 1x1 classification convolution at the end, exactly as in the official cfg; ``conv_count``
      is 19 and 53 respectively, which is where the two network names come from.
    - **Detection backbone** (``num_classes=0``): the classification head is removed and
      ``forward`` returns the grid feature map downsampled by 32 (13x13 for a 416 input) for the
      detection head.
    - **Multi-scale features** (``forward(x, stage='all')``): returns the feature maps of the
      three ``prelude / body / coda`` scales. For ``variant='53'`` these are 104x104 (8x) /
      26x26 (16x) / 13x13 (32x), exactly the FPN scales of YOLOv3; for ``variant='19'`` they are
      208x208 (2x) / 52x52 (8x) / 13x13 (32x).

    **Size and multi-scale training**

    The backbone is "fully convolutional + global average pooling + 1x1 classification
    convolution" with no Flatten fully connected layer, so **any side length that is a multiple
    of 32 can be forwarded**; ``size`` only determines the recorded ``grid_size`` and affects no
    weight shape (the parameter count is constant). The same model instance can therefore change
    size throughout training -- which is the standard way to remove the "train at 256, infer at
    416" distribution mismatch:

    1. Every batch (or every few epochs) draw a side length with ``sample_train_size``, resize /
       random-crop the image to it and forward. The running stats of batch norm accumulate from
       the forward passes automatically, so no extra work is needed.
    2. The deployment size must take a large enough share (guaranteed with
       ``sample_train_size(focus=(416,))``), otherwise the statistics are still dominated by the
       other sizes. The ``min_crop=128, max_crop=448`` of the official ``darknet19.cfg`` follows
       exactly this idea.
    3. Evaluate directly at the deployment size once training is done; do not extrapolate from
       the numbers measured at the training size.

    To reproduce a single-size experiment strictly, pass ``keep_size=True`` to ``forward`` to
    restore the old check that "the side length must equal the ``size`` used at construction
    time". Note that the batch norm statistics are tied to the resolution: training at a single
    size makes inference at another size lose accuracy, in which case either do multi-scale
    training as above or re-estimate the statistics at the target size afterwards.

    Attributes:
        variant (str): '19' or '53', determining the depth and the channel specifications.
        num_classes (int): Number of classes; 0 means the classification head was removed.
        size (int): Reference input side length declared at construction time (also the one used
            by ``verify_shapes`` and ``grid_size``); it does not restrict the forward pass: any
            side length that is a multiple of 32 is accepted.
        downsample (int): Global downsampling factor, fixed at 32.
        grid_size (int): Side length of the output feature map at ``size``.
        stem (nn.Sequential): The leading convolution sequence.
        prelude (nn.Sequential): The first stage.
        body (nn.Sequential): The middle stage.
        coda (nn.Sequential): The last stage.
        avgpool (nn.AdaptiveAvgPool2d, optional): Global average pooling, None when
            ``num_classes=0``.
        classifier (nn.Conv2d, optional): 1x1 classification convolution, None when
            ``num_classes=0``.
        block_channels (Tuple[int, ...]): Input channels of every block.
        stage_channels (Tuple[int, int, int]): Output channels of prelude / body / coda.
        conv_specs (Tuple[Tuple[int, int], ...]): Per-convolution ``(output channels, kernel
            size)`` table (including the classification convolution).
        conv_count (int): Total number of convolution layers (19 or 53 when ``num_classes>0``,
            one fewer otherwise).
        residual_count (int): Total number of residual blocks (0 for Darknet19, 23 for
            Darknet53).
    '''

    #: The official configuration table, keyed by variant name.
    SPECS = DARKNET_SPECS

    def __setattr__(self, name: str, value: Any) -> None:
        '''
        Intercepts assignment to ``activation`` so that a shared activation module is not
        registered as a submodule.

        When the activation being passed in is a ``BasicModel`` subclass (for example
        ``LeakyReLU(0.1)``), assigning it directly makes ``nn.Module`` register it in
        ``_modules``, so ``state_dict`` gains entries such as ``activation.negative_slope`` and
        the key matching of ``from_remote`` / ``load_state_dict`` breaks. Here the activation is
        split in two: ``activation`` only exposes a stable string description, while the original
        object lives in the plain attribute ``_activation_value`` (which never enters
        ``_modules``) and is retrieved through ``activation_module`` when building the model --
        this way the slope of ``LeakyReLU(0.1)`` takes effect unchanged.

        Args:
            name (str): Attribute name.
            value (Any): Attribute value.

        Raises:
            ValueError: If ``activation`` receives a module with learnable parameters, or an
                unsupported type.
        '''
        if name == 'activation':
            if isinstance(value, BasicModel):
                params = [n for n, _ in value.named_parameters()]
                if params:
                    raise ValueError(
                        f'activation module {type(value).__name__} has learnable parameters '
                        f'{params}, so it cannot be passed as a shared instance; pass the '
                        f'activation name string instead'
                    )
            elif value is not None and not isinstance(value, str):
                raise ValueError(
                    f'activation must be a string, None or a parameter-free activation module, '
                    f'got {type(value).__name__}'
                )
            # Both attributes are written straight into the instance dict: this bypasses the
            # submodule registration of nn.Module and keeps state_dict clean.
            object.__setattr__(self, '_activation_value', value)
            object.__setattr__(self, '_activation_name', activation_name(value))
            return

        super().__setattr__(name, value)

    @property
    def activation(self) -> Optional[str]:
        '''
        String description of the activation configuration: 'leakyrelu', None, or the class
        name when an instance was passed in (for example 'leakyrelu').
        '''
        return getattr(self, '_activation_name', None)

    @property
    def activation_module(self) -> Union[str, None, nn.Module]:
        '''
        Returns the activation object actually used when building the model: the instance
        itself when a module was passed in, otherwise the string name.

        Model building always uses this instead of the ``activation`` string, so that a custom
        slope such as ``LeakyReLU(0.1)`` takes effect unchanged -- the string 'leakyrelu' going
        through ``get_activation`` would fall back to the codon default slope of 0.01.
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
        Initializes the Darknet backbone.

        Args:
            variant (str, optional): '19' (YOLOv2) or '53' (YOLOv3). Defaults to '19'.
            in_channels (int, optional): Number of input image channels. Defaults to 3.
            num_classes (int, optional): Number of classes; 0 removes global pooling and the
                classification convolution so that only the feature map is output.
                Defaults to 1000.
            size (int, optional): Input image side length, which must be a multiple of 32. 416
                gives a 13x13 grid, 608 gives 19x19 and ImageNet pretraining commonly uses 256.
                This value is only a reference (it determines ``grid_size`` and the size used for
                the shape self-check at construction time), while the forward pass accepts any
                side length that is a multiple of 32. Defaults to 416.
            norm (str, optional): Normalization type; the official configuration uses 'batch'.
                Pass None to disable normalization. Defaults to 'batch'.
            activation (Union[str, BasicModel], optional): Activation function, a leaky ReLU
                with slope 0.1 by default. Defaults to act.

        Raises:
            ValueError: If ``variant`` is not in the configuration table, if ``size`` is not a
                multiple of 32, or if the channels/kernel sizes inside a specification contradict
                each other.
        '''
        super().__init__()

        variant = str(variant)
        if variant not in self.SPECS:
            raise ValueError(
                f'unsupported Darknet variant {variant!r}, choose from {sorted(self.SPECS)}'
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

        # The last step of construction: really run a dummy input so that errors such as
        # "channels do not line up" surface at instantiation time instead of during training or
        # inference. See verify_shapes.
        self.verify_shapes()

    # ------------------------------------------------------------------
    # Configuration derivation
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_grid(size: int, downsample: int = GLOBAL_DOWNSAMPLE) -> int:
        '''
        Validates the input side length and derives the output feature map side length.

        The downsampling of Darknet consists of maxpools and stride=2 convolutions, both of
        which are exact 2x reductions for even side lengths, so the total factor is fixed at
        ``downsample`` (32 for both Darknet19 and Darknet53).

        Args:
            size (int): Input image side length.
            downsample (int, optional): Global downsampling factor. Defaults to 32.

        Returns:
            int: Side length of the output feature map, i.e. ``size // downsample``.

        Raises:
            ValueError: If ``size`` is not a multiple of ``downsample``.
        '''
        if size % downsample != 0:
            raise ValueError(
                f'input side {size} is not a multiple of {downsample}, so {size}x{size} cannot '
                f'be downsampled exactly; the closest valid side is '
                f'{Darknet.nearest_valid_size(size, downsample)}'
            )
        return size // downsample

    @staticmethod
    def nearest_valid_size(size: int, downsample: int = GLOBAL_DOWNSAMPLE) -> int:
        '''
        Returns the closest valid input side length (a multiple of ``downsample``).

        Args:
            size (int): Desired input side length.
            downsample (int, optional): Global downsampling factor. Defaults to 32.

        Returns:
            int: The closest multiple of ``downsample``, at least ``downsample``.
        '''
        return max(downsample, int(round(size / downsample)) * downsample)

    def _stem_schema(self) -> Dict[str, Any]:
        '''Returns the stem specification of the current variant and validates
        ``channels/kernels/strides`` into tuples of ints item by item.'''
        schema = self.SPECS[self.variant]['stem']
        channels = tuple(int(c) for c in schema['channels'])
        kernels = tuple(int(k) for k in schema['kernels'])
        strides = tuple(int(s) for s in schema['strides'])
        if not (len(channels) == len(kernels) == len(strides)) or not channels:
            raise ValueError(
                f'the channels/kernels/strides of the stem must have equal length and be '
                f'non-empty, got {channels} / {kernels} / {strides}'
            )
        if any(c <= 0 for c in channels) or any(s not in (1, 2) for s in strides):
            raise ValueError(
                f'the stem channels must be positive and the strides must be 1 or 2, got '
                f'{channels} / {strides}'
            )
        if any(k <= 0 or k % 2 == 0 for k in kernels):
            raise ValueError(f'the stem kernel sizes must be positive odd numbers, got {kernels}')
        return {'channels': channels, 'kernels': kernels, 'strides': strides,
                'pool': bool(schema['pool'])}

    @staticmethod
    def _expand_spec(spec: Sequence[Any], block_type: str) -> Tuple[Tuple[int, int], ...]:
        '''
        Expands the channel specification of a block into a per-convolution
        ``(out_channels, kernel_size)`` table.

        Two equivalent notations (matching the line order of the official cfg):

        - **All integers**: every integer is the number of output channels of one convolution and
          the kernel sizes are cycled from ``DEFAULT_KERNELS[block_type]`` (3x3 / 1x1 / 3x3 for a
          residual block, which is exactly "downsampling convolution + (1x1, 3x3) pairs").
        - **All (channels, kernel size) tuples**: the kernel size of every layer is given
          explicitly, for cases the default notation cannot express (such as a residual block
          whose compression ratio is not 1:1).

        Args:
            spec (Sequence[Any]): Channel specification of this block.
            block_type (str): ``CONV_BLOCK`` or ``RESIDUAL_BLOCK``.

        Returns:
            Tuple[Tuple[int, int], ...]: Per-convolution (output channels, kernel size).

        Raises:
            ValueError: If the specification is empty, if the two notations are mixed, or if a
                channel count / kernel size is invalid.
        '''
        if block_type not in DEFAULT_KERNELS:
            raise ValueError(
                f'unsupported block type {block_type!r}, choose from {sorted(DEFAULT_KERNELS)}'
            )
        if not spec:
            raise ValueError(f'the block specification cannot be empty, got {spec!r}')

        if all(isinstance(item, int) for item in spec):
            kernels = DEFAULT_KERNELS[block_type]
            if len(spec) % len(kernels) != 0:
                raise ValueError(
                    f'the integer notation of a {block_type} block must have a length that is '
                    f'a multiple of {len(kernels)}, got {len(spec)} channels {tuple(spec)}; to '
                    f'specify the kernel size of each layer, write (channels, kernel size) '
                    f'tuples instead'
                )
            expanded = tuple(
                (int(channel), kernels[index % len(kernels)])
                for index, channel in enumerate(spec)
            )
        elif all(isinstance(item, (tuple, list)) and len(item) == 2 for item in spec):
            expanded = tuple((int(channel), int(kernel)) for channel, kernel in spec)
        else:
            raise ValueError(
                f'the specification of a {block_type} block must be either all integers (using '
                f'the default kernel sizes) or all (channels, kernel size) tuples, mixing is not '
                f'allowed, got {spec!r}'
            )

        for channel, kernel in expanded:
            if channel <= 0 or kernel <= 0 or kernel % 2 == 0:
                raise ValueError(
                    f'channel counts must be positive integers and kernel sizes must be '
                    f'positive odd numbers, got (channels={channel}, kernel={kernel})'
                )
        return expanded

    def _part_schema(self, part: str) -> Dict[str, Any]:
        '''Returns the specification of one stage and validates ``maxpool`` and the block
        notation.'''
        if part not in ('prelude', 'body', 'coda'):
            raise ValueError(f'unsupported stage name {part!r}')
        schema = self.SPECS[self.variant][part]
        pools = int(schema['maxpool'])
        if not 0 <= pools <= MAX_POOL_PER_PART:
            raise ValueError(
                f'the {part} stage of {self.variant} declares {pools} poolings, which must be '
                f'between 0 and {MAX_POOL_PER_PART}'
            )
        if not schema['blocks']:
            raise ValueError(f'the {part} stage of {self.variant} needs at least one block')
        return {'maxpool': pools, 'blocks': tuple(schema['blocks'])}

    def _walk_block(
        self,
        expanded: Sequence[Tuple[int, int]],
        block_type: str,
        block_name: str
    ) -> int:
        '''
        Derives the channel flow along the convolution sequence of one block and validates that
        the configuration is internally consistent.

        The rules map one to one onto the notation of the official cfg:

        - The sequence is cut by kernel size into runs of consecutive layers that share a kernel.
          The layers of one run consume the same feature map in turn, so their channel counts must
          be identical -- this explains both why a 3x3 up-channel must be followed immediately by
          a 1x1 down-channel and why ``filters=C`` repeats consecutively inside a Darknet53
          residual block.
        - A channel change across runs is legal: the 1x1 squeeze (128 -> 64 in Darknet19) and the
          3x3 projection (32 -> 64 in Darknet53) both change channels by switching kernels.
        - The entry channels of a block are decided by the upstream, so **the first layer may
          change the channel count freely** (it is the downsampling/projection convolution of this
          block); the inputs of every remaining layer must be lined up inside the block itself and
          an inconsistency raises here immediately.

        Args:
            expanded (Sequence[Tuple[int, int]]): The per-convolution (channels, kernel) expanded
                by ``_expand_spec``.
            block_type (str): ``CONV_BLOCK`` or ``RESIDUAL_BLOCK``.
            block_name (str): Block name used in error messages.

        Returns:
            int: Number of output channels of the block.

        Raises:
            ValueError: If the channels inside a single-kernel run disagree, or if the residual
                specification is invalid.
        '''
        if block_type == RESIDUAL_BLOCK:
            if len(expanded) < 3 or len(expanded) % 2 == 0:
                raise ValueError(
                    f'the residual specification of {block_name} must be 1 + 2n convolutions '
                    f'(one downsampling convolution + n (1x1, 3x3) pairs), got {expanded}'
                )
            for index in range(1, len(expanded) - 1, 2):
                inner, outer = expanded[index][0], expanded[index + 1][0]
                if inner != outer:
                    raise ValueError(
                        f'the 1x1 and 3x3 output channels of residual block {(index + 1) // 2} '
                        f'of {block_name} must be identical (the shortcut is added '
                        f'element-wise), got {inner} and {outer}'
                    )

        start = 0
        while start < len(expanded):
            kernel = expanded[start][1]
            end = start
            while end + 1 < len(expanded) and expanded[end + 1][1] == kernel:
                end += 1
            group = expanded[start:end + 1]

            if len(group) > 1 and start:
                # Several consecutive layers with the same kernel stack on one feature map, so
                # their channel counts must agree; the leading run has its input decided by the
                # upstream, so only agreement inside the run is required, not with the upstream.
                first_channel = group[0][0]
                for offset, (channel, _) in enumerate(group[1:], start=1):
                    if channel != first_channel:
                        raise ValueError(
                            f'the channel flow of {block_name} breaks at layer '
                            f'{start + offset + 1}: consecutive {kernel}x{kernel} convolutions '
                            f'share one feature map, but the previous one has {first_channel} '
                            f'channels while this one declares {channel}'
                        )
            start = end + 1

        return expanded[-1][0]

    def resolve_conv_specs(self) -> Tuple[Tuple[int, int], ...]:
        '''
        Expands every convolution in the order of the official cfg: stem -> prelude -> body ->
        coda -> classification convolution.

        It does two things at the same time:

        1. Validate the channel flow block by block (see ``_walk_block``), so configuration errors
           surface at construction time instead of raising hard-to-locate errors such as
           ``mat1 and mat2 shapes cannot be multiplied`` during the forward pass.
        2. Collect the per-convolution ``(output channels, kernel size)`` table for downstream use
           (the YOLO detection head, the shape checks).

        Returns:
            Tuple[Tuple[int, int], ...]: Per-convolution ``(output channels, kernel size)`` table
                (including the classification convolution).

        Raises:
            ValueError: If the channel flow inside a block breaks, or if a residual specification
                is invalid.
        '''
        stem = self._stem_schema()
        specs: List[Tuple[int, int]] = list(zip(stem['channels'], stem['kernels']))

        for part in ('prelude', 'body', 'coda'):
            schema = self._part_schema(part)
            for index, (block_type, spec) in enumerate(schema['blocks']):
                expanded = self._expand_spec(spec, block_type)
                self._walk_block(expanded, block_type, f'block {index + 1} of the {part} stage')
                specs.extend(expanded)

        if self.num_classes:
            specs.append((self.num_classes, 1))
        return tuple(specs)

    def resolve_block_strides(self) -> Tuple[Tuple[str, int, int], ...]:
        '''
        Derives the input channels of every block and the stride of its leading convolution.

        The stride rules correspond exactly to ``[maxpool]`` / ``# Downsample`` in the official
        cfg:

        - When the stage has poolings: the poolings perform the downsampling of the stage, so the
          leading convolution of the **first block** has stride 1 and every remaining block
          downsamples once more with a stride=2 convolution.
        - When the stage has no pooling: the leading convolution of every block has stride 2.

        Returns:
            Tuple[Tuple[str, int, int], ...]: Per-block ``(block name, input channels, stride of
                the leading convolution)``.

        Raises:
            ValueError: If the derived number of downsampling steps does not match
                ``log2(downsample)``.
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
                # The leading poolings already performed the downsampling of this stage, so only
                # the blocks after them continue downsampling with a stride=2 convolution.
                stride = 2 if index >= pools else 1
                blocks.append((f'{part}[{index}]', current, stride))
                downsample_points += 1 if stride == 2 else 0
                current = self._expand_spec(spec, block_type)[-1][0]

        expect = int(math.log2(self.downsample))
        if downsample_points != expect:
            raise ValueError(
                f'Darknet-{self.variant} has {downsample_points} 2x downsampling steps for a '
                f'total factor of {2 ** downsample_points}, which does not match '
                f'downsample={self.downsample}'
            )
        return tuple(blocks)

    def resolve_part_inputs(self) -> Dict[str, int]:
        '''
        Derives the input channels of the three stages ``prelude / body / coda``.

        The connections between stages are explicit: prelude takes the output of stem, body takes
        the output of the last block of prelude and coda takes the output of the last block of
        body; there is no convolution between adjacent stages, only pooling, so the channel count
        does not change.

        Returns:
            Dict[str, int]: ``{'prelude': c0, 'body': c1, 'coda': c2}``.
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
        Validates that the total number of convolution layers matches the official configuration
        and stores it in ``conv_count``.

        The Darknet names come directly from the number of weight layers: Darknet19 is 18
        convolutions + 1 fully connected classification layer and Darknet53 is 52 convolutions + 1
        fully connected classification layer; this implementation writes the classification layer
        as a 1x1 convolution too, so ``conv_count`` equals 19 and 53 respectively.

        Args:
            conv_specs (Sequence[Tuple[int, int]]): The result of ``resolve_conv_specs``.

        Raises:
            ValueError: If the expansion does not match ``EXPECTED_CONVS``.
        '''
        self.conv_count = len(conv_specs)
        expected = EXPECTED_CONVS.get(self.variant)
        if expected is not None:
            expected += 1 if self.num_classes else 0
            if self.conv_count != expected:
                raise ValueError(
                    f'the configuration of Darknet-{self.variant} expands to '
                    f'{self.conv_count} convolution layers, which does not match the {expected} '
                    f'of the official configuration (numb_classes={self.num_classes}); check '
                    f'SPECS[{self.variant!r}]'
                )

    # ------------------------------------------------------------------
    # Stage construction
    # ------------------------------------------------------------------

    def _build_stem(self) -> int:
        '''
        Builds ``stem``: the leading convolution sequence.

        The stem of both variants is a single 3x3 convolution with stride=1 (downsampling is left
        to the pooling of prelude or the stride=2 convolution at the start of a block); the
        specification table keeps the full ``channels / kernels / strides`` notation to make
        adding variants easy.

        Returns:
            int: Number of output channels of the stem.
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
        Builds a Darknet convolution with normalization and activation (``pad = kernel // 2``).

        The activation is taken from ``activation_module`` rather than the ``activation`` string:
        a passed-in ``LeakyReLU(0.1)`` is reused as is instead of letting
        ``get_activation('leakyrelu')`` fall back to the codon default slope of 0.01.
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
        Builds one block: the leading convolution downsamples and is followed by several
        convolutions or bottleneck residual blocks with unchanged channels.

        Args:
            in_channels (int): Number of input channels of the block.
            block_type (str): ``CONV_BLOCK`` or ``RESIDUAL_BLOCK``.
            spec (Sequence[Any]): Channel specification, see ``_expand_spec`` for its meaning.
            stride (int, optional): Stride of the leading convolution. Defaults to 2.

        Returns:
            nn.Sequential: The layer sequence of this block.
        '''
        expanded = self._expand_spec(spec, block_type)
        layers: List[nn.Module] = []
        current = in_channels
        index = 0
        while index < len(expanded):
            if block_type == RESIDUAL_BLOCK and index:
                # residual block: starting from the second element, every two elements (1x1
                # squeeze, 3x3 expand) form one residual block. The residual block contains both
                # layers itself, so two elements must be skipped as a whole, not just one.
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
        Builds one of the stages ``prelude / body / coda``: leading poolings + several blocks.

        Both the input channels of a block and the stride of its leading convolution come from
        ``resolve_block_strides``, so this method is at once "assembling according to the official
        cfg" and "a runtime validation of configuration consistency": any combination whose
        channels do not line up is ruled out during construction.

        Args:
            part (str): 'prelude', 'body' or 'coda'.

        Returns:
            nn.Sequential: The layer sequence of this stage (poolings + blocks).

        Raises:
            ValueError: If the stage name is not supported.
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
    # Forward pass
    # ------------------------------------------------------------------

    def verify_shapes(self) -> Tuple[int, int, int]:
        '''
        Runs one forward pass with a dummy input to check that the channels and the spatial sizes
        really line up.

        The static checks (the self-consistency of the spec tables) can only cover structural
        constraints such as "the channels inside a single-kernel run agree"; what really decides
        whether feature maps connect is the in/out channels of every convolution, and in_channels
        depends on the upstream -- so a real forward pass is run here: any channel mismatch
        raises a ``RuntimeError`` when ``Darknet`` is instantiated instead of blowing up during
        training.

        Returns:
            Tuple[int, int, int]: Number of output channels of the prelude / body / coda feature
                maps.

        Raises:
            RuntimeError: If a shape mismatch occurs during the forward pass (channels or spatial
                sizes do not line up).
        '''
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                probe = torch.zeros(1, self.in_channels, self.size, self.size)
                channels = tuple(feature.shape[1] for feature in self.stage_features(probe))
        except RuntimeError as exc:  # pragma: no cover - only triggered by a miswritten config
            raise RuntimeError(
                f'the channel/size validation of Darknet-{self.variant} failed '
                f'(size={self.size}): {exc}; check the channel specifications in '
                f'SPECS[{self.variant!r}]'
            ) from exc
        finally:
            self.train(was_training)
        return channels

    def stage_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        '''
        Returns the feature maps of the three scales, i.e. the outputs of the ``prelude / body /
        coda`` stages.

        For a 416 input, ``variant='53'`` gives 104x104 (8x downsampling), 26x26 (16x) and 13x13
        (32x), exactly the FPN scales of YOLOv3; ``variant='19'`` gives 208x208 (2x), 52x52 (8x)
        and 13x13 (32x).

        Args:
            x (torch.Tensor): Input image of shape [Batch, in_channels, size, size].

        Returns:
            List[torch.Tensor]: A list of 3 feature maps whose spatial sizes are halved in turn.
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
        The backbone is "fully convolutional + global average pooling + 1x1 classification
        convolution", so **any side length that is a multiple of 32 can be forwarded directly**
        and need not equal the ``size`` used at construction time: ``size`` only derives and
        records ``grid_size`` and has nothing to do with the kernel shapes. This is how
        multi-scale training works -- one model instance switches side length every batch so that
        the batch norm statistics cover all the resolutions, removing the "train at 256, infer at
        416" distribution mismatch.

        Two things to keep in mind:

        - Changing the size **changes the batch norm statistics** (in training mode the running
          stats are pulled towards the distribution of the new resolution), which is exactly what
          multi-scale training wants; if only one size is wanted, do not mix them, or re-estimate
          the statistics at that size afterwards.
        - ``keep_size=True`` restores the old behaviour (the side length must equal the ``size``
          used at construction time), for cases that need to reproduce a single-size experiment
          strictly.

        Args:
            x (torch.Tensor): Input image of shape [Batch, in_channels, side, side], where
                ``side`` must be a multiple of 32 (224 / 256 / 320 / 416 / 448 and so on); with
                ``keep_size=True`` it must also equal the ``size`` used at construction time.
            stage (Literal['coda', 'all'], optional): Which output to take. 'coda' returns the
                highest-level feature map (the class logits when there is a classification head);
                'all' returns the list of three feature maps from ``stage_features``.
                Defaults to 'coda'.
            keep_size (bool, optional): Whether to force the input side length to match the
                ``size`` used at construction time. Defaults to False.

        Returns:
            torch.Tensor: With ``stage='coda'``, the class logits ``[Batch, num_classes]`` when
                ``num_classes > 0``, otherwise the grid feature map
                ``[Batch, C, side // 32, side // 32]``; with ``stage='all'``, a list of 3 feature
                maps.

        Raises:
            ValueError: If the input is not 4-dimensional, not square or not a multiple of 32 (or,
                with ``keep_size=True``, its side length does not match ``size``), or if ``stage``
                has an invalid value.
        '''
        if x.ndim != 4:
            raise ValueError(
                f'the Darknet backbone expects a 4D input [B, C, H, W], got {tuple(x.shape)}'
            )

        height, width = x.shape[-2:]
        if height != width:
            raise ValueError(
                f'the Darknet backbone is a square downsampling structure, so a square input is '
                f'expected, got {height}x{width}'
            )
        if height % self.downsample != 0:
            raise ValueError(
                f'the input side {height} is not a multiple of {self.downsample} (the backbone '
                f'has {int(math.log2(self.downsample))} 2x downsampling steps); the closest '
                f'valid side is {self.nearest_valid_size(height, self.downsample)}'
            )
        if keep_size and height != self.size:
            raise ValueError(
                f'with keep_size=True the input side must equal the size used at construction '
                f'time, size={self.size}, got {height}; the output grid of this backbone is '
                f'fixed at {self.grid_size}x{self.grid_size}'
            )

        if stage == 'all':
            return self.stage_features(x)
        if stage != 'coda':
            raise ValueError(f"stage only supports 'coda' or 'all', got {stage!r}")

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
        Lists every valid input side length (a multiple of ``step``) inside
        ``[min_size, max_size]``.

        Multi-scale training uses it to obtain all the candidate steps at once, for example
        ``(256, 320, 352, 416, 448)``.

        Args:
            min_size (int, optional): Lower bound of the interval (inclusive), rounded up to a
                multiple of ``step``. Defaults to 256.
            max_size (int, optional): Upper bound of the interval (inclusive), rounded down to a
                multiple of ``step``. Defaults to 448.
            step (int, optional): Step, which must equal the downsampling factor of the backbone.
                Defaults to 32.

        Returns:
            Tuple[int, ...]: Tuple of valid side lengths in ascending order; an exception is
                raised when the interval contains no valid value.

        Raises:
            ValueError: If ``min_size``/``max_size`` are invalid, or if the interval contains no
                multiple of ``step``.
        '''
        if step <= 0 or min_size <= 0 or max_size < min_size:
            raise ValueError(
                f'invalid size interval: min_size={min_size}, max_size={max_size}, step={step}'
            )
        low = math.ceil(min_size / step) * step
        high = math.floor(max_size / step) * step
        if low > high:
            raise ValueError(
                f'there is no multiple of {step} inside [{min_size}, {max_size}], widen the '
                f'interval'
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
        Draws a random input side length for multi-scale training.

        By default the value is drawn **uniformly** from the valid steps inside
        ``[min_size, max_size]``. Giving ``focus`` biases it towards certain sizes (for instance
        ``focus=(416,)`` when deploying at 416) -- in that case the value comes from ``focus``
        with probability ``focus_ratio`` and uniformly from the interval otherwise. This keeps the
        batch norm coverage that multi-scale training provides while giving the statistics of the
        target size enough weight (feeding 416 only occasionally is useless, the statistics are
        still dominated by 256).

        The intended use is to draw once per batch (or per epoch), resize / random-crop the image
        to that side length and forward; the statistics then update themselves because the model
        accumulates them during its own forward passes, so no extra work is needed.

        Args:
            min_size (int, optional): Lower bound of the interval (inclusive). Defaults to 256.
            max_size (int, optional): Upper bound of the interval (inclusive). Defaults to 448.
            focus (Optional[Sequence[int]], optional): Sizes to cover especially, aligned to a
                multiple of ``step`` first, with invalid entries ignored. Defaults to None.
            focus_ratio (float, optional): Probability of drawing from ``focus``, in 0..1.
                Defaults to 0.5.
            step (int, optional): Step of the valid side lengths. Defaults to 32.
            generator (Optional[torch.Generator], optional): Random number generator, useful for
                reproducibility. Defaults to None.

        Returns:
            int: The input side length to use for this training step.

        Raises:
            ValueError: If the interval is invalid, if ``focus_ratio`` is out of range, or if
                ``focus`` contains no valid size at all.
        '''
        candidates = cls.valid_sizes(min_size, max_size, step)
        if not 0.0 <= focus_ratio <= 1.0:
            raise ValueError(f'focus_ratio must lie in [0, 1], got {focus_ratio}')

        focused: Tuple[int, ...] = ()
        if focus:
            focused = tuple(sorted({
                int(size) for size in focus
                if int(size) % step == 0 and min_size <= int(size) <= max_size
            }))
            if not focused:
                raise ValueError(
                    f'focus={tuple(focus)} contains no size that lies in [{min_size}, {max_size}] '
                    f'and is a multiple of {step}'
                )

        if focused and focus_ratio > 0:
            # A single uniform draw decides both "focus or interval" and "which element", so the
            # proportion is exactly focus_ratio and does not depend on the initial state of the
            # random stream.
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
        Pads the input with zeros up to a multiple of ``downsample`` (only on the right and
        bottom).

        With irregular data sizes (images of arbitrary resolution, for example) padding the side
        lengths to a multiple of 32 before feeding the backbone avoids the size validation error.
        Zero padding does not change the batch norm statistics, it only turns the extra region
        into a constant.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
            downsample (int, optional): Target downsampling factor. Defaults to 32.

        Returns:
            Tuple[torch.Tensor, Tuple[int, int, int, int]]: The padded tensor and the
                ``(left, right, top, bottom)`` padding, which makes it easy to map predicted
                coordinates back onto the original image.
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
        Builds the model automatically from the input/output shapes.

        Args:
            input_shape (Tuple[int, ...]): Input shape without the batch dimension, e.g.
                (3, 416, 416).
            output_shape (Optional[Tuple[int, ...]], optional): Output shape. When given, its last
                dimension becomes the number of classes and the classification head is kept; when
                None the classification head is removed (detection backbone). Defaults to None.
            variant (str, optional): '19' or '53'. Defaults to '19'.
            norm (str, optional): Normalization type. Defaults to 'batch'.
            activation (Union[str, BasicModel], optional): Activation function. Defaults to act.

        Returns:
            Darknet: The constructed backbone.
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
    '''Self-check: prints the output shape, the three-scale features, the convolution count and
    the parameter count of both variants.'''
    for variant, expect in (('19', 19), ('53', 53)):
        model = Darknet(variant=variant, size=416)
        with torch.no_grad():
            logits = model(torch.randn(1, 3, 416, 416))
            features = model(torch.randn(1, 3, 416, 416), stage='all')
        print(f'Darknet-{variant} 416 -> {tuple(logits.shape)}')
        print('  three-scale features:', [tuple(f.shape) for f in features])
        print('  conv layers :', model.conv_count, '| expect', expect,
              '| residual blocks', model.residual_count)
        print('  channel table:', model.block_channels, '/', model.stage_channels)
        print('  parameters  :', model.count_params(human_readable=True))

    backbone = Darknet(variant='53', num_classes=0, size=416)
    with torch.no_grad():
        feat = backbone(torch.randn(1, 3, 416, 416))
    print('detection backbone Darknet-53 ->', tuple(feat.shape),
          '| params', backbone.count_params(human_readable=True))

    for size in (224, 256, 416, 608):
        model = Darknet(variant='19', size=size)
        with torch.no_grad():
            out = model(torch.randn(1, 3, size, size))
        print(f'Darknet-19 {size} -> {tuple(out.shape)}')
