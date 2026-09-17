from codon import *
from safetensors.torch import load_file

from codon.block.activation import get_activation
from codon.block.norm import LayerNorm


convnext_sizes = {
    'tiny': dict(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
    ),
    'small': dict(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
    ),
    'base': dict(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
    ),
    'large': dict(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
    ),
}


class DropPath(BasicModel):
    '''
    Stochastic Depth (DropPath): randomly drops the whole residual branch per sample during
    training and passes through unchanged at inference time.

    Attributes:
        drop_prob (float): Drop probability.
    '''

    def __init__(self, drop_prob: float = 0.0):
        '''
        Args:
            drop_prob (float, optional): Drop probability. Defaults to 0.0.
        '''
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


class ConvNeXtBlock(BasicModel):
    '''
    ConvNeXt residual block (depthwise + pointwise inverted bottleneck + Layer Scale).

    Attributes:
        dwconv (nn.Conv2d): 7x7 depthwise convolution.
        norm (LayerNorm): channels_last LayerNorm (eps=1e-6).
        proj_pw1 (nn.Linear): Expanding pointwise convolution (dim -> 4*dim).
        act (nn.Module): GELU.
        proj_pw2 (nn.Linear): Contracting pointwise convolution (4*dim -> dim).
        gamma (nn.Parameter, optional): Layer Scale parameter; None when
            ``layer_scale_init_value <= 0``.
        drop_path (nn.Module): DropPath or Identity.
    '''

    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        '''
        Args:
            dim (int): Number of input/output channels.
            drop_path (float, optional): Stochastic depth probability. Defaults to 0.0.
            layer_scale_init_value (float, optional): Initial value of Layer Scale.
                Defaults to 1e-6.
        '''
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        self.norm = LayerNorm(dim, eps=1e-6)
        self.proj_pw1 = nn.Linear(dim, 4 * dim)
        self.act = get_activation('gelu')
        self.proj_pw2 = nn.Linear(4 * dim, dim)
        self.layer_scale_init_value = layer_scale_init_value
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.proj_pw1(x)
        x = self.act(x)
        x = self.proj_pw2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        return residual + self.drop_path(x)


class ConvNeXtBlockTransition(BasicModel):
    '''
    ConvNeXt resolution-changing layer: downsamples the feature map by 2x and switches the number
    of channels to the width of the next stage.

    Attributes:
        pre_norm (bool): True means the order is Norm-Conv.
        conv (nn.Conv2d): Downsampling convolution.
        norm (LayerNorm): channels_first LayerNorm (eps=1e-6).
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        pre_norm: bool = True,
    ):
        '''
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int, optional): Convolution kernel size. Defaults to 2.
            stride (int, optional): Stride. Defaults to 2.
            pre_norm (bool, optional): True for Norm-Conv (stage transition), False for Conv-Norm
                (stem). Defaults to True.
        '''
        super().__init__()
        self.pre_norm = pre_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        
        norm_channels = in_channels if pre_norm else out_channels
        self.norm = LayerNorm(norm_channels, eps=1e-6, channel_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = self.norm(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = self.norm(x)
        return x


class ConvNeXt(BasicModel):
    '''
    DINOv3-style implementation of ConvNeXt (A ConvNet for the 2020s,
    https://arxiv.org/pdf/2201.03545.pdf).

    The variant is selected through ``variant`` (tiny / small / base / large), or ``depths`` and
    ``dims`` may be passed directly.

    Attributes:
        stem (ConvNeXtBlockTransition): 4x downsampling stem.
        transitions (nn.ModuleList): Three 2x downsampling transition layers.
        stages (nn.ModuleList): Four stages, each an ``nn.Sequential(ConvNeXtBlock x depths[i])``.
        norm (LayerNorm): Final LayerNorm (eps=1e-6).
        norms (list): ``[Identity, Identity, Identity, norm]``, used by
            ``get_intermediate_layers``.
        head (nn.Identity): This implementation is a backbone only, so the head is the identity.
        embed_dim (int): Number of output channels (``dims[-1]``).
        embed_dims (List[int]): Number of output channels of every stage.
        n_blocks (int): Number of stages (4).
        n_storage_tokens (int): Number of register/storage tokens, 0 in this implementation.
        patch_size (int, optional): Pseudo patch size; when not None the feature map is
            interpolated onto the ViT grid.
        input_pad_size (int): Multiple the input must be padded to (the stem has stride=4).
    '''

    variant = None

    def __init__(
        self,
        variant: Optional[str] = None,
        in_channels: int = 3,
        depths: Optional[List[int]] = None,
        dims: Optional[List[int]] = None,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        patch_size: Optional[int] = None,
        **ignored_kwargs,
    ):
        '''
        Args:
            variant (Optional[str], optional): 'tiny' / 'small' / 'base' / 'large'; a subclass may
                instead pin it with a class attribute. When neither variant nor a complete
                depths/dims pair is given it defaults to 'tiny'. Defaults to None.
            in_channels (int, optional): Number of input channels. Defaults to 3.
            depths (Optional[List[int]], optional): Number of blocks per stage; overrides variant
                when given explicitly. Defaults to None.
            dims (Optional[List[int]], optional): Number of channels per stage; overrides variant
                when given explicitly. Defaults to None.
            drop_path_rate (float, optional): Total stochastic depth probability, increasing
                linearly per block. Defaults to 0.0.
            layer_scale_init_value (float, optional): Initial value of Layer Scale.
                Defaults to 1e-6.
            patch_size (Optional[int], optional): Pseudo patch size, used to resize the feature
                map to the ViT grid. Defaults to None.
            **ignored_kwargs: Other configuration entries (such as num_classes) are ignored
                outright, which makes it easy to reuse an external config.
        '''
        super().__init__()

        variant = variant if variant is not None else self.variant
        if variant is None and (depths is None or dims is None): variant = 'tiny'
        if variant is not None:
            if variant not in convnext_sizes:
                raise NotImplementedError(
                    f"didn't recognize convnext variant string: {variant!r}; "
                    f'choose from {list(convnext_sizes)}'
                )
            size_dict = convnext_sizes[variant]
            depths = size_dict['depths'] if depths is None else depths
            dims = size_dict['dims'] if dims is None else dims
        if depths is None or dims is None:
            raise ValueError('either pass variant= or both depths= and dims=')
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError(f'ConvNeXt expects 4 stages, got depths={depths} / dims={dims}')

        self.variant = variant
        self.in_channels = in_channels
        self.depths = list(depths)
        self.dims = list(dims)
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value

        self.stem = ConvNeXtBlockTransition(
            in_channels, dims[0], kernel_size=4, stride=4, pre_norm=False
        )
        self.transitions = nn.ModuleList([
            ConvNeXtBlockTransition(dims[i], dims[i + 1], kernel_size=2, stride=2, pre_norm=True)
            for i in range(3)
        ])

        dp_rates = [float(rate) for rate in np.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[
                ConvNeXtBlock(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                )
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = LayerNorm(dims[-1], eps=1e-6)
        self.norms = [nn.Identity() for _ in range(3)]
        self.norms.append(self.norm)

        self.head = nn.Identity()
        self.embed_dim = dims[-1]
        self.embed_dims = list(dims)
        self.n_blocks = len(self.transitions) + 1
        self.chunked_blocks = False
        self.n_storage_tokens = 0

        self.patch_size = patch_size
        self.input_pad_size = 4

        self.init_weights()

    def init_weights(self) -> None:
        '''Initializes the parameters following ConvNeXt: truncated normal (0.02) for conv/linear
        layers and a constant Layer Scale.'''
        self.apply(self._init_weights)
        for stage in self.stages:
            for block in stage:
                if block.gamma is not None:
                    nn.init.constant_(block.gamma, block.layer_scale_init_value)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        masks: Optional[torch.Tensor] = None,
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        '''
        Extracts the backbone features.

        Args:
            x (Union[torch.Tensor, List[torch.Tensor]]): Images [N, C, H, W], or a list of image
                crops.
            masks (Optional[torch.Tensor], optional): Passed through into the output dictionary
                unchanged, which keeps external crop/mask logic aligned. Defaults to None.

        Returns:
            Union[Dict, List[Dict]]: A single input returns one dictionary (a list input returns a
                list of the same length) with the keys:

                - `x_norm_clstoken`   [N, C]      globally average-pooled CLS token (normalized)
                - `x_storage_tokens`  [N, 0, C]   register tokens, empty in this implementation
                - `x_norm_patchtokens`[N, HW, C]  patch tokens (normalized)
                - `x_prenorm`         [N, HW, C]  patch tokens before normalization
                - `masks`                         the masks passed through
        '''
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        return self.forward_features_list(x, masks)

    def forward_features_list(
        self,
        x_list: List[torch.Tensor],
        masks_list: List[Optional[torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        output = []
        for x, masks in zip(x_list, masks_list):
            x = self.stem(x)
            for i in range(4):
                if i > 0:
                    x = self.transitions[i - 1](x)
                x = self.checkpoint(self.stages[i], x)

            x_pool = x.mean([-2, -1])  # (N, C, H, W) -> (N, C)
            x = torch.flatten(x, 2).transpose(1, 2)  # (N, C, HW) -> (N, HW, C)

            x_norm = self.norm(torch.cat([x_pool.unsqueeze(1), x], dim=1))
            output.append({
                'x_norm_clstoken': x_norm[:, 0],
                'x_storage_tokens': x_norm[:, 1: self.n_storage_tokens + 1],
                'x_norm_patchtokens': x_norm[:, self.n_storage_tokens + 1:],
                'x_prenorm': x,
                'masks': masks,
            })

        return output

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        is_training: bool = False,
        masks: Optional[torch.Tensor] = None,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        '''
        Args:
            x (Union[torch.Tensor, List[torch.Tensor]]): Images [N, C, H, W], or a list of images.
            is_training (bool, optional): True returns the dictionary from ``forward_features``,
                False returns only the CLS features. Defaults to False.
            masks (Optional[torch.Tensor], optional): Passed through to ``forward_features``.
                Defaults to None.

        Returns:
            Union[Dict, torch.Tensor]: The feature dictionary when training, otherwise the
                [N, embed_dim] CLS features.
        '''
        ret = self.forward_features(x, masks=masks)
        if is_training:
            return ret
        return self.head(ret['x_norm_clstoken'])

    def _get_intermediate_layers(self, x: torch.Tensor, n: Union[int, Sequence] = 1) -> List[list]:
        h, w = x.shape[-2:]
        output = []
        blocks_to_take = range(self.n_blocks - n, self.n_blocks) if isinstance(n, int) else n

        x = self.stem(x)
        for i in range(self.n_blocks):
            if i > 0:
                x = self.transitions[i - 1](x)
            x = self.stages[i](x)
            if i in blocks_to_take:
                x_patches = x
                if self.patch_size is not None:
                    x_patches = F.interpolate(
                        x,
                        size=(h // self.patch_size, w // self.patch_size),
                        mode='bilinear',
                        antialias=True,
                    )
                output.append([x.mean([-2, -1]), x_patches])  # [CLS (N, C), patch (N, C, H, W)]

        assert len(output) == len(blocks_to_take), f'only {len(output)} / {len(blocks_to_take)} blocks found'
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> Tuple:
        '''
        Returns intermediate-layer features.

        Args:
            x (torch.Tensor): Images [N, C, H, W].
            n (Union[int, Sequence], optional): Take the last n layers, or pass the layer indices
                directly. Defaults to 1.
            reshape (bool, optional): When True, patch tokens are restored to (N, C, H, W).
                Defaults to False.
            return_class_token (bool, optional): When True, every layer returns a (patch, cls)
                tuple. Defaults to False.
            norm (bool, optional): Whether to apply ``self.norms``. Note that upstream only gives
                the last stage a real LayerNorm while the first three stages are Identity, so with
                ``norm=True`` only the last layer yields normalized features. Defaults to True.

        Returns:
            Tuple: One tensor per layer (or a (patch, cls) tuple per layer).
        '''
        outputs = self._get_intermediate_layers(x, n)

        if norm:
            nchw_shapes = [out[-1].shape for out in outputs]
            if isinstance(n, int):
                norms = self.norms[-n:]
            else:
                norms = [self.norms[i] for i in n]
            outputs = [
                (
                    layer_norm(cls_token),  # N x C
                    layer_norm(patches.flatten(-2, -1).permute(0, 2, 1)),  # N x HW x C
                )
                for (cls_token, patches), layer_norm in zip(outputs, norms)
            ]
            if reshape:
                outputs = [
                    (cls_token, patches.permute(0, 2, 1).reshape(*nchw).contiguous())
                    for (cls_token, patches), nchw in zip(outputs, nchw_shapes)
                ]
        elif not reshape:
            outputs = [
                (cls_token, patches.flatten(-2, -1).permute(0, 2, 1))
                for (cls_token, patches) in outputs
            ]

        class_tokens = [out[0] for out in outputs]
        outputs = [out[1] for out in outputs]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def load_pretrained(
        self,
        path: str,
        strict: bool = True,
        remap: bool = True,
    ) -> 'ConvNeXt':
        '''
        Loads pretrained weights from a local file (.safetensors / .pth / .pt).

        Args:
            path (str): Path to the weight file.
            strict (bool, optional): When True, key names and shapes must match the model exactly.
                Defaults to True.
            remap (bool, optional): When True, ``remap_convnext_keys`` is applied first to map the
                key names of DINOv3/timm/transformers onto this implementation's naming (dropping
                the classification head). Defaults to True.

        Returns:
            ConvNeXt: self.
        '''
        if path.endswith('.safetensors'):
            tensors = dict(load_file(path, device='cpu'))
        else:
            raw = torch.load(path, map_location='cpu')
            if isinstance(raw, dict):
                for wrapper in ('model_state_dict', 'state_dict', 'model'):
                    sub = raw.get(wrapper)
                    if isinstance(sub, dict):
                        raw = sub
                        break
            if not isinstance(raw, dict):
                raise RuntimeError(f'{path} is not a valid PyTorch weight file (expected a dict)')
            tensors = {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}

        if remap:
            tensors = remap_convnext_keys(tensors)

        ref = self.state_dict()
        extra = sorted(k for k in tensors if k not in ref)
        missing = sorted(k for k in ref if k not in tensors)
        if strict and (extra or missing):
            raise RuntimeError(
                f'weights do not match {type(self).__name__}({self.variant}): '
                f'{len(extra)} extra / {len(missing)} missing -> '
                f'extra {extra[:5]}..., missing {missing[:5]}...'
            )

        for key in list(tensors):
            if key not in ref:
                tensors.pop(key)
                continue
            want = tuple(ref[key].shape)
            got = tuple(tensors[key].shape)
            if got != want:
                raise RuntimeError(f'shape mismatch for {key}: weights {got} vs model {want}')
            if tensors[key].dtype != ref[key].dtype:
                tensors[key] = tensors[key].to(ref[key].dtype)

        self.load_state_dict(tensors, strict=False)
        return self


class ConvNeXt_Tiny(ConvNeXt):
    '''ConvNeXt-Tiny (depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]).'''

    variant = 'tiny'

    def __init__(self, **kwargs):
        super().__init__(variant='tiny', **kwargs)


class ConvNeXt_Small(ConvNeXt):
    '''ConvNeXt-Small (depths=[3, 3, 27, 3], dims=[96, 192, 384, 768]).'''

    variant = 'small'

    def __init__(self, **kwargs):
        super().__init__(variant='small', **kwargs)


class ConvNeXt_Base(ConvNeXt):
    '''ConvNeXt-Base (depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]).'''

    variant = 'base'

    def __init__(self, **kwargs):
        super().__init__(variant='base', **kwargs)


class ConvNeXt_Large(ConvNeXt):
    '''ConvNeXt-Large (depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]).'''

    variant = 'large'

    def __init__(self, **kwargs):
        super().__init__(variant='large', **kwargs)


_STRIP_PREFIXES = ('module.', 'backbone.', 'convnext.')
_HEAD_PREFIXES = ('head.', 'classifier.')


def _strip_prefixes(key: str) -> str:
    '''Strips the prefixes left behind by DataParallel / DINOv3 training artifacts / HuggingFace
    top-level containers.'''
    while True:
        for prefix in _STRIP_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        else:
            return key


def _remap_one_key(key: str) -> Optional[str]:
    '''Maps one weight key onto the codon naming; returning None drops it (the classification head).'''
    key = _strip_prefixes(key)

    if key.startswith(_HEAD_PREFIXES):
        return None

    if key.startswith('embeddings.patch_embeddings.'):
        return 'stem.conv.' + key[len('embeddings.patch_embeddings.'):]
    if key.startswith('embeddings.convolution.'):
        return 'stem.conv.' + key[len('embeddings.convolution.'):]
    if key.startswith('embeddings.layernorm.'):
        return 'stem.norm.' + key[len('embeddings.layernorm.'):]
    if key.startswith('encoder.stages.'):
        parts = key.split('.')  # ['encoder', 'stages', i, 'layers'|'downsample'|'downsampling_layer', ...]
        idx = int(parts[2])
        tail = parts[3:]
        if tail[0] in ('downsample', 'downsampling_layer'):
            kind = 'conv' if tail[1] in ('1', 'convolution') else 'norm'
            return f'transitions.{idx - 1}.{kind}.' + '.'.join(tail[2:])
        if tail[0] == 'layers':
            rest = '.'.join(tail[2:])
            rest = rest.replace('layernorm.', 'norm.')
            rest = rest.replace('pwconv1.', 'proj_pw1.').replace('pwconv2.', 'proj_pw2.')
            rest = rest.replace('layer_scale_parameter', 'gamma')
            return f'stages.{idx}.{tail[1]}.' + rest
    if key.startswith('layernorm.'):
        return 'norm.' + key[len('layernorm.'):]

    if key.startswith('downsample_layers.'):
        parts = key.split('.')  # ['downsample_layers', i, j, ...rest]
        idx, sub = int(parts[1]), parts[2]
        rest = '.'.join(parts[3:])
        if idx == 0:
            kind = 'conv' if sub == '0' else 'norm'
            return f'stem.{kind}.{rest}'
        kind = 'conv' if sub == '1' else 'norm'
        return f'transitions.{idx - 1}.{kind}.{rest}'
    if key.startswith('stages.'):
        key = key.replace('.pwconv1.', '.proj_pw1.').replace('.pwconv2.', '.proj_pw2.')
        return key
    if key.startswith('norm.'):
        return key
    if key.startswith('features.'):
        raise NotImplementedError(
            f'the torchvision features.* layout differs from this implementation '
            f'(convolution/normalization settings do not match); convert the key '
            f'names by hand first, got: {key!r}'
        )

    return key


def remap_convnext_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    '''
    Maps the key names of common ConvNeXt weight files onto this implementation's codon naming.

    Args:
        state_dict (Dict[str, torch.Tensor]): The original weight dictionary.

    Returns:
        Dict[str, torch.Tensor]: A new dictionary with remapped key names (the input is not
            modified).
    '''
    remapped = {}
    for key, value in state_dict.items():
        new_key = _remap_one_key(key)
        if new_key is not None:
            remapped[new_key] = value
    return remapped
