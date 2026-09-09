import os
from codon import *
from torch.nn import init

from safetensors.torch import load_file, save_file

# (kernel, in_ch, expand_ch, out_ch, is_relu, use_se, stride)
_BACKEND_BLOCKS = {
    'small': [
        (3, 16, 16, 16, True, True, 2),
        (3, 16, 72, 24, True, False, 2),
        (3, 24, 88, 24, True, False, 1),
        (5, 24, 96, 40, False, True, 2),
        (5, 40, 240, 40, False, True, 1),
        (5, 40, 240, 40, False, True, 1),
        (5, 40, 120, 48, False, True, 1),
        (5, 48, 144, 48, False, True, 1),
        (5, 48, 288, 96, False, True, 2),
        (5, 96, 576, 96, False, True, 1),
        (5, 96, 576, 96, False, True, 1),
    ],
    'large': [
        (3, 16, 16, 16, True, False, 1),
        (3, 16, 64, 24, True, False, 2),
        (3, 24, 72, 24, True, False, 1),
        (5, 24, 72, 40, True, True, 2),
        (5, 40, 120, 40, True, True, 1),
        (5, 40, 120, 40, True, True, 1),
        (3, 40, 240, 80, False, False, 2),
        (3, 80, 200, 80, False, False, 1),
        (3, 80, 184, 80, False, False, 1),
        (3, 80, 184, 80, False, False, 1),
        (3, 80, 480, 112, False, True, 1),
        (3, 112, 672, 112, False, True, 1),
        (5, 112, 672, 160, False, True, 2),
        (5, 160, 672, 160, False, True, 1),
        (5, 160, 960, 160, False, True, 1),
    ],
}

# (head_in_ch, head_out_ch) fed to conv2 / proj
_BACKEND_HEAD = {
    'small': (96, 576),
    'large': (160, 960),
}

# conv2.weight (out, in, 1, 1) used to tell Small from Large
_BACKEND_CONV2_SHAPE = {
    'small': (576, 96),
    'large': (960, 160),
}

# Number of bneck blocks per backend (fallback signature when conv2 is absent)
_BACKEND_NBLOCKS = {
    'small': 11,
    'large': 15,
}

# Remote repo shared by both backends; each file name encodes its backend.
_REMOTE_REPO = 'CodonProject/MobileNetv3'
_REMOTE_FILES = {
    'small': 'mobilenetv3_small.safetensors',
    'large': 'mobilenetv3_large.safetensors',
}


class SEModule(BasicModel):
    '''Squeeze-and-excitation block (same layout as upstream, feature extractor only).'''

    def __init__(self, in_size, reduction=4):
        super(SEModule, self).__init__()
        expand_size = max(in_size // reduction, 8)

        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid(),
        )

    def forward(self, x):
        return x * self.features(x)


class Block(BasicModel):
    '''expand + depthwise + pointwise.'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = nn.Conv2d(
            expand_size, expand_size, kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, groups=expand_size, bias=False,
        )
        self.norm2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)

        self.se = SEModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size),
            )
        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3,
                          groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size),
            )
        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3,
                          groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.norm1(self.conv1(x)))
        out = self.act2(self.norm2(self.conv2(out)))
        out = self.se(out)
        out = self.norm3(self.conv3(out))

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)


def _read_tensors(path: str):
    '''Read a checkpoint into a {key: Tensor} dict (any tensors only).

    Supports .safetensors and .pth/.pt. State-dict wrappers
    ({'state_dict': ...} / {'model': ...}) and a DataParallel 'module.' prefix
    are handled transparently. No strictness checks here.
    '''
    if path.endswith('.safetensors'):
        raw = load_file(path, device='cpu')
    elif path.endswith(('.pth', '.pt')):
        raw = torch.load(path, map_location='cpu')
        if isinstance(raw, dict):
            for wrapper in ('state_dict', 'model'):
                sub = raw.get(wrapper)
                if isinstance(sub, dict):
                    raw = sub
                    break
        if not isinstance(raw, dict):
            raise RuntimeError(f'{path} is not a valid PyTorch weight file (expected a dict)')
    else:
        raise ValueError(
            f'unsupported weight format (only .safetensors / .pth / .pt): {path!r}')

    tensors = {}
    for key, val in raw.items():
        if not isinstance(val, torch.Tensor):
            continue  # skip non-weight entries such as epoch / optimizer
        if key.startswith('module.'):
            key = key[len('module.'):]  # strip DataParallel prefix
        tensors[key] = val
    return tensors


def detect_backend(path: str) -> str:
    '''Return 'small' or 'large' for the backend stored in a checkpoint file.'''
    tensors = _read_tensors(path)

    conv2 = tensors.get('conv2.weight')
    if conv2 is not None:
        shape = tuple(conv2.shape[:2])
        for name, expected in _BACKEND_CONV2_SHAPE.items():
            if shape == expected:
                return name
        raise ValueError(
            f'cannot tell Small from Large: conv2.weight shape {shape} matches neither '
            f'{_BACKEND_CONV2_SHAPE}')

    n_blocks = max(
        (int(key.split('.')[1]) for key in tensors if key.startswith('bneck.') and key.split('.')[1].isdigit()),
        default=-1,
    ) + 1
    for name, expected in _BACKEND_NBLOCKS.items():
        if n_blocks == expected:
            return name
    raise ValueError(
        f'cannot tell Small from Large: {n_blocks} bneck blocks match neither '
        f'{_BACKEND_NBLOCKS}')


class MobileNetV3(BasicModel):
    '''MobileNetV3 feature extractor. Outputs a 1280-dim feature vector per image.

    Args:
        backend: 'small' or 'large'. Defaults to 'small' for a bare instance;
            prefer `MobileNetV3.from_pretrained(path)` to pick it automatically.
        act: activation used by the hard-swish blocks (default nn.Hardswish).
    '''

    __remote_resource__ = {
        'repo': _REMOTE_REPO,
        'files': list(_REMOTE_FILES.values()),
        'repo_type': 'model'
    }

    backend = None

    def __init__(self, backend: str | None = None, act=nn.Hardswish, return_features: bool = False):
        super(MobileNetV3, self).__init__()
        if backend is None:
            backend = 'small' if self.backend is None else self.backend
        if backend not in _BACKEND_BLOCKS:
            raise ValueError(f'unknown backend {backend!r}; choose from {list(_BACKEND_BLOCKS)}')
        self.backend = backend
        self.return_features = return_features

        head_in, head_out = _BACKEND_HEAD[backend]

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(16)
        self.act1 = act(inplace=True)

        def act_for(is_relu):
            return nn.ReLU if is_relu else act

        self.bneck = nn.Sequential(*[
            Block(k, i, e, o, act_for(relu), se, s)
            for (k, i, e, o, relu, se, s) in _BACKEND_BLOCKS[backend]
        ])

        self.conv2 = nn.Conv2d(head_in, head_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm2 = nn.BatchNorm2d(head_out)
        self.act2 = act(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.proj = nn.Linear(head_out, 1280, bias=False)
        self.norm3 = nn.BatchNorm1d(1280)
        self.act3 = act(inplace=True)
        self.drop = nn.Dropout(0.2)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.bneck(out)
        out = self.act2(self.norm2(self.conv2(out)))
        if self.return_features: return out
        out = self.gap(out).flatten(1)
        out = self.drop(self.act3(self.norm3(self.proj(out))))

        return out

    def _load_remote(self, local_paths: list[str], **kwargs):
        '''Load the weight file that matches this instance's backend.

        The remote repo ships both backends, so this picks the matching file
        instead of blindly taking the first download. It prefers an exact
        filename match; if that fails (e.g. a custom URL or a renamed file) it
        falls back to content-based detection. A pinned subclass that receives
        the wrong backend is rejected the same way `from_pretrained` rejects a
        conflicting checkpoint.
        '''
        backend = self.backend
        if backend is None:
            raise ValueError(
                f'{self.__class__.__name__} has no backend pinned; construct a '
                f'specific variant (MobileNetV3_Small / MobileNetV3_Large) or '
                f'pass backend= to the constructor before calling from_remote().')

        target = None
        for path in local_paths:
            if os.path.basename(path) == _REMOTE_FILES[backend]:
                target = path
                break
        if target is None:
            for path in local_paths:
                try:
                    detected = detect_backend(path)
                except ValueError:
                    continue  # not a MobileNet checkpoint; try the next file
                if detected == backend:
                    target = path
                    break
        if target is None:
            raise ValueError(
                f'no downloaded file matches backend {backend!r}: {local_paths}')

        return self.load_pretrained(target)

    def save_pretrained(self, path: str):
        '''Save the current weights by extension: safetensors or torch .pth/.pt.'''
        sd = self.state_dict()
        if path.endswith('.safetensors'):
            save_file(sd, path)
        elif path.endswith(('.pth', '.pt')):
            torch.save(sd, path)
        else:
            raise ValueError(
                f'unsupported weight format (only .safetensors / .pth / .pt): {path!r}')
        return self

    def load_pretrained(self, path: str):
        '''Load weights by extension (.safetensors / .pth / .pt).

        Strictly requires the current naming: keys must match this model exactly
        (no extra, none missing, per-tensor shapes equal). No legacy fallback.
        '''
        tensors = _read_tensors(path)

        ref = self.state_dict()
        extra = sorted(k for k in tensors if k not in ref)
        missing = sorted(k for k in ref if k not in tensors)
        if extra or missing:
            raise RuntimeError(
                f'weights do not match this model ({self.backend}): {len(extra)} extra / '
                f'{len(missing)} missing -> extra {extra[:5]}..., missing {missing[:5]}...')

        for k, v in tensors.items():
            want = tuple(ref[k].shape)
            if tuple(v.shape) != want:
                raise RuntimeError(f'shape mismatch for {k}: weights {tuple(v.shape)} vs model {want}')
            if v.dtype != ref[k].dtype:
                tensors[k] = v.to(ref[k].dtype)
        self.load_state_dict(tensors, strict=True)
        return self

    @classmethod
    def from_pretrained(cls, path: str) -> 'MobileNetV3':
        '''Infer the backend ('small'/'large') from the checkpoint and load it.

        Calling it on a pinned subclass raises if that subclass disagrees with
        the backend detected in the file.
        '''
        backend = detect_backend(path)
        pinned = cls.backend
        if pinned is not None and pinned != backend:
            raise ValueError(
                f'checkpoint at {path!r} is a {backend} model, but {cls.__name__} '
                f'is pinned to {pinned!r}')
        model = cls(backend=backend) if pinned is None else cls()
        return model.load_pretrained(path)


class MobileNetV3_Small(MobileNetV3):
    '''MobileNetV3-Small feature extractor (explicit backend, no auto-detection).'''

    backend = 'small'
    __remote_resource__ = {**MobileNetV3.__remote_resource__, 'files': [_REMOTE_FILES['small']]}

    def __init__(self, act=nn.Hardswish, return_features: bool = False):
        super(MobileNetV3_Small, self).__init__(backend='small', act=act, return_features=return_features)


class MobileNetV3_Large(MobileNetV3):
    '''MobileNetV3-Large feature extractor (explicit backend, no auto-detection).'''

    backend = 'large'
    __remote_resource__ = {**MobileNetV3.__remote_resource__, 'files': [_REMOTE_FILES['large']]}

    def __init__(self, act=nn.Hardswish, return_features: bool = False):
        super(MobileNetV3_Large, self).__init__(backend='large', act=act, return_features=return_features)
