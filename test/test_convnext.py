import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import gc
import shutil

from safetensors.torch import save_file
from transformers import ConvNextConfig, ConvNextModel

from codon import *
from codon.impl import (
    ConvNeXt,
    ConvNeXtBlock,
    ConvNeXtBlockTransition,
    ConvNeXt_Tiny,
    ConvNeXt_Small,
    ConvNeXt_Base,
    ConvNeXt_Large,
    DropPath,
    convnext_sizes,
    remap_convnext_keys,
)


_FAILURES = []
_CHECKS = [0]


def section(title):
    print(f'\n=== {title} ===')


def check(name, condition, detail=''):
    _CHECKS[0] += 1
    if condition:
        print(f'  [ok]   {name}')
    else:
        _FAILURES.append(name)
        print(f'  [FAIL] {name}' + (f'  <- {detail}' if detail else ''))


def close(a, b, atol=1e-5):
    return float((a - b).abs().max()) <= atol


def maxdiff(a, b):
    return f'max|diff|={float((a - b).abs().max()):.3e}'


def test_contract():
    section('1. 输出契约与形状（四个变体）')
    x = torch.randn(2, 3, 64, 64)
    for variant, size in convnext_sizes.items():
        model = ConvNeXt(variant=variant).eval()
        dim = size['dims'][-1]
        with torch.no_grad():
            feats = model.forward_features(x)

        check(f'{variant}: embed_dim == dims[-1]', model.embed_dim == dim)
        check(f'{variant}: n_blocks == 4 / n_storage_tokens == 0',
              model.n_blocks == 4 and model.n_storage_tokens == 0)
        check(f'{variant}: dict 键齐全',
              set(feats) == {'x_norm_clstoken', 'x_storage_tokens', 'x_norm_patchtokens', 'x_prenorm', 'masks'})
        check(f'{variant}: x_norm_clstoken', tuple(feats['x_norm_clstoken'].shape) == (2, dim),
              str(tuple(feats['x_norm_clstoken'].shape)))
        check(f'{variant}: x_storage_tokens 为空', tuple(feats['x_storage_tokens'].shape) == (2, 0, dim),
              str(tuple(feats['x_storage_tokens'].shape)))
        # 64x64 -> stem /4 -> 16x16 -> 三个 transition /8 -> 2x2 = 4 个 patch
        check(f'{variant}: x_norm_patchtokens', tuple(feats['x_norm_patchtokens'].shape) == (2, 4, dim),
              str(tuple(feats['x_norm_patchtokens'].shape)))
        check(f'{variant}: x_prenorm 与 patchtokens 同形',
              tuple(feats['x_prenorm'].shape) == (2, 4, dim))
        check(f'{variant}: masks 透传', feats['masks'] is None)

        del model
        gc.collect()


def test_norm_semantics():
    section('2. CLS / patch 归一化语义（池化->norm，逐 token 等价）')
    model = ConvNeXt_Tiny().eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        feats = model.forward_features(x)
        # LayerNorm 逐 token，所以 norm(cat([pool, patches])) 等价于各自单独归一化
        check('x_norm_patchtokens == norm(x_prenorm)',
              close(feats['x_norm_patchtokens'], model.norm(feats['x_prenorm'])),
              maxdiff(feats['x_norm_patchtokens'], model.norm(feats['x_prenorm'])))
        # 手工复算 CLS：先对 stage-4 特征图做全局平均池化再归一化
        feat_map = feats['x_prenorm'].permute(0, 2, 1).reshape(2, model.embed_dim, 2, 2)
        cls_manual = model.norm(feat_map.mean([-2, -1]))
        check('x_norm_clstoken == norm(mean(spatial))',
              close(feats['x_norm_clstoken'], cls_manual),
              maxdiff(feats['x_norm_clstoken'], cls_manual))
    del model


def test_training_switch():
    section('3. forward 的 is_training 开关')
    model = ConvNeXt_Tiny().eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        train_out = model(x, is_training=True)
        eval_out = model(x)
    check('is_training=True 返回字典', isinstance(train_out, dict))
    check('is_training=False 返回 CLS 张量', torch.is_tensor(eval_out) and tuple(eval_out.shape) == (2, 768))
    check('head 为 Identity，两者一致', close(eval_out, train_out['x_norm_clstoken']))
    del model


def test_intermediate_layers():
    section('4. get_intermediate_layers 各分支')
    model = ConvNeXt_Tiny().eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model.get_intermediate_layers(x, n=1)
        check('n=1: (N, HW, C)', len(out) == 1 and tuple(out[0].shape) == (2, 4, 768),
              str(tuple(out[0].shape)))

        out = model.get_intermediate_layers(x, n=1, return_class_token=True)
        patch, cls_token = out[0]
        check('n=1 + return_class_token: (patch, cls)',
              tuple(patch.shape) == (2, 4, 768) and tuple(cls_token.shape) == (2, 768))

        out = model.get_intermediate_layers(x, n=1, reshape=True)
        check('n=1 + reshape: (N, C, H, W)', tuple(out[0].shape) == (2, 768, 2, 2),
              str(tuple(out[0].shape)))

        out = model.get_intermediate_layers(x, n=1, norm=False)
        check('n=1 + norm=False: (N, HW, C)', tuple(out[0].shape) == (2, 4, 768))

        out = model.get_intermediate_layers(x, n=1, norm=False, reshape=True)
        check('n=1 + norm=False + reshape: (N, C, H, W)', tuple(out[0].shape) == (2, 768, 2, 2),
              str(tuple(out[0].shape)))

        # 早层特征图更大：stem 后 16x16(=256)，之后每过一个 transition 减半 -> 64, 16, 4
        out = model.get_intermediate_layers(x, n=[0])
        check('n=[0]: 早层网格 16x16、通道 dims[0]=96', tuple(out[0].shape) == (2, 256, 96),
              str(tuple(out[0].shape)))

        out = model.get_intermediate_layers(x, n=[0, 3])
        check('n=[0, 3]: 两层分别为 256 与 4 个 patch',
              len(out) == 2
              and tuple(out[0].shape) == (2, 256, 96) and tuple(out[1].shape) == (2, 4, 768),
              str([tuple(o.shape) for o in out]))

        out = model.get_intermediate_layers(x, n=4)
        check('n=4: 四层网格依次为 256/64/16/4',
              [tuple(o.shape) for o in out]
              == [(2, 256, 96), (2, 64, 192), (2, 16, 384), (2, 4, 768)],
              str([tuple(o.shape) for o in out]))

    resized = ConvNeXt_Tiny(patch_size=16).eval()
    with torch.no_grad():
        out = resized.get_intermediate_layers(x, n=1, reshape=True)
    # patch_size=16 -> 特征图插值到 (64/16, 64/16) = 4x4
    check('patch_size=16: 插值到 ViT 网格 (4, 4)', tuple(out[0].shape) == (2, 768, 4, 4),
          str(tuple(out[0].shape)))
    del model, resized


def test_init_and_backward():
    section('5. 初始化与反传')
    model = ConvNeXt_Tiny()
    gammas = [float(b.gamma[0].detach()) for s in model.stages for b in s if b.gamma is not None]
    check('gamma 初值 == layer_scale_init_value',
          len(gammas) == sum(convnext_sizes['tiny']['depths']) and all(abs(g - 1e-6) < 1e-12 for g in gammas))
    check('LayerNorm 权重为 1 / 偏置为 0',
          float(model.norm.weight.mean().detach()) == 1.0
          and float(model.norm.bias.abs().max().detach()) == 0.0)

    model.train()
    feats = model(torch.randn(2, 3, 64, 64), is_training=True)
    feats['x_norm_patchtokens'].sum().backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    check('所有参数的梯度都已回传', all(grads), f'{grads.count(False)} / {len(grads)} 无梯度')

    # 二次调用 init_weights 不应破坏形状与归一化参数
    model.init_weights()
    check('重复 init_weights 后 LayerNorm 仍为 1 / 0',
          float(model.norm.weight.mean().detach()) == 1.0
          and float(model.norm.bias.abs().max().detach()) == 0.0)
    del model


def test_key_remap():
    section('6. 键名映射（单元）')
    dino = {
        'downsample_layers.0.0.weight': 'stem.conv.weight',
        'downsample_layers.0.1.bias': 'stem.norm.bias',
        'downsample_layers.2.0.weight': 'transitions.1.norm.weight',
        'downsample_layers.2.1.weight': 'transitions.1.conv.weight',
        'downsample_layers.3.1.bias': 'transitions.2.conv.bias',
        'stages.1.2.pwconv1.weight': 'stages.1.2.proj_pw1.weight',
        'stages.1.2.pwconv2.bias': 'stages.1.2.proj_pw2.bias',
        'stages.0.0.dwconv.weight': 'stages.0.0.dwconv.weight',
        'stages.0.0.gamma': 'stages.0.0.gamma',
        'norm.weight': 'norm.weight',
        'backbone.stages.0.1.norm.bias': 'stages.0.1.norm.bias',
        'module.backbone.norm.bias': 'norm.bias',
    }
    got = remap_convnext_keys({k: torch.zeros(1) for k in dino})
    for src, dst in dino.items():
        check(f'DINOv3/timm: {src} -> {dst}', dst in got)

    hf = {
        'convnext.embeddings.patch_embeddings.weight': 'stem.conv.weight',
        'convnext.embeddings.patch_embeddings.bias': 'stem.conv.bias',
        'convnext.embeddings.layernorm.bias': 'stem.norm.bias',
        'convnext.encoder.stages.1.downsampling_layer.0.weight': 'transitions.0.norm.weight',
        'convnext.encoder.stages.1.downsampling_layer.1.weight': 'transitions.0.conv.weight',
        'convnext.encoder.stages.2.layers.3.pwconv1.weight': 'stages.2.3.proj_pw1.weight',
        'convnext.encoder.stages.2.layers.3.layernorm.weight': 'stages.2.3.norm.weight',
        'convnext.encoder.stages.2.layers.3.layer_scale_parameter': 'stages.2.3.gamma',
        'convnext.layernorm.weight': 'norm.weight',
        # transformers v4 的旧名字
        'convnext.convnext.embeddings.convolution.weight': 'stem.conv.weight',
        'convnext.convnext.encoder.stages.1.downsample.0.weight': 'transitions.0.norm.weight',
        'convnext.convnext.encoder.stages.1.downsample.1.bias': 'transitions.0.conv.bias',
    }
    got = remap_convnext_keys({k: torch.zeros(1) for k in hf})
    for src, dst in hf.items():
        check(f'HuggingFace: {src} -> {dst}', dst in got)

    dropped = remap_convnext_keys({'head.weight': torch.zeros(1), 'classifier.bias': torch.zeros(1)})
    check('分类头键被丢弃', dropped == {})

    try:
        remap_convnext_keys({'features.0.0.weight': torch.zeros(1)})
        check('torchvision features.* 显式报错', False, '没有抛异常')
    except NotImplementedError:
        check('torchvision features.* 显式报错', True)


def test_parity_with_transformers():
    section('7. 与 transformers 官方实现数值对拍')
    variant = 'tiny'
    size = convnext_sizes[variant]
    cfg = ConvNextConfig(
        depths=size['depths'],
        hidden_sizes=size['dims'],
        layer_norm_eps=1e-6,
        layer_scale_init_value=1e-6,
        drop_path_rate=0.0,
    )
    ref = ConvNextModel(cfg).eval()
    model = ConvNeXt(variant=variant).eval()

    # 不用 tempfile：mkdtemp/TemporaryDirectory 以 0o700 建目录，本沙箱会拒绝写入该目录
    tmp = os.path.join(project_root, '_convnext_tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    try:
        # 走真实的本地加载路径（safetensors + remap + strict 校验）
        safe_path = os.path.join(tmp, 'hf_convnext.safetensors')
        save_file(ref.state_dict(), safe_path)
        model.load_pretrained(safe_path, strict=True)
        check('safetensors: 官方权重按键名映射后严格加载成功', True)

        # strict 校验确实生效：错配的权重应当报错
        try:
            ConvNeXt(variant='small').load_pretrained(safe_path, strict=True)
            check('strict: 变体不匹配时报错', False, '没有抛异常')
        except RuntimeError:
            check('strict: 变体不匹配时报错', True)

        torch_path = os.path.join(tmp, 'hf_convnext.pth')
        torch.save({'state_dict': ref.state_dict()}, torch_path)
        ConvNeXt(variant=variant).eval().load_pretrained(torch_path, strict=True)
        check('.pth + state_dict 包装: 加载成功', True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        ref_out = ref(x)
        feats = model.forward_features(x)

    n, c, h, w = ref_out.last_hidden_state.shape
    feat_map = feats['x_prenorm'].permute(0, 2, 1).reshape(n, c, h, w)
    check('stage-4 特征图 == HF last_hidden_state',
          close(feat_map, ref_out.last_hidden_state),
          maxdiff(feat_map, ref_out.last_hidden_state))
    check('x_norm_clstoken == HF pooler_output（池化->norm 语义一致）',
          close(feats['x_norm_clstoken'], ref_out.pooler_output),
          maxdiff(feats['x_norm_clstoken'], ref_out.pooler_output))

    # 不同输入尺寸再确认一次
    x = torch.randn(1, 3, 96, 96)
    with torch.no_grad():
        ref_out = ref(x)
        feats = model.forward_features(x)
    n, c, h, w = ref_out.last_hidden_state.shape
    feat_map = feats['x_prenorm'].permute(0, 2, 1).reshape(n, c, h, w)
    check('96x96 输入下特征图一致', close(feat_map, ref_out.last_hidden_state),
          maxdiff(feat_map, ref_out.last_hidden_state))

    del ref, model
    gc.collect()


def test_helpers():
    section('8. 子类与 DropPath')
    check('ConvNeXt_Tiny 变体固定为 tiny', ConvNeXt_Tiny.variant == 'tiny')
    check('四个子类都指向对应变体',
          [ConvNeXt_Tiny.variant, ConvNeXt_Small.variant, ConvNeXt_Base.variant, ConvNeXt_Large.variant]
          == ['tiny', 'small', 'base', 'large'])

    tiny = ConvNeXt_Tiny()
    check('子类实例的 variant 正确', tiny.variant == 'tiny' and tiny.embed_dim == 768)
    del tiny

    # depths/dims 显式覆盖 variant 表
    custom = ConvNeXt(depths=[1, 1, 1, 1], dims=[16, 32, 64, 128])
    check('显式 depths/dims 生效',
          custom.depths == [1, 1, 1, 1] and custom.embed_dim == 128)
    with torch.no_grad():
        out = custom.forward_features(torch.randn(1, 3, 32, 32))
    check('自定义规格前向可用', tuple(out['x_norm_patchtokens'].shape) == (1, 1, 128),
          str(tuple(out['x_norm_patchtokens'].shape)))

    # 变体名错误 / stage 数错误 都要显式报错
    try:
        ConvNeXt(variant='huge')
        check('未知 variant 报错', False, '没有抛异常')
    except NotImplementedError:
        check('未知 variant 报错', True)
    try:
        ConvNeXt(depths=[1, 1, 1], dims=[16, 32, 64])
        check('非 4 stage 报错', False, '没有抛异常')
    except ValueError:
        check('非 4 stage 报错', True)

    drop = DropPath(0.5)
    drop.train()
    x = torch.ones(1000, 4, 2, 2)
    kept = drop(x)
    check('DropPath 训练态按样本置零',
          bool((kept == 0).any()) and float(kept.mean()) > 0)
    drop.eval()
    check('DropPath 推理态直通', torch.equal(drop(x), x))
    check('drop_prob=0 时等价于 Identity', torch.equal(DropPath(0.0)(x), x))

    check('ConvNeXtBlockTransition 两种顺序',
          ConvNeXtBlockTransition(4, 8, 4, 4, pre_norm=False).pre_norm is False
          and ConvNeXtBlockTransition(4, 8, 2, 2, pre_norm=True).pre_norm is True)
    stem = ConvNeXtBlockTransition(3, 16, kernel_size=4, stride=4, pre_norm=False)
    with torch.no_grad():
        out = stem(torch.randn(1, 3, 32, 32))
    check('stem 形状 32x32 -> 8x8', tuple(out.shape) == (1, 16, 8, 8), str(tuple(out.shape)))
    check('ConvNeXtBlock 保形',
          tuple(ConvNeXtBlock(16)(torch.randn(1, 16, 8, 8)).shape) == (1, 16, 8, 8))


def test_serialization():
    section('9. state_dict 唯一性与本地保存/加载往返')
    model = ConvNeXt_Tiny().eval()
    keys = list(model.state_dict().keys())
    # 回归保护：self.norm 曾同时以 norm.* 和 norms.3.* 注册，导致严格加载要求两份重复键
    check('state_dict 不含重复注册的 norms.* 键', not any(k.startswith('norms.') for k in keys))
    check('最终归一化只以 norm.* 出现', 'norm.weight' in keys and 'norm.bias' in keys)
    check('过渡层的 norm/conv 键齐全',
          all(k in keys for k in ('transitions.0.norm.weight', 'transitions.0.conv.weight',
                                  'transitions.2.norm.bias', 'transitions.2.conv.bias')))
    check('Layer Scale 键沿用 gamma', 'stages.0.0.gamma' in keys)

    tmp = os.path.join(project_root, '_convnext_tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        ref_out = model(x)
    try:
        for ext in ('.safetensors', '.pth'):
            path = os.path.join(tmp, f'roundtrip{ext}')
            model.save(path)
            fresh = ConvNeXt_Tiny().eval().load(path, strict=True)
            with torch.no_grad():
                out = fresh(x)
            check(f'codon save/load 往返 ({ext})', close(out, ref_out), maxdiff(out, ref_out))
            del fresh
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    del model


if __name__ == '__main__':
    test_contract()
    test_norm_semantics()
    test_training_switch()
    test_intermediate_layers()
    test_init_and_backward()
    test_key_remap()
    test_parity_with_transformers()
    test_helpers()
    test_serialization()

    print(f'\n{_CHECKS[0]} checks, {len(_FAILURES)} failed')
    if _FAILURES:
        for name in _FAILURES:
            print(f'  FAILED: {name}')
        sys.exit(1)
    print('ALL PASSED')
