import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import gc
import shutil

from safetensors.torch import save_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from codon import *
from codon.impl import (
    DINOv3ViT,
    DINOv3ViTBlock,
    DINOv3ViTRope,
    DINOv3ViT_Small,
    DINOv3ViT_Base,
    DINOv3ViT_Large,
    DINOv3ViT_So400m,
    DINOv3ViT_HugePlus,
    DINOv3ViT_7B,
    ViTDropPath,
    dinov3_sizes,
    remap_dinov3_keys,
)


# 官方发布的 DINOv3 ViT 权重都带 4 个 register token
NUM_REGISTERS = 4
# 本机 modelscope 快照（有则跑真实权重迁移）
MODELSCOPE_CKPT = os.path.join(
    os.path.expanduser('~'),
    '.cache', 'modelscope', 'models',
    'facebook--dinov3-vitb16-pretrain-lvd1689m', 'snapshots', 'master',
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


def tiny_model(**kwargs):
    '''一个 4 层的小 DINOv3ViT，用于快速契约测试（规格尽量贴近官方 ViT-S/16）。'''
    kwargs.setdefault('n_storage_tokens', NUM_REGISTERS)
    return DINOv3ViT(
        embed_dim=64,
        depth=4,
        num_heads=4,
        patch_size=16,
        **kwargs,
    )


def test_contract():
    section('1. 输出契约与形状')
    model = tiny_model().eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        feats = model.forward_features(x)

    dim = 64
    num_patches = (64 // 16) ** 2  # 4x4
    expected_keys = {
        'x_norm_clstoken', 'x_storage_tokens', 'x_norm_patchtokens',
        'x_norm_alltokens', 'x_prenorm', 'x_prenorm_patchtokens', 'masks',
    }
    check('键齐全', set(feats) == expected_keys, str(sorted(set(feats) ^ expected_keys)))
    check('embed_dim / n_blocks / n_storage_tokens',
          model.embed_dim == dim and model.n_blocks == 4 and model.n_storage_tokens == NUM_REGISTERS)
    check('x_norm_clstoken [N, C]', tuple(feats['x_norm_clstoken'].shape) == (2, dim),
          str(tuple(feats['x_norm_clstoken'].shape)))
    check('x_storage_tokens [N, R, C]', tuple(feats['x_storage_tokens'].shape) == (2, NUM_REGISTERS, dim),
          str(tuple(feats['x_storage_tokens'].shape)))
    check('x_norm_patchtokens [N, HW, C]', tuple(feats['x_norm_patchtokens'].shape) == (2, num_patches, dim),
          str(tuple(feats['x_norm_patchtokens'].shape)))
    check('x_norm_alltokens [N, 1+R+HW, C]',
          tuple(feats['x_norm_alltokens'].shape) == (2, 1 + NUM_REGISTERS + num_patches, dim),
          str(tuple(feats['x_norm_alltokens'].shape)))
    check('x_prenorm 与 x_norm_alltokens 同形（上游语义：完整序列）',
          tuple(feats['x_prenorm'].shape) == tuple(feats['x_norm_alltokens'].shape))
    check('x_prenorm_patchtokens 与 x_norm_patchtokens 同形',
          tuple(feats['x_prenorm_patchtokens'].shape) == (2, num_patches, dim))
    check('masks 透传', feats['masks'] is None)

    # 关闭 register token 时 storage 维度为 0，其余不变
    plain = DINOv3ViT(embed_dim=64, depth=4, num_heads=4, patch_size=16).eval()
    with torch.no_grad():
        f2 = plain.forward_features(x)
    check('n_storage_tokens=0: x_storage_tokens 为空 [N, 0, C]',
          tuple(f2['x_storage_tokens'].shape) == (2, 0, dim), str(tuple(f2['x_storage_tokens'].shape)))
    check('n_storage_tokens=0: patch 数不变',
          tuple(f2['x_norm_patchtokens'].shape) == (2, num_patches, dim))
    check('n_storage_tokens=0: 序列更短',
          tuple(f2['x_norm_alltokens'].shape) == (2, 1 + num_patches, dim),
          str(tuple(f2['x_norm_alltokens'].shape)))

    # 非方形输入
    with torch.no_grad():
        f3 = model.forward_features(torch.randn(1, 3, 96, 64))
    check('非方形输入 96x64 -> 6x4=24 patch',
          tuple(f3['x_norm_patchtokens'].shape) == (1, 24, dim),
          str(tuple(f3['x_norm_patchtokens'].shape)))

    del model, plain
    gc.collect()


def test_norm_semantics():
    section('2. 归一化语义')
    model = tiny_model().eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        feats = model.forward_features(x)
        alltokens = feats['x_norm_alltokens']
        prenorm = feats['x_prenorm']

    check('x_norm_alltokens == norm(x_prenorm)',
          close(alltokens, model.norm(prenorm)), maxdiff(alltokens, model.norm(prenorm)))
    check('x_norm_clstoken == norm(x_prenorm)[:, 0]',
          close(feats['x_norm_clstoken'], model.norm(prenorm)[:, 0]))
    check('x_norm_patchtokens == norm(x_prenorm_patchtokens)',
          close(feats['x_norm_patchtokens'], model.norm(feats['x_prenorm_patchtokens'])))
    check('x_storage_tokens == norm(x_prenorm)[:, 1:1+R]',
          close(feats['x_storage_tokens'], model.norm(prenorm)[:, 1: 1 + NUM_REGISTERS]))

    # CLS / storage 与 patch token 拼接后确实是完整序列
    with torch.no_grad():
        rebuilt = torch.cat([
            feats['x_norm_clstoken'].unsqueeze(1),
            feats['x_storage_tokens'],
            feats['x_norm_patchtokens'],
        ], dim=1)
    check('cls + storage + patch 可拼回完整序列', close(rebuilt, alltokens))
    del model


def test_rope():
    section('3. 2D 轴向 RoPE')
    rope = DINOv3ViTRope(embed_dim=64, num_heads=4, base=100.0, num_prefix_tokens=1 + NUM_REGISTERS)
    check('head_dim = 16', rope.head_dim == 16)
    check('inv_freq 长度 = head_dim/4', tuple(rope.inv_freq.shape) == (4,), str(tuple(rope.inv_freq.shape)))
    check('inv_freq 非持久化（不进 state_dict）',
          not any('inv_freq' in k for k in DINOv3ViT(embed_dim=64, depth=1, num_heads=4).state_dict()))

    cos, sin = rope.sincos(4, 4, torch.device('cpu'), batch_size=2)
    check('sincos 形状 [B, 1, P, head_dim]',
          tuple(cos.shape) == (2, 1, 16, 16) and tuple(sin.shape) == (2, 1, 16, 16),
          f'{tuple(cos.shape)} / {tuple(sin.shape)}')
    check('cos^2 + sin^2 == 1', close(cos ** 2 + sin ** 2, torch.ones_like(cos), atol=1e-6))

    # 前缀不旋转、patch 旋转
    q = torch.randn(2, 4, 1 + NUM_REGISTERS + 16, 16)
    cos1, sin1 = rope.sincos(4, 4, torch.device('cpu'), batch_size=2)
    out = rope(q, cos1, sin1)
    check('前缀 token 原样保留', torch.equal(out[:, :, : 1 + NUM_REGISTERS], q[:, :, : 1 + NUM_REGISTERS]))
    check('patch token 被旋转', not torch.allclose(out[:, :, 1 + NUM_REGISTERS:], q[:, :, 1 + NUM_REGISTERS:]))

    # 位置不同 -> 旋转结果不同；相同网格 -> 相同表
    cos2, sin2 = rope.sincos(4, 4, torch.device('cpu'), batch_size=2)
    check('同一网格 cos/sin 可复现', torch.equal(cos1, cos2) and torch.equal(sin1, sin2))
    cos3, _ = rope.sincos(8, 8, torch.device('cpu'), batch_size=2)
    check('不同网格 cos 表不同', cos3.shape[-2] == 64 and not torch.equal(cos1, cos3))

    # 旋转是保范数的
    rotated = (out[:, :, 1 + NUM_REGISTERS:] ** 2).sum(-1)
    original = (q[:, :, 1 + NUM_REGISTERS:] ** 2).sum(-1)
    check('RoPE 保范数', close(rotated, original, atol=1e-4), maxdiff(rotated, original))

    # 训练态的坐标增强会改变表；eval 态不会
    rope_aug = DINOv3ViTRope(
        embed_dim=64, num_heads=4, num_prefix_tokens=1, rescale_coords=2.0,
    )
    rope_aug.eval()
    c_eval, _ = rope_aug.sincos(4, 4, torch.device('cpu'), batch_size=1)
    rope_aug.train()
    c_train, _ = rope_aug.sincos(4, 4, torch.device('cpu'), batch_size=1)
    check('训练态坐标增强生效（eval 与 train 表不同）', not torch.equal(c_eval, c_train))
    del rope, rope_aug


def test_attention_is_bidirectional():
    section('4. 非 causal（双向注意力）')
    torch.manual_seed(0)
    model = tiny_model().eval()
    x = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        feats_full = model.forward_features(x)

    # 把右下角 patch 换成别的值，前面的 patch 表示应当随之改变（若 causal 则不会）
    x2 = x.clone()
    x2[:, :, -16:, -16:] += 3.0
    with torch.no_grad():
        feats2 = model.forward_features(x2)

    first_patch_changed = not torch.allclose(
        feats_full['x_norm_patchtokens'][:, 0], feats2['x_norm_patchtokens'][:, 0], atol=1e-6
    )
    check('改动末尾 patch 会影响第一个 patch（双向注意力生效）', first_patch_changed)

    with torch.no_grad():
        out = model.forward_features(x, output_attentions=True)
    check('output_attentions 返回 4 层权重',
          'attentions' in out and len(out['attentions']) == 4)
    attn = out['attentions'][0]
    check('注意力权重形状 [N, H, L, L]', tuple(attn.shape) == (1, 4, 21, 21), str(tuple(attn.shape)))
    check('注意力权重每行和为 1', close(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-5))
    check('注意力非因果（上三角非零）', float(attn[..., :10, 10:].abs().max()) > 0.0)
    del model


def test_forward_switch():
    section('5. forward 的 is_training 开关与多裁剪输入')
    model = tiny_model().eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        train_out = model(x, is_training=True)
        eval_out = model(x)
    check('is_training=True 返回字典', isinstance(train_out, dict))
    check('is_training=False 返回 CLS 张量 [N, C]',
          torch.is_tensor(eval_out) and tuple(eval_out.shape) == (2, 64))
    check('head 为 Identity，两者一致', close(eval_out, train_out['x_norm_clstoken']))

    # 多裁剪：不同分辨率的列表输入
    crops = [torch.randn(2, 3, 64, 64), torch.randn(2, 3, 32, 48)]
    with torch.no_grad():
        feats = model.forward_features(crops)
    check('列表输入返回等长列表', isinstance(feats, list) and len(feats) == 2)
    check('crop0 patch 数 16', tuple(feats[0]['x_norm_patchtokens'].shape) == (2, 16, 64),
          str(tuple(feats[0]['x_norm_patchtokens'].shape)))
    check('crop1 patch 数 2x3=6', tuple(feats[1]['x_norm_patchtokens'].shape) == (2, 6, 64),
          str(tuple(feats[1]['x_norm_patchtokens'].shape)))

    try:
        model(crops)
        check('列表输入 + is_training=False 显式报错', False, '没有抛异常')
    except ValueError:
        check('列表输入 + is_training=False 显式报错', True)

    # 单张图走 head
    with torch.no_grad():
        out = model(x)
    check('单张图 + is_training=False 返回 [N, C]',
          torch.is_tensor(out) and tuple(out.shape) == (2, 64), str(tuple(out.shape)))
    del model


def test_intermediate_layers():
    section('6. get_intermediate_layers 各分支')
    model = tiny_model().eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model.get_intermediate_layers(x, n=1)
        check('n=1: (N, HW, C)', len(out) == 1 and tuple(out[0].shape) == (2, 16, 64),
              str(tuple(out[0].shape)))

        out = model.get_intermediate_layers(x, n=1, return_class_token=True)
        patch, cls_token = out[0]
        check('n=1 + return_class_token: (patch, cls)',
              tuple(patch.shape) == (2, 16, 64) and tuple(cls_token.shape) == (2, 64))

        out = model.get_intermediate_layers(x, n=1, return_extra_tokens=True)
        patch, extra = out[0]
        check('n=1 + return_extra_tokens: (patch, storage)',
              tuple(patch.shape) == (2, 16, 64) and tuple(extra.shape) == (2, NUM_REGISTERS, 64))

        out = model.get_intermediate_layers(x, n=1, return_class_token=True, return_extra_tokens=True)
        check('n=1 + 两个 flag: 三元组', len(out[0]) == 3)

        out = model.get_intermediate_layers(x, n=1, reshape=True)
        check('n=1 + reshape: (N, C, H, W)', tuple(out[0].shape) == (2, 64, 4, 4),
              str(tuple(out[0].shape)))

        out = model.get_intermediate_layers(x, n=1, norm=False)
        check('n=1 + norm=False', tuple(out[0].shape) == (2, 16, 64))

        out = model.get_intermediate_layers(x, n=1, norm=False, reshape=True)
        check('n=1 + norm=False + reshape', tuple(out[0].shape) == (2, 64, 4, 4))

        out = model.get_intermediate_layers(x, n=4)
        check('n=4: 四层', len(out) == 4 and all(tuple(o.shape) == (2, 16, 64) for o in out))

        out = model.get_intermediate_layers(x, n=[0, 3])
        check('n=[0, 3]: 指定层号', len(out) == 2 and tuple(out[0].shape) == (2, 16, 64))

        # 不同层的输出应当不同
        out = model.get_intermediate_layers(x, n=[0, 3])
        check('第 0 层与第 3 层输出不同', not torch.allclose(out[0], out[1]))

    # 中间层 norm 语义
    with torch.no_grad():
        raw = model.get_intermediate_layers(x, n=1, norm=False)[0]
        normed = model.get_intermediate_layers(x, n=1, norm=True)[0]
    check('norm=True == norm(norm=False)', close(normed, model.norm(raw)), maxdiff(normed, model.norm(raw)))
    del model


def test_init_and_backward():
    section('7. 初始化与反传')
    model = tiny_model()
    gammas1 = [float(b.gamma1[0].detach()) for b in model.blocks]
    gammas2 = [float(b.gamma2[0].detach()) for b in model.blocks]
    check('gamma1 / gamma2 初值 == layerscale_init',
          len(gammas1) == 4 and all(abs(g - 1.0) < 1e-12 for g in gammas1 + gammas2))
    check('LayerNorm 权重为 1 / 偏置为 0',
          float(model.norm.weight.mean().detach()) == 1.0
          and float(model.norm.bias.abs().max().detach()) == 0.0)
    check('mask_token 初始化为 0', float(model.mask_token.abs().max().detach()) == 0.0)
    check('cls_token / storage_tokens 非零初始化',
          float(model.cls_token.abs().max().detach()) > 0
          and float(model.storage_tokens.abs().max().detach()) > 0)

    model.train()
    feats = model(torch.randn(2, 3, 64, 64), is_training=True)
    feats['x_norm_patchtokens'].sum().backward()
    # mask_token 只在传 masks 时参与计算，这里先排除
    grads = [p.grad is not None for n, p in model.named_parameters()
             if p.requires_grad and n != 'mask_token']
    check('所有（除 mask_token 外）参数的梯度都已回传', all(grads),
          f'{grads.count(False)} / {len(grads)} 无梯度')

    # mask_token 只在有 masks 时参与计算，单独验证它的梯度
    model.zero_grad()
    masks = torch.zeros(2, 16, dtype=torch.bool)
    masks[:, :2] = True
    feats = model(torch.randn(2, 3, 64, 64), is_training=True, masks=masks)
    feats['x_norm_patchtokens'].sum().backward()
    check('mask_token 有梯度', model.mask_token.grad is not None
          and float(model.mask_token.grad.abs().max()) > 0)

    model.init_weights()
    check('重复 init_weights 后 LayerNorm 仍为 1 / 0',
          float(model.norm.weight.mean().detach()) == 1.0
          and float(model.norm.bias.abs().max().detach()) == 0.0)
    del model


def test_mask_token_path():
    section('8. MAE mask token 路径')
    model = tiny_model().eval()
    x = torch.randn(2, 3, 64, 64)
    masks = torch.zeros(2, 16, dtype=torch.bool)
    masks[:, :4] = True
    with torch.no_grad():
        masked = model.forward_features(x, masks=masks)
        plain = model.forward_features(x)
        tokens = model.prepare_tokens_with_masks(x, masks)
        patch_embed_ref = model.prepare_tokens_with_masks(x)

    check('masks 透传', masked['masks'] is masks)
    # patch embedding 之后，被 mask 的 4 个位置应当都等于 mask_token
    mt = model.mask_token[0, 0]
    check('mask 位置在 patch_embed 后被替换为 mask_token',
          close(tokens[:, 5:9], mt.expand(2, 4, -1), atol=1e-6),
          maxdiff(tokens[:, 5:9], mt.expand(2, 4, -1)))
    check('未 mask 位置保留 patch 嵌入',
          close(tokens[:, 1 + 4 + 4:], patch_embed_ref[:, 1 + 4 + 4:], atol=1e-6)
          and close(tokens[:, 1:5], patch_embed_ref[:, 1:5], atol=1e-6),
          maxdiff(tokens[:, 1 + 4 + 4:], patch_embed_ref[:, 1 + 4 + 4:]))
    # 注意：mask 位置虽然输入相同，但 RoPE 是位置相关的，因此层输出并不相同
    check('mask 位置经 RoPE 后不再相同（位置相关）',
          not torch.allclose(masked['x_prenorm_patchtokens'][:, 0], masked['x_prenorm_patchtokens'][:, 1]))
    del model


def test_key_remap():
    section('9. 键名映射（单元）')
    hf = {
        'embeddings.patch_embeddings.weight': 'patch_embed.weight',
        'embeddings.patch_embeddings.bias': 'patch_embed.bias',
        'embeddings.cls_token': 'cls_token',
        'embeddings.register_tokens': 'storage_tokens',
        'layer.0.attention.q_proj.weight': 'blocks.0.q_proj.weight',
        'layer.0.attention.k_proj.weight': 'blocks.0.k_proj.weight',
        'layer.0.attention.v_proj.bias': 'blocks.0.v_proj.bias',
        'layer.0.attention.o_proj.weight': 'blocks.0.o_proj.weight',
        'layer.3.norm1.weight': 'blocks.3.norm1.weight',
        'layer.3.norm2.bias': 'blocks.3.norm2.bias',
        'layer.11.mlp.up_proj.weight': 'blocks.11.up_proj.weight',
        'layer.11.mlp.down_proj.bias': 'blocks.11.down_proj.bias',
        'layer.11.layer_scale1.lambda1': 'blocks.11.gamma1',
        'layer.11.layer_scale2.lambda1': 'blocks.11.gamma2',
        'norm.weight': 'norm.weight',
        'norm.bias': 'norm.bias',
        # 容器前缀
        'model.embeddings.patch_embeddings.weight': 'patch_embed.weight',
        'backbone.layer.0.norm1.weight': 'blocks.0.norm1.weight',
        'module.model.norm.bias': 'norm.bias',
    }
    got = remap_dinov3_keys({k: torch.zeros(1) for k in hf})
    for src, dst in hf.items():
        check(f'HuggingFace: {src} -> {dst}', dst in got)

    dropped = remap_dinov3_keys({
        'embeddings.mask_token': torch.zeros(1),
        'head.weight': torch.zeros(1),
        'classifier.bias': torch.zeros(1),
        # RoPE 逆频率表是动态重建的非持久化 buffer，不是权重
        'rope_embeddings.inv_freq': torch.zeros(1),
        'rope.inv_freq': torch.zeros(1),
    })
    check('mask_token / 分类头 / rope buffer 键被丢弃', dropped == {}, str(dropped))

    # codon 命名的输入应当幂等
    codon_named = remap_dinov3_keys({'blocks.0.gamma1': torch.zeros(1), 'norm.weight': torch.zeros(1)})
    check('codon 命名幂等', set(codon_named) == {'blocks.0.gamma1', 'norm.weight'})

    check('dinov3_sizes 覆盖 6 个变体', len(dinov3_sizes) == 6, str(sorted(dinov3_sizes)))
    check('vitb16 规格与 config 一致',
          dinov3_sizes['vitb16'] == dict(embed_dim=768, depth=12, num_heads=12, ffn_ratio=4.0))


def test_from_remote_config():
    section('9b. from_remote 从 config 推断结构与精度')
    origin_dir = MODELSCOPE_CKPT
    converted_dir = os.path.join(project_root, 'dinov3-vitb16-fp16')
    if not os.path.exists(os.path.join(origin_dir, 'config.json')):
        print(f'  [skip] 未找到 {origin_dir}/config.json')
        return

    from codon.builtin import repo as repo_mod

    # 用一个可控的桩替换 Repo，直接把指定目录当成「已下载好的缓存」
    holder = {'dir': origin_dir}

    class _StubRepo:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def download_configured_files(self, platform=None):
            d = holder['dir']
            return [os.path.join(d, 'config.json'), os.path.join(d, 'model.safetensors')]

    original = repo_mod.Repo
    repo_mod.Repo = _StubRepo
    try:
        # ---- A. Meta 原始权重（fp32，无 dtype 字段）----
        model = DINOv3ViT_Base.from_remote()
        check('from_remote(原始): 从 config 推断 num_register_tokens=4',
              model.n_storage_tokens == NUM_REGISTERS, str(model.n_storage_tokens))
        check('from_remote(原始): 从 config 推断规格',
              model.embed_dim == 768 and model.n_blocks == 12 and model.num_heads == 12)
        check('from_remote(原始): 从 config 取到 rope_rescale_coords=2.0',
              model.rope.rescale_coords == 2.0, str(model.rope.rescale_coords))
        check('from_remote(原始): 从 config 取到 layer_norm_eps=1e-5', model.norm.eps == 1e-5)
        check('from_remote(原始): dtype 保持 fp32',
              next(model.parameters()).dtype == torch.float32)
        check('from_remote: 用的是 codon 转好的仓库',
              DINOv3ViT_Base.__remote_resource__['repo'] == 'CodonProject/DINOv3-ViT-Base',
              DINOv3ViT_Base.__remote_resource__['repo'])
        check('from_remote: 保留了 Meta 原始仓库作为来源',
              DINOv3ViT_Base.__origin_resource__['repo'] == 'facebook/dinov3-vitb16-pretrain-lvd1689m')
        del model

        # 用户显式传参优先于 config
        try:
            DINOv3ViT_Base.from_remote(n_storage_tokens=0)
            check('from_remote: 显式覆盖与权重不符时 strict 报错', False, '没有抛异常')
        except RuntimeError:
            check('from_remote: 显式覆盖与权重不符时 strict 报错', True)

        # ---- B. codon 转好的 fp16 产物 ----
        if os.path.exists(os.path.join(converted_dir, 'config.json')):
            holder['dir'] = converted_dir
            model_h = DINOv3ViT_Base.from_remote()
            check('from_remote(codon fp16): config 里 dtype=float16 -> 模型为 fp16',
                  next(model_h.parameters()).dtype == torch.float16,
                  str(next(model_h.parameters()).dtype))
            check('from_remote(codon fp16): 结构仍正确',
                  model_h.embed_dim == 768 and model_h.n_storage_tokens == NUM_REGISTERS)
            with torch.no_grad():
                out = model_h(torch.randn(1, 3, 224, 224).half())
            check('from_remote(codon fp16): 前向可用', tuple(out.shape) == (1, 768) and out.dtype == torch.float16)

            # 显式指定 fp32 -> 权重升回 fp32 计算
            model_f = DINOv3ViT_Base.from_remote(dtype=torch.float32)
            check('from_remote(dtype=float32): fp16 权重升回 fp32',
                  next(model_f.parameters()).dtype == torch.float32)
            with torch.no_grad():
                out32 = model_f(torch.randn(1, 3, 224, 224))
            check('from_remote(dtype=float32): 前向可用且为 fp32',
                  out32.dtype == torch.float32)
            del model_h, model_f
        else:
            print(f'  [skip] 未找到 {converted_dir}（先跑 dev/convert_dinov3_fp16.py）')
    finally:
        repo_mod.Repo = original

    gc.collect()


def test_dtype_handling():
    section('9c. 精度处理（fp16 权重 / dtype 参数）')
    from codon.impl.dinov3_vit import _parse_config_dtype

    check('_parse_config_dtype 识别 float16/fp16/half',
          _parse_config_dtype({'dtype': 'float16'}) == torch.float16
          and _parse_config_dtype({'torch_dtype': 'fp16'}) == torch.float16
          and _parse_config_dtype({'weight_dtype': 'half'}) == torch.float16)
    check('_parse_config_dtype 识别 bfloat16', _parse_config_dtype({'dtype': 'bf16'}) == torch.bfloat16)
    check('_parse_config_dtype 无字段时返回 None', _parse_config_dtype({}) is None)

    model = tiny_model()
    tmp = os.path.join(project_root, '_dinov3_tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    half_path = os.path.join(tmp, 'half.safetensors')
    try:
        # 写一份 fp16 权重
        save_file({k: v.half() for k, v in model.state_dict().items()}, half_path)

        # 1) 不指定 dtype：权重被转回模型当前的 fp32
        m1 = tiny_model().eval()
        m1.load_pretrained(half_path, strict=True)
        check('fp16 权重 + fp32 模型：自动升回 fp32',
              next(m1.parameters()).dtype == torch.float32)

        # 2) 指定 dtype=float16：严格校验精度
        m2 = tiny_model().eval().half()
        m2.load_pretrained(half_path, strict=True, dtype=torch.float16)
        check('fp16 权重 + dtype=float16：加载成功',
              next(m2.parameters()).dtype == torch.float16)

        # 3) dtype 是「目标精度」而不是「校验精度」：fp16 权重可以升回 fp32
        m3 = tiny_model().eval()
        m3.load_pretrained(half_path, strict=True, dtype=torch.float32)
        check('fp16 权重 + dtype=float32：升回 fp32 加载成功',
              next(m3.parameters()).dtype == torch.float32)

        # 非浮点目标精度应当显式报错（不会被静默改精度）
        try:
            tiny_model().eval().load_pretrained(half_path, strict=True, dtype=torch.int32)
            check('目标精度非浮点时报错', False, '没有抛异常')
        except RuntimeError:
            check('目标精度非浮点时报错', True)
        del m3

        # 3) fp16 全链路前向可用（验证 RoPE 的 cos/sin 会跟随 q/k 精度）
        x32 = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out_fp32 = m1(x32)
            out_fp16 = m2(x32.half())
        check('fp16 前向输出为 fp16', out_fp16.dtype == torch.float16)
        diff = float((out_fp16.float() - out_fp32).abs().max())
        check('fp16 前向与 fp32 前向接近（<1e-1）', diff < 1e-1, f'max|diff|={diff:.3e}')

        # fp16 下 attention 的中间张量不应被 RoPE 意外提升成 fp32
        with torch.no_grad():
            q = m2.blocks[0].q_proj(m2.blocks[0].norm1(m2.prepare_tokens_with_masks(x32.half())))
        check('fp16 下投影输出保持 fp16', q.dtype == torch.float16)

        del m1, m2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    del model
    gc.collect()


def test_converted_artifacts():
    section('9d. 转换产物目录（dinov3-vitb16-fp16）')
    out_dir = os.path.join(project_root, 'dinov3-vitb16-fp16')
    weight_path = os.path.join(out_dir, 'model.safetensors')
    if not os.path.exists(weight_path):
        print(f'  [skip] 未找到 {weight_path}（先跑 dev/convert_dinov3_fp16.py）')
        return

    import json
    from safetensors import safe_open
    from codon.impl.dinov3_vit import _parse_config_dtype

    with open(os.path.join(out_dir, 'config.json'), 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    check('config: key_layout 为 codon', cfg.get('key_layout') == 'codon')
    check('config: dtype 为 float16', cfg.get('dtype') == 'float16')
    check('config: 指向 CodonProject/DINOv3-ViT-Base',
          cfg.get('dinov3_repo') == 'CodonProject/DINOv3-ViT-Base', str(cfg.get('dinov3_repo')))
    check('config: num_register_tokens == 4', cfg.get('num_register_tokens') == NUM_REGISTERS)
    check('config: _parse_config_dtype 能解析出 fp16',
          _parse_config_dtype(cfg) == torch.float16)
    check('preprocessor_config.json 存在',
          os.path.exists(os.path.join(out_dir, 'preprocessor_config.json')))
    check('README.md 存在且提到 DINOv3 License',
          os.path.exists(os.path.join(out_dir, 'README.md'))
          and 'dinov3-license' in open(os.path.join(out_dir, 'README.md'), encoding='utf-8').read())

    # 键名与精度
    with safe_open(weight_path, framework='pt') as f:
        keys = list(f.keys())
        dtypes = {f.get_slice(k).get_dtype() for k in keys}
        shapes = {k: tuple(f.get_slice(k).get_shape()) for k in keys}
    check('权重键名是 codon 布局',
          'blocks.0.q_proj.weight' in keys and 'patch_embed.weight' in keys
          and 'blocks.0.gamma1' in keys and 'storage_tokens' in keys)
    check('不含上游 layer.* / lambda1 键',
          not any(k.startswith('layer.') or 'lambda1' in k for k in keys))
    # mask_token 是本实现的 MAE 占位参数（原始权重里没有，导出时补成全 0）；
    # RoPE 的 inv_freq 是动态重建的非持久化 buffer，不应落盘
    check('不含 rope inv_freq（非持久化 buffer）', not any('inv_freq' in k for k in keys))
    check('含 mask_token 占位（全 0）',
          'mask_token' in keys and shapes['mask_token'] == (1, 1, 768))
    check(f'权重全为 F16（{len(keys)} 个张量）', dtypes == {'F16'}, str(dtypes))
    check('层数与尺寸正确',
          shapes['blocks.11.q_proj.weight'] == (768, 768)
          and shapes['blocks.0.up_proj.weight'] == (3072, 768)
          and shapes['storage_tokens'] == (1, NUM_REGISTERS, 768),
          str({k: shapes[k] for k in ('blocks.11.q_proj.weight', 'blocks.0.up_proj.weight', 'storage_tokens')}))

    # 按 config 建模型并把产物加载回去，与原始 fp32 权重对比
    model = DINOv3ViT(
        embed_dim=cfg['hidden_size'],
        depth=cfg['num_hidden_layers'],
        num_heads=cfg['num_attention_heads'],
        ffn_ratio=cfg['intermediate_size'] / cfg['hidden_size'],
        patch_size=cfg['patch_size'],
        n_storage_tokens=cfg['num_register_tokens'],
        norm_eps=cfg['layer_norm_eps'],
        rope_base=cfg['rope_theta'],
        rope_rescale_coords=cfg['pos_embed_rescale'],
        key_bias=cfg['key_bias'],
    ).eval().half()
    model.load_pretrained(weight_path, strict=True, dtype=torch.float16)
    check('按 config 建模型 + fp16 产物严格加载成功', True)

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x.half())
    check('fp16 产物前向可用 [N, C]', tuple(out.shape) == (1, 768), str(tuple(out.shape)))
    check('输出为 fp16', out.dtype == torch.float16)

    # 与原始 Meta 权重（fp32）对比：fp16 量化误差应在合理范围内
    if os.path.exists(os.path.join(MODELSCOPE_CKPT, 'model.safetensors')):
        ref_model = DINOv3ViT(
            variant='vitb16', n_storage_tokens=NUM_REGISTERS,
            rope_base=100.0, rope_rescale_coords=2.0,
        ).eval()
        ref_model.load_pretrained(os.path.join(MODELSCOPE_CKPT, 'model.safetensors'), strict=True)
        with torch.no_grad():
            want = ref_model(x)
        diff = float((out.float() - want).abs().max())
        check('fp16 产物 vs 原始 fp32 权重（max|diff| < 0.5）', diff < 0.5, f'max|diff|={diff:.3e}')
        del ref_model

    del model
    gc.collect()


def test_helpers():
    section('10. 子类与辅助模块')
    check('DINOv3ViT_Small 变体固定', DINOv3ViT_Small.variant == 'vits16')
    check('六个子类都指向对应变体',
          [c.variant for c in (DINOv3ViT_Small, DINOv3ViT_Base, DINOv3ViT_Large,
                               DINOv3ViT_So400m, DINOv3ViT_HugePlus, DINOv3ViT_7B)]
          == ['vits16', 'vitb16', 'vitl16', 'vitso400m', 'vith16plus', 'vit7b16'])
    check('子类变体规格生效', DINOv3ViT_Large().embed_dim == 1024 and DINOv3ViT_Large().n_blocks == 24)

    try:
        DINOv3ViT(variant='huge')
        check('未知 variant 报错', False, '没有抛异常')
    except NotImplementedError:
        check('未知 variant 报错', True)

    drop = ViTDropPath(0.5)
    drop.train()
    x = torch.ones(1000, 8)
    kept = drop(x)
    check('DropPath 训练态按样本置零', bool((kept == 0).any()) and float(kept.mean()) > 0)
    drop.eval()
    check('DropPath 推理态直通', torch.equal(drop(x), x))
    check('drop_prob=0 时等价于 Identity', torch.equal(ViTDropPath(0.0)(x), x))

    # drop_path_rate>0 时每个 block 的速率线性递增
    dp = DINOv3ViT(embed_dim=32, depth=4, num_heads=2, drop_path_rate=0.4)
    rates = [getattr(b.drop_path1, 'drop_prob', None) for b in dp.blocks]
    check('stochastic depth 线性递增',
          rates[0] is None and abs(rates[-1] - 0.4) < 1e-9
          and [r for r in rates if r is not None] == sorted(r for r in rates if r is not None),
          str(rates))
    check('无随机深度时 drop_path 为 Identity',
          isinstance(DINOv3ViT(embed_dim=32, depth=2, num_heads=2).blocks[0].drop_path1, nn.Identity))

    block = DINOv3ViTBlock(embed_dim=32, num_heads=4)
    with torch.no_grad():
        out = block(torch.randn(2, 10, 32))
    check('DINOv3ViTBlock 保形', tuple(out.shape) == (2, 10, 32), str(tuple(out.shape)))

    check('k_proj 默认无 bias', DINOv3ViT(embed_dim=32, depth=1, num_heads=2).blocks[0].k_proj.bias is None)
    check('q/v/o/up/down 默认有 bias',
          all(getattr(DINOv3ViT(embed_dim=32, depth=1, num_heads=2).blocks[0], n).bias is not None
              for n in ('q_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj')))
    del dp, block


def test_serialization():
    section('11. state_dict 唯一性与本地保存/加载往返')
    model = tiny_model().eval()
    keys = list(model.state_dict().keys())
    check('state_dict 不含 inv_freq（非持久化 buffer）',
          not any('inv_freq' in k for k in keys))
    check('codon 命名键齐全',
          all(k in keys for k in (
              'patch_embed.weight', 'patch_embed.bias', 'cls_token', 'storage_tokens',
              'blocks.0.q_proj.weight', 'blocks.0.k_proj.weight', 'blocks.0.gamma1',
              'blocks.0.gamma2', 'blocks.0.up_proj.weight', 'blocks.0.down_proj.weight',
              'blocks.0.norm1.weight', 'norm.weight', 'norm.bias',
          )), str([k for k in keys][:12]))
    check('不含上游的 lambda1 命名', not any('lambda1' in k for k in keys))

    tmp = os.path.join(project_root, '_dinov3_tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        ref_out = model(x)
    try:
        for ext in ('.safetensors', '.pth'):
            path = os.path.join(tmp, f'roundtrip{ext}')
            model.save(path)
            fresh = tiny_model().eval().load(path, strict=True)
            with torch.no_grad():
                out = fresh(x)
            check(f'codon save/load 往返 ({ext})', close(out, ref_out), maxdiff(out, ref_out))
            del fresh
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    del model


def test_parity_with_transformers():
    section('12. 与 transformers 官方实现数值对拍')

    def build_pair(seed=0):
        torch.manual_seed(seed)
        size = dinov3_sizes['vits16']
        cfg = DINOv3ViTConfig(
            hidden_size=size['embed_dim'],
            num_hidden_layers=size['depth'],
            num_attention_heads=size['num_heads'],
            intermediate_size=int(size['embed_dim'] * size['ffn_ratio']),
            layer_norm_eps=1e-5,
            rope_theta=100.0,
            image_size=64,
            patch_size=16,
            num_register_tokens=NUM_REGISTERS,
            query_bias=True,
            key_bias=False,
            value_bias=True,
            proj_bias=True,
            mlp_bias=True,
            layerscale_value=1.0,
            drop_path_rate=0.0,
            pos_embed_rescale=None,
        )
        ref = DINOv3ViTModel(cfg).eval()
        model = DINOv3ViT(
            variant='vits16',
            n_storage_tokens=NUM_REGISTERS,
            rope_base=100.0,
            rope_rescale_coords=None,
        ).eval()
        return ref, model

    ref, model = build_pair()

    tmp = os.path.join(project_root, '_dinov3_tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    try:
        # 走真实的本地加载路径（safetensors + remap + strict 校验）
        safe_path = os.path.join(tmp, 'hf_dinov3.safetensors')
        save_file(ref.state_dict(), safe_path)
        model.load_pretrained(safe_path, strict=True)
        check('safetensors: 官方权重按键名映射后严格加载成功', True)

        try:
            DINOv3ViT(embed_dim=32, depth=2, num_heads=2, n_storage_tokens=NUM_REGISTERS).load_pretrained(
                safe_path, strict=True
            )
            check('strict: 规格不匹配时报错', False, '没有抛异常')
        except RuntimeError:
            check('strict: 规格不匹配时报错', True)

        # register token 数不一致也应被 strict 拦下
        try:
            DINOv3ViT(variant='vits16', n_storage_tokens=0).load_pretrained(safe_path, strict=True)
            check('strict: register token 数不匹配时报错', False, '没有抛异常')
        except RuntimeError:
            check('strict: register token 数不匹配时报错', True)

        torch_path = os.path.join(tmp, 'hf_dinov3.pth')
        torch.save({'state_dict': ref.state_dict()}, torch_path)
        DINOv3ViT(variant='vits16', n_storage_tokens=NUM_REGISTERS, rope_rescale_coords=None).eval().load_pretrained(
            torch_path, strict=True
        )
        check('.pth + state_dict 包装: 加载成功', True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    for hw in [(64, 64), (96, 64), (128, 96)]:
        x = torch.randn(2, 3, *hw)
        with torch.no_grad():
            ref_out = ref(x)
            feats = model.forward_features(x)
        check(f'{hw}: 完整序列 == HF last_hidden_state',
              close(feats['x_norm_alltokens'], ref_out.last_hidden_state),
              maxdiff(feats['x_norm_alltokens'], ref_out.last_hidden_state))
        check(f'{hw}: CLS == HF pooler_output',
              close(feats['x_norm_clstoken'], ref_out.pooler_output),
              maxdiff(feats['x_norm_clstoken'], ref_out.pooler_output))
        check(f'{hw}: patch token == HF 序列去掉前缀',
              close(feats['x_norm_patchtokens'], ref_out.last_hidden_state[:, 1 + NUM_REGISTERS:]),
              maxdiff(feats['x_norm_patchtokens'], ref_out.last_hidden_state[:, 1 + NUM_REGISTERS:]))

    # 无 register token 的配置也要对拍
    size = dinov3_sizes['vits16']
    cfg0 = DINOv3ViTConfig(
        hidden_size=size['embed_dim'],
        num_hidden_layers=size['depth'],
        num_attention_heads=size['num_heads'],
        intermediate_size=int(size['embed_dim'] * size['ffn_ratio']),
        layer_norm_eps=1e-5,
        image_size=64,
        patch_size=16,
        num_register_tokens=0,
        pos_embed_rescale=None,
    )
    torch.manual_seed(1)
    ref0 = DINOv3ViTModel(cfg0).eval()
    model0 = DINOv3ViT(variant='vits16', n_storage_tokens=0, rope_rescale_coords=None).eval()
    model0.load_state_dict(remap_dinov3_keys(ref0.state_dict()), strict=False)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        check('无 register token: 序列一致',
              close(model0.forward_features(x)['x_norm_alltokens'], ref0(x).last_hidden_state),
              maxdiff(model0.forward_features(x)['x_norm_alltokens'], ref0(x).last_hidden_state))

    del ref, model, ref0, model0
    gc.collect()


def test_real_checkpoint():
    section('13. 真实 DINOv3 ViT-B/16 权重迁移（modelscope 快照）')
    weight_path = os.path.join(MODELSCOPE_CKPT, 'model.safetensors')
    if not os.path.exists(weight_path):
        print(f'  [skip] 未找到 {weight_path}')
        return

    ref = DINOv3ViTModel.from_pretrained(MODELSCOPE_CKPT, dtype=torch.float32).eval()
    model = DINOv3ViT(
        variant='vitb16',
        n_storage_tokens=NUM_REGISTERS,
        rope_base=100.0,
        rope_rescale_coords=2.0,
    ).eval()
    model.load_pretrained(weight_path, strict=True)
    check('真实权重严格加载成功（重命名为 codon 布局）', True)

    keys = list(model.state_dict().keys())
    check('加载后键名是 codon 命名', 'blocks.0.q_proj.weight' in keys and 'blocks.0.gamma1' in keys)
    check('embed_dim == 768 / 12 层 / 12 头',
          model.embed_dim == 768 and model.n_blocks == 12 and model.num_heads == 12)

    for hw in [(224, 224), (256, 192)]:
        x = torch.randn(2, 3, *hw)
        with torch.no_grad():
            ref_out = ref(x)
            feats = model.forward_features(x)
        check(f'{hw}: 与官方权重逐元素一致',
              close(feats['x_norm_alltokens'], ref_out.last_hidden_state, atol=1e-5),
              maxdiff(feats['x_norm_alltokens'], ref_out.last_hidden_state))
        check(f'{hw}: CLS 与 pooler_output 一致',
              close(feats['x_norm_clstoken'], ref_out.pooler_output, atol=1e-5),
              maxdiff(feats['x_norm_clstoken'], ref_out.pooler_output))

    # 中间层：与 HF 的 output_hidden_states 对拍
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        hf_out = ref(x, output_hidden_states=True)
        codon_layers = model.get_intermediate_layers(x, n=[0, 5, 11], norm=False)
    for k, layer_idx in enumerate([0, 5, 11]):
        # HF 的 hidden_states[0] 是 embedding 输出，[i+1] 是第 i 层输出，且带前缀 token；
        # 本实现的 get_intermediate_layers 只返回 patch token，因此按前缀长度对齐
        want = hf_out.hidden_states[layer_idx + 1][:, 1 + NUM_REGISTERS:]
        check(f'block {layer_idx} patch 输出与 HF hidden_states 一致',
              close(codon_layers[k], want, atol=1e-5), maxdiff(codon_layers[k], want))

    # 带前缀、带 norm 的完整对比：HF 的 hidden_states 是 norm 之前的，需手工套 norm
    with torch.no_grad():
        codon_full = model.get_intermediate_layers(x, n=[11], norm=True, return_class_token=True,
                                                   return_extra_tokens=True)[0]
        want_normed = ref.norm(hf_out.hidden_states[12])
    check('最后一层 norm 后 patch token 一致',
          close(codon_full[0], want_normed[:, 1 + NUM_REGISTERS:], atol=1e-5),
          maxdiff(codon_full[0], want_normed[:, 1 + NUM_REGISTERS:]))
    check('最后一层 norm 后 cls token 一致',
          close(codon_full[1], want_normed[:, 0], atol=1e-5),
          maxdiff(codon_full[1], want_normed[:, 0]))
    check('最后一层 norm 后 register token 一致',
          close(codon_full[2], want_normed[:, 1: 1 + NUM_REGISTERS], atol=1e-5),
          maxdiff(codon_full[2], want_normed[:, 1: 1 + NUM_REGISTERS]))

    # reshape 分支
    with torch.no_grad():
        reshaped = model.get_intermediate_layers(x, n=1, reshape=True)[0]
    check('reshape: (N, C, 14, 14)', tuple(reshaped.shape) == (1, 768, 14, 14), str(tuple(reshaped.shape)))

    del ref, model
    gc.collect()


if __name__ == '__main__':
    test_contract()
    test_norm_semantics()
    test_rope()
    test_attention_is_bidirectional()
    test_forward_switch()
    test_intermediate_layers()
    test_init_and_backward()
    test_mask_token_path()
    test_key_remap()
    test_helpers()
    test_serialization()
    test_parity_with_transformers()
    test_from_remote_config()
    test_dtype_handling()
    test_converted_artifacts()
    test_real_checkpoint()

    print(f'\n{_CHECKS[0]} checks, {len(_FAILURES)} failed')
    if _FAILURES:
        for name in _FAILURES:
            print(f'  FAILED: {name}')
        sys.exit(1)
    print('ALL PASSED')
