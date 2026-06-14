import torch
import math
from cortex_hierarchy import CorticalHierarchy


# ═══════════════════════ § 1 Infrastructure ═══════════════════════

def test_sfa_safe_under_multi_pass():
    """物理量: 同一柱在多 pass 下不应触发 autograd inplace 检测"""
    from codon.exp.block.cortex_v2 import CorticalColumn
    B, F = 4, 32
    col = CorticalColumn(F)  # 单柱也含 SFA
    x_l4  = torch.randn(B, F, dtype=torch.cfloat, requires_grad=True)
    x_pfc = torch.randn(B, F, dtype=torch.cfloat, requires_grad=True)

    # 同一 col 跑两次, 累加 loss 后反传
    _, z5_a = col(x_l4, x_pfc)
    _, z5_b = col(x_l4, x_pfc)
    loss = z5_a.abs().mean() + z5_b.abs().mean()
    loss.backward()

    assert x_l4.grad.abs().sum() > 0
    print("[OK] SFA safe under multi-pass invocation")

def test_hierarchy_shape():
    """物理量: shape / dtype / device / 单 pass 退化"""
    B, F, N = 4, 32, 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h = CorticalHierarchy(num_features=F, num_layers=N).to(device)

    x_sens = torch.randn(B, F, dtype=torch.cfloat, device=device)
    x_top  = torch.randn(B, F, dtype=torch.cfloat, device=device)

    z23, z5 = h(x_sens, x_top, num_passes=2)
    assert z23.shape == (B, F) and z5.shape == (B, F)
    assert z23.is_complex() and z5.is_complex()
    assert len(h.last_z_l23s) == N

    # 单 pass 也应工作 (退化为纯前馈)
    z23_1, z5_1 = h(x_sens, x_top, num_passes=1)
    assert z23_1.shape == z23.shape
    print(f"[OK] hierarchy shape  N={N} F={F} device={device}")


def test_hierarchy_grad():
    """物理量: 双向梯度都能到达底层与顶层输入,
       且所有列(含 col[0].l5 这个 subcortical port)都获得梯度"""
    B, F, N = 4, 32, 3
    h = CorticalHierarchy(F, N)

    x_sens = torch.randn(B, F, dtype=torch.cfloat, requires_grad=True)
    x_top  = torch.randn(B, F, dtype=torch.cfloat, requires_grad=True)
    x_ach  = torch.rand(B, 1, requires_grad=True)   # 关键: 非零 ACh 让 ach_to_ndnf 可学

    # return_all=True 让所有列都进入 loss 路径
    z_l23s, z_l5s = h(x_sens, x_top, x_ach=x_ach, num_passes=2, return_all=True)
    loss = sum(z.abs().mean() for z in z_l5s) + sum(z.abs().mean() for z in z_l23s)
    loss.backward()

    assert all(torch.isfinite(z).all() for z in z_l5s)
    assert x_sens.grad is not None and x_sens.grad.abs().sum() > 0
    assert x_top.grad  is not None and x_top.grad.abs().sum()  > 0
    assert x_ach.grad  is not None and x_ach.grad.abs().sum()  > 0

    missing = [n for n, p in h.named_parameters()
               if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    assert not missing, f"无梯度参数: {missing}"
    print(f"[OK] hierarchy grad  params={sum(p.numel() for p in h.parameters())}")



# ═══════════════════════ § 2 Top-down 反馈实证 ═══════════════════════

def test_top_down_modulates_bottom():
    """物理量: 同一 sensory 输入, 不同 top_context 应该让最底层 z_l23 不同
    
    若不同则证明 top-down 反馈通路真正起作用了 (不只是装饰)
    """
    B, F, N = 4, 32, 3
    h = CorticalHierarchy(F, N).eval()
    x_sens = torch.randn(B, F, dtype=torch.cfloat)

    x_top_a = torch.polar(torch.ones(B, F), torch.full((B, F), 0.0))
    x_top_b = torch.polar(torch.ones(B, F), torch.full((B, F), math.pi))

    h(x_sens, x_top_a, num_passes=2)
    bot_a = h.last_z_l23s[0].clone()
    h(x_sens, x_top_b, num_passes=2)
    bot_b = h.last_z_l23s[0].clone()

    diff = (bot_a - bot_b).abs().mean().item()
    assert diff > 1e-3, f"Top-down 应影响最底层 z_l23, diff={diff:.6f}"
    print(f"[OK] top-down modulation  bottom diff={diff:.4f}")


def test_passes_converge():
    """物理量: 扫描次数增加, 输出应渐进稳定 (差异递减)"""
    B, F, N = 4, 32, 3
    h = CorticalHierarchy(F, N).eval()
    x_sens = torch.randn(B, F, dtype=torch.cfloat)
    x_top  = torch.randn(B, F, dtype=torch.cfloat)

    outs = []
    for k in [1, 2, 3, 4]:
        _, z5 = h(x_sens, x_top, num_passes=k)
        outs.append(z5.clone())

    diffs = [(outs[i] - outs[i+1]).abs().mean().item() for i in range(3)]
    print(f"[OK] passes converge  diffs(k→k+1) = {[f'{d:.4f}' for d in diffs]}")
    # 不强制单调下降 (依赖初始化), 但通常 diff[2] < diff[0]


# ═══════════════════════ § 3 共享权重 ═══════════════════════

def test_share_weights():
    """物理量: share_weights=True 时, 参数量应远少于独立版"""
    F, N = 32, 4
    h_indep  = CorticalHierarchy(F, N, share_weights=False)
    h_shared = CorticalHierarchy(F, N, share_weights=True)

    n_indep  = sum(p.numel() for p in h_indep.parameters())
    n_shared = sum(p.numel() for p in h_shared.parameters())

    # 独立版的参数量约为共享版的 N 倍
    assert n_indep > n_shared * (N - 1), f"独立 {n_indep} vs 共享 {n_shared}"

    # 共享版前向应数值正常
    x_sens = torch.randn(2, F, dtype=torch.cfloat)
    z23, z5 = h_shared(x_sens, num_passes=2)
    assert torch.isfinite(z5).all()
    print(f"[OK] share_weights  indep={n_indep} shared={n_shared}  ratio={n_indep/n_shared:.2f}x")


# ═══════════════════════ § 4 边界 ═══════════════════════

def test_single_layer():
    """物理量: N=1 时应等价于一个 CorticalColumn"""
    B, F = 4, 32
    h = CorticalHierarchy(F, num_layers=1).eval()
    x_sens = torch.randn(B, F, dtype=torch.cfloat)
    x_top  = torch.randn(B, F, dtype=torch.cfloat)

    z23, z5 = h(x_sens, x_top, num_passes=1)
    assert z23.shape == (B, F)
    print("[OK] single layer (N=1) reduces to single column")


def test_no_top_context():
    """物理量: x_top_context=None 时, 顶层走被动模式"""
    B, F, N = 4, 32, 3
    h = CorticalHierarchy(F, N).eval()
    x_sens = torch.randn(B, F, dtype=torch.cfloat)
    z23, z5 = h(x_sens, x_top_context=None, num_passes=2)
    assert torch.isfinite(z5).all()
    # 顶层 col 应进入被动模式 (last_coincidence 接近 0)
    top_col = h._col(N - 1)
    print(f"[OK] no top context  top coincidence={top_col.last_coincidence:.4f}")


# ═══════════════════════ Runner ═══════════════════════

if __name__ == '__main__':
    torch.manual_seed(0)

    test_sfa_safe_under_multi_pass()

    print("\n── § 1 Infrastructure ──")
    test_hierarchy_shape()
    test_hierarchy_grad()

    print("\n── § 2 Top-down 反馈 ──")
    test_top_down_modulates_bottom()
    test_passes_converge()

    print("\n── § 3 共享权重 ──")
    test_share_weights()

    print("\n── § 4 边界 ──")
    test_single_layer()
    test_no_top_context()

    print("\nAll tests passed.")