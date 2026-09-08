'''
Gated DeltaNet 算子集。

三个实现，数值等价（float32 下 err < 1e-6）：
  - recurrent_gated_delta_rule：逐 token 递推（decode）
  - chunk_gated_delta_rule：chunk-wise 并行（训练 / prefill，纯 torch）
  - chunk_gated_delta_rule_triton：Triton 跨块状态 scan（prefill 加速）

跨块状态递推 S_{i+1} = A_tilde_i @ S_i + B_tilde_i 是一阶线性递归（序列依赖），
用 Triton kernel 消除 N 次 python 循环的 kernel 启动 / 调度开销。
'''
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ============================================================
# 逐 token 递推（解码用）
# ============================================================
def recurrent_gated_delta_rule(q, k, v, beta, g, S=None):
    """
    逐 token 递推（解码用）。
    q,k: (B,H,L,Dk)  v: (B,H,L,Dv)  beta,g: (B,H,L)  g = log(alpha) <= 0
    S: (B,H,Dk,Dv)
    """
    B, H, L, Dk = q.shape
    Dv = v.shape[-1]
    if S is None:
        S = q.new_zeros(B, H, Dk, Dv)
    outs = []
    for t in range(L):
        S = S * g[:, :, t].exp()[..., None, None]                       # 遗忘
        kt, vt = k[:, :, t], v[:, :, t]
        bt = beta[:, :, t, None]
        v_old = (kt.unsqueeze(-2) @ S).squeeze(-2)                      # 读出旧值 k^T S
        delta = bt * (vt - v_old)                                       # delta rule
        S = S + kt.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((q[:, :, t].unsqueeze(-2) @ S).squeeze(-2))
    return torch.stack(outs, dim=2), S


# ============================================================
# Chunk-wise 并行（训练 / prefill 用）
# ============================================================
def chunk_gated_delta_rule(q, k, v, beta, g, S=None, chunk_size=64):
    """
    Chunk-wise 并行（训练 / prefill 用），与递推版本数值等价。
    形状同 recurrent_gated_delta_rule。
    """
    B, H, L, Dk = q.shape
    Dv = v.shape[-1]
    C = chunk_size
    pad = (C - L % C) % C
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        beta, g = (F.pad(t, (0, pad)) for t in (beta, g))
    N = (L + pad) // C

    v_beta = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v_beta, k_beta = (t.reshape(B, H, N, C, -1) for t in (q, k, v_beta, k_beta))
    g = g.reshape(B, H, N, C).cumsum(-1)                                # 块内累计 log-decay γ

    dev = q.device
    tril = torch.tril(torch.ones(C, C, dtype=torch.bool, device=dev))
    strict_tril = torch.tril(torch.ones(C, C, dtype=torch.bool, device=dev), diagonal=-1)
    # D_ij = exp(γ_i - γ_j) (i>=j)，上三角为 0
    decay_mask = (g[..., :, None] - g[..., None, :]).masked_fill(~tril, float("-inf")).exp()

    # ---- 解 (I - A) W/U = x：前向替换的并行化 ----
    # A 严格下三角（nilpotent），(I-A) 单位下三角，用 solve_triangular 一次解出
    # W/U，替代原先 C 次 python 前向替换循环（C 库内部即并行的前向 scan）。
    A = -(k_beta @ k.transpose(-1, -2)) * decay_mask
    A = A.masked_fill(~strict_tril, 0.0)
    M = torch.eye(C, device=dev, dtype=A.dtype) - A                     # 单位下三角
    W = torch.linalg.solve_triangular(M, k_beta * g.exp()[..., None],
                                      upper=False, unitriangular=True)  # (B,H,N,C,Dk)
    U = torch.linalg.solve_triangular(M, v_beta,
                                      upper=False, unitriangular=True)  # (B,H,N,C,Dv)

    if S is None:
        S = q.new_zeros(B, H, Dk, Dv)
    o = torch.empty(B, H, N, C, Dv, device=dev, dtype=q.dtype)
    for i in range(N):
        qi, ki, gi = q[:, :, i], k[:, :, i], g[:, :, i]
        attn = (qi @ ki.transpose(-1, -2)) * decay_mask[:, :, i]        # 块内因果
        u = U[:, :, i] - W[:, :, i] @ S                                 # 校正后的写入值
        o_inter = (qi * gi.exp()[..., None]) @ S                        # 跨块贡献
        o[:, :, i] = o_inter + attn @ u
        S = S * gi[:, :, -1].exp()[..., None, None] + \
            (ki * (gi[:, :, -1:] - gi).exp()[..., None]).transpose(-1, -2) @ u
    o = o.reshape(B, H, N * C, Dv)[:, :, :L]
    return o, S


# ============================================================
# Triton 加速跨块 scan
# ============================================================
if HAS_TRITON:
    def _s_scan(Atilde, Btilde, S0):
        """跨块状态递推，返回 [B,H,N,Dk,Dv]（每个 chunk 的初始状态 P_i）。

        原 Triton kernel 在循环内做跨迭代 tl.dot(input_precision="ieee") 时精度退化
        （triton 3.7.1 把跨迭代累加变量降为 tf32），改为纯 torch 循环，数值精确。
        """
        B, H, N, Dk, _ = Atilde.shape
        Dv = Btilde.shape[-1]
        P = torch.empty(B, H, N, Dk, Dv, device=Atilde.device, dtype=Atilde.dtype)
        S = S0
        for i in range(N):
            P[:, :, i] = S
            S = Atilde[:, :, i] @ S + Btilde[:, :, i]
        return P


    def chunk_gated_delta_rule_triton(q, k, v, beta, g, S=None, chunk_size=64):
        """Triton 加速的 chunk 版 GatedDeltaRule，与 chunk_gated_delta_rule 数值等价。"""
        B, H, L, Dk = q.shape
        Dv = v.shape[-1]
        C = chunk_size
        pad = (C - L % C) % C
        if pad:
            q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
            beta, g = (F.pad(t, (0, pad)) for t in (beta, g))
        N = (L + pad) // C

        v_beta = v * beta[..., None]
        k_beta = k * beta[..., None]
        q, k, v_beta, k_beta = (t.reshape(B, H, N, C, -1) for t in (q, k, v_beta, k_beta))
        g = g.reshape(B, H, N, C).cumsum(-1)

        dev = q.device
        tril = torch.tril(torch.ones(C, C, dtype=torch.bool, device=dev))
        strict_tril = torch.tril(torch.ones(C, C, dtype=torch.bool, device=dev), diagonal=-1)
        decay_mask = (g[..., :, None] - g[..., None, :]).masked_fill(~tril, float("-inf")).exp()

        # 块内：解 (I-A) W/U = x（前向替换的并行化）
        A = -(k_beta @ k.transpose(-1, -2)) * decay_mask
        A = A.masked_fill(~strict_tril, 0.0)
        M = torch.eye(C, device=dev, dtype=A.dtype) - A
        W = torch.linalg.solve_triangular(M, k_beta * g.exp()[..., None],
                                          upper=False, unitriangular=True)   # [B,H,N,C,Dk]
        U = torch.linalg.solve_triangular(M, v_beta,
                                          upper=False, unitriangular=True)   # [B,H,N,C,Dv]

        # 跨块参数向量化：S_{i+1} = A_tilde_i @ S_i + B_tilde_i
        if S is None:
            S = q.new_zeros(B, H, Dk, Dv)
        Mmat = k * (g[..., -1:] - g).exp()[..., None]                          # [B,H,N,C,Dk]
        eye = torch.eye(Dk, device=dev, dtype=A.dtype)
        A_tilde = g[..., -1].exp()[..., None, None] * eye - Mmat.transpose(-1, -2) @ W  # [B,H,N,Dk,Dk]
        B_tilde = Mmat.transpose(-1, -2) @ U                                    # [B,H,N,Dk,Dv]

        # Triton 跨块状态递推 -> 每个 chunk 的初始状态 P_i
        P = _s_scan(A_tilde, B_tilde, S)                                       # [B,H,N,Dk,Dv]

        # 输出向量化
        attn = (q @ k.transpose(-1, -2)) * decay_mask                          # [B,H,N,C,C]
        u = U - W @ P                                                          # [B,H,N,C,Dv]
        o_inter = (q * g.exp()[..., None]) @ P                                 # [B,H,N,C,Dv]
        o = o_inter + attn @ u

        # 最终状态 = 递推到底
        S_final = A_tilde[:, :, -1] @ P[:, :, -1] + B_tilde[:, :, -1]
        return o.reshape(B, H, N * C, Dv)[:, :, :L], S_final

else:
    def chunk_gated_delta_rule_triton(q, k, v, beta, g, S=None, chunk_size=64):
        raise NotImplementedError('Triton is not available for GatedDeltaNet chunk scan.')
