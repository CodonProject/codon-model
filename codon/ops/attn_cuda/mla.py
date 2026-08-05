import torch
import math

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _mla_forward_kernel(
        Q_c, Q_p, KV_latent, K_p, Out,
        stride_qcz, stride_qch, stride_qcm, stride_qck,
        stride_qpz, stride_qph, stride_qpm, stride_qpk,
        stride_kvz, stride_kvn, stride_kvk,
        stride_kpz, stride_kph, stride_kpn, stride_kpk,
        stride_oz, stride_oh, stride_om, stride_ok,
        W_kc, W_vc,
        stride_wkc_h, stride_wkc_d, stride_wkc_r,
        stride_wvc_h, stride_wvc_r, stride_wvc_d,
        B, H, L_q, L_kv, D_c, D_p, R_kv,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        SM_SCALE: tl.constexpr
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_dc = tl.arange(0, D_c)
        offs_dp = tl.arange(0, D_p)
        offs_r = tl.arange(0, R_kv)

        qc_ptrs = Q_c + off_z * stride_qcz + off_h * stride_qch + (offs_m[:, None] * stride_qcm + offs_dc[None, :] * stride_qck)
        qp_ptrs = Q_p + off_z * stride_qpz + off_h * stride_qph + (offs_m[:, None] * stride_qpm + offs_dp[None, :] * stride_qpk)
        
        qc = tl.load(qc_ptrs, mask=offs_m[:, None] < L_q, other=0.0)
        qp = tl.load(qp_ptrs, mask=offs_m[:, None] < L_q, other=0.0)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D_c + D_p], dtype=tl.float32)

        for start_n in range(0, L_kv, BLOCK_N):
            cols = start_n + offs_n
            kv_ptrs = KV_latent + off_z * stride_kvz + (cols[None, :] * stride_kvn + offs_r[:, None] * stride_kvk)
            kp_ptrs = K_p + off_z * stride_kpz + off_h * stride_kph + (cols[None, :] * stride_kpn + offs_dp[:, None] * stride_kpk)

            kv_lat = tl.load(kv_ptrs, mask=cols[None, :] < L_kv, other=0.0)
            kp = tl.load(kp_ptrs, mask=cols[None, :] < L_kv, other=0.0)

            qk_c = tl.dot(qc, tl.load(W_kc + off_h * stride_wkc_h + offs_dc[:, None] * stride_wkc_d + offs_r[None, :] * stride_wkc_r))
            qk_lat = tl.dot(qk_c, kv_lat)
            qk_p = tl.dot(qp, kp)
            
            qk = (qk_lat + qk_p) * SM_SCALE
            qk = tl.where(offs_m[:, None] < L_q, qk, float("-inf"))
            qk = tl.where(cols[None, :] < L_kv, qk, float("-inf"))

            m_ij = tl.max(qk, 1)
            m_next = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_next)
            beta = tl.exp(m_ij - m_next)

            l_i = l_i * alpha + tl.sum(beta[:, None] * tl.exp(qk - m_ij[:, None]), 1)
            m_i = m_next
            p = tl.exp(qk - m_i[:, None])

            v_lat = tl.dot(kv_lat.to(Q_c.dtype), tl.load(W_vc + off_h * stride_wvc_h + offs_r[:, None] * stride_wvc_r + offs_dc[None, :] * stride_wvc_d))
            acc[:, :D_c] = acc[:, :D_c] * alpha[:, None] + tl.dot(p.to(v_lat.dtype), v_lat.t())

        acc = acc / l_i[:, None]
        o_ptrs = Out + off_z * stride_oz + off_h * stride_oh + (offs_m[:, None] * stride_om + tl.arange(0, D_c + D_p)[None, :] * stride_ok)
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < L_q)


def triton_mla_forward(q_c: torch.Tensor, q_p: torch.Tensor, kv_latent: torch.Tensor, k_p: torch.Tensor, w_kc: torch.Tensor, w_vc: torch.Tensor, sm_scale: float = None) -> torch.Tensor:
    if not HAS_TRITON or not q_c.is_cuda:
        raise NotImplementedError("Triton MLA is not available or input is not CUDA tensor.")

    q_c, q_p, kv_latent, k_p, w_kc, w_vc = (
        q_c.contiguous(), q_p.contiguous(), kv_latent.contiguous(),
        k_p.contiguous(), w_kc.contiguous(), w_vc.contiguous()
    )

    orig_dtype = q_c.dtype
    if orig_dtype == torch.float32:
        q_c, q_p, kv_latent, k_p, w_kc, w_vc = (
            q_c.half(), q_p.half(), kv_latent.half(),
            k_p.half(), w_kc.half(), w_vc.half()
        )

    B, H, L_q, D_c = q_c.shape
    _, _, _, D_p = q_p.shape
    _, L_kv, R_kv = kv_latent.shape
    
    out = torch.empty(B, H, L_q, D_c + D_p, device=q_c.device, dtype=q_c.dtype)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D_c + D_p)

    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(L_q, BLOCK_M), B * H)

    _mla_forward_kernel[grid](
        q_c, q_p, kv_latent, k_p, out,
        q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
        q_p.stride(0), q_p.stride(1), q_p.stride(2), q_p.stride(3),
        kv_latent.stride(0), kv_latent.stride(1), kv_latent.stride(2),
        k_p.stride(0), k_p.stride(1), k_p.stride(2), k_p.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        w_kc, w_vc,
        w_kc.stride(0), w_kc.stride(1), w_kc.stride(2),
        w_vc.stride(0), w_vc.stride(1), w_vc.stride(2),
        B, H, L_q, L_kv, D_c, D_p, R_kv,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, SM_SCALE=sm_scale
    )
    return out.to(orig_dtype)