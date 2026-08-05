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
    def _hca_forward_kernel(
        Q, Merged_KV, W_k, W_v, Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kn, stride_kd,
        stride_wk_d, stride_wk_h,
        stride_wv_d, stride_wv_h,
        stride_oz, stride_oh, stride_om, stride_ok,
        B, H, L_q, Num_Blocks, Latent_Dim, Head_Dim,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        SM_SCALE: tl.constexpr
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, Head_Dim)
        offs_lat = tl.arange(0, Latent_Dim)

        q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
        q = tl.load(q_ptrs, mask=offs_m[:, None] < L_q, other=0.0)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, Head_Dim], dtype=tl.float32)

        wk_ptrs = W_k + offs_lat[:, None] * stride_wk_d + off_h * Head_Dim + offs_d[None, :]
        wv_ptrs = W_v + offs_lat[:, None] * stride_wv_d + off_h * Head_Dim + offs_d[None, :]
        w_k = tl.load(wk_ptrs)
        w_v = tl.load(wv_ptrs)

        for start_n in range(0, Num_Blocks, BLOCK_N):
            cols = start_n + offs_n
            kv_ptrs = Merged_KV + off_z * stride_kz + cols[:, None] * stride_kn + offs_lat[None, :] * stride_kd
            merged_lat = tl.load(kv_ptrs, mask=cols[:, None] < Num_Blocks, other=0.0)

            k = tl.dot(merged_lat, w_k)
            v = tl.dot(merged_lat, w_v)

            qk = tl.dot(q, k.t()) * SM_SCALE
            qk = tl.where(offs_m[:, None] < L_q, qk, float("-inf"))
            qk = tl.where(cols[None, :] < Num_Blocks, qk, float("-inf"))

            m_ij = tl.max(qk, 1)
            m_next = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_next)
            beta = tl.exp(m_ij - m_next)

            l_i = l_i * alpha + tl.sum(beta[:, None] * tl.exp(qk - m_ij[:, None]), 1)
            m_i = m_next
            p = tl.exp(qk - m_i[:, None])

            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

        acc = acc / l_i[:, None]
        o_ptrs = Out + off_z * stride_oz + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < L_q)


def triton_hca_forward(q: torch.Tensor, merged_kv: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor, sm_scale: float = None) -> torch.Tensor:
    if not HAS_TRITON or not q.is_cuda:
        raise NotImplementedError("Triton HCA is not available or input is not CUDA tensor.")

    q, merged_kv, w_k, w_v = q.contiguous(), merged_kv.contiguous(), w_k.contiguous(), w_v.contiguous()

    orig_dtype = q.dtype
    if orig_dtype == torch.float32:
        q, merged_kv, w_k, w_v = q.half(), merged_kv.half(), w_k.half(), w_v.half()

    B, H, L_q, Head_Dim = q.shape
    _, Num_Blocks, Latent_Dim = merged_kv.shape

    out = torch.empty_like(q)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(Head_Dim)

    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(L_q, BLOCK_M), B * H)

    _hca_forward_kernel[grid](
        q, merged_kv, w_k, w_v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        merged_kv.stride(0), merged_kv.stride(1), merged_kv.stride(2),
        w_k.stride(0), w_k.stride(1),
        w_v.stride(0), w_v.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, L_q, Num_Blocks, Latent_Dim, Head_Dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, SM_SCALE=sm_scale
    )
    return out.to(orig_dtype)