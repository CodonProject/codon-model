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
    def _gqa_forward_kernel(
        Q, K, V, Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        Z, H_q, H_kv, N_CTX_Q, N_CTX_KV,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
        GQA_GROUP: tl.constexpr, SM_SCALE: tl.constexpr
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H_q
        off_hq = off_hz % H_q
        off_hkv = off_hq // GQA_GROUP

        q_offset = off_z * stride_qz + off_hq * stride_qh
        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        o_offset = off_z * stride_oz + off_hq * stride_oh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)

        q_ptrs = Q + q_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
        o_ptrs = Out + o_offset + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)

        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        for start_n in range(0, N_CTX_KV, BLOCK_N):
            cols = start_n + offs_n
            k_ptrs = K + k_offset + (cols[None, :] * stride_kn + offs_d[:, None] * stride_kk)
            v_ptrs = V + v_offset + (cols[:, None] * stride_vn + offs_d[None, :] * stride_vk)

            k = tl.load(k_ptrs, mask=cols[None, :] < N_CTX_KV, other=0.0)
            
            qk = tl.dot(q, k) * SM_SCALE
            qk = tl.where(offs_m[:, None] < N_CTX_Q, qk, float("-inf"))
            qk = tl.where(cols[None, :] < N_CTX_KV, qk, float("-inf"))

            m_ij = tl.max(qk, 1)
            m_next = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_next)
            beta = tl.exp(m_ij - m_next)

            l_i = l_i * alpha + tl.sum(beta[:, None] * tl.exp(qk - m_ij[:, None]), 1)
            m_i = m_next

            p = tl.exp(qk - m_i[:, None])
            v = tl.load(v_ptrs, mask=cols[:, None] < N_CTX_KV, other=0.0)
            
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

        acc = acc / l_i[:, None]
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX_Q)


def triton_gqa_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float = None) -> torch.Tensor:
    if not HAS_TRITON or not q.is_cuda:
        raise NotImplementedError("Triton GQA is not available or input is not CUDA tensor.")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    orig_dtype = q.dtype
    if orig_dtype == torch.float32:
        q, k, v = q.half(), k.half(), v.half()

    B, H_q, L_q, D = q.shape
    _, H_kv, L_kv, _ = k.shape
    gqa_group = H_q // H_kv
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(q)
    BLOCK_M = 32
    BLOCK_N = 32

    grid = (triton.cdiv(L_q, BLOCK_M), B * H_q)

    _gqa_forward_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H_q, H_kv, L_q, L_kv,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        GQA_GROUP=gqa_group, SM_SCALE=sm_scale
    )
    return out.to(orig_dtype)