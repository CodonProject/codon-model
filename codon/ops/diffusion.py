import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _rms_norm_modulation_kernel(
        X_ptr, W_ptr, Alpha_ptr, Beta_ptr, Out_ptr,
        stride_x_m, stride_x_n,
        stride_w_n,
        stride_alpha_b, stride_alpha_n,
        stride_beta_b, stride_beta_n,
        stride_out_m, stride_out_n,
        M, N, Seq,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        if row_idx >= M:
            return

        batch_idx = row_idx // Seq
        col_offsets = tl.arange(0, BLOCK_N)
        safe_offsets = tl.where(col_offsets < N, col_offsets, 0)   # 保证偏移在 [0, N-1]

        x_ptrs = X_ptr + row_idx * stride_x_m + safe_offsets * stride_x_n
        alpha_ptrs = Alpha_ptr + batch_idx * stride_alpha_b + safe_offsets * stride_alpha_n
        beta_ptrs = Beta_ptr + batch_idx * stride_beta_b + safe_offsets * stride_beta_n
        out_ptrs = Out_ptr + row_idx * stride_out_m + safe_offsets * stride_out_n
        w_ptrs = W_ptr + safe_offsets * stride_w_n

        mask = col_offsets < N

        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        alpha = tl.load(alpha_ptrs, mask=mask, other=1.0).to(tl.float32)
        beta = tl.load(beta_ptrs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptrs, mask=mask, other=1.0).to(tl.float32)

        var = tl.sum(x * x, axis=0) / N
        rrms = tl.sqrt(var + eps)

        normed = x * rrms * w
        out = normed * alpha + beta

        tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty), mask=mask)


def fused_rms_norm_modulation(x, norm_layer, alpha, beta, use_triton=True):
    """
    融合 RMSNorm 与 Modulation，alpha/beta 形状为 (B, Dim)
    """
    # 回退条件：禁用 Triton / 无 Triton / 非 CUDA / ONNX 导出
    if not use_triton or not HAS_TRITON or not x.is_cuda or torch.onnx.is_in_onnx_export():
        x_norm = norm_layer(x)
        return x_norm * alpha.unsqueeze(1) + beta.unsqueeze(1)

    B, Seq, Dim = x.shape
    x = x.contiguous()

    # 获取 weight 和 eps
    weight = norm_layer.weight if hasattr(norm_layer, 'weight') and norm_layer.weight is not None else torch.ones(Dim, device=x.device, dtype=x.dtype)
    eps = float(norm_layer.eps) if hasattr(norm_layer, 'eps') and norm_layer.eps is not None else 1e-5

    M = B * Seq
    N = Dim
    out = torch.empty_like(x)

    BLOCK_N = triton.next_power_of_2(N)
    if BLOCK_N > 8192:   # 超大维度回退
        x_norm = norm_layer(x)
        return x_norm * alpha.unsqueeze(1) + beta.unsqueeze(1)

    grid = (M,)
    _rms_norm_modulation_kernel[grid](
        x, weight, alpha, beta, out,
        x.stride(0), x.stride(1),
        weight.stride(0),
        alpha.stride(0), alpha.stride(1),   # alpha 原本就是 (B, Dim)
        beta.stride(0), beta.stride(1),
        out.stride(0), out.stride(1),
        M, N, Seq,                         # 多传 Seq
        eps,
        BLOCK_N=BLOCK_N,
    )
    return out