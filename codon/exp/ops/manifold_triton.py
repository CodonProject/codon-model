import torch

try:
    import triton
    import triton.language as tl
    JIT = True
except ImportError:
    JIT = False

if JIT:
    @triton.jit
    def manifold_linear_fuse_kernel_forward(
        c_ptr, scale_ptr, bias_ptr, out_ptr,
        kappa_val, lambda_val,
        n_elements, N,
        BLOCK_SIZE: tl.constexpr, RULE_IS_NEAR: tl.constexpr
    ):
        '''
        Forward pass fusion kernel.
        Fuses clamp, acos, exp, attraction logic, and cosine projection.
        '''
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Determine which feature (column) this thread is computing
        feat_idx = offsets % N

        # Load data from HBM to SRAM
        c = tl.load(c_ptr + offsets, mask=mask)
        scale = tl.load(scale_ptr + feat_idx, mask=mask)
        bias = tl.load(bias_ptr + feat_idx, mask=mask)

        # 1. Clamp to prevent acos NaN
        c_clamp = tl.maximum(c, -1.0 + 1e-6)
        c_clamp = tl.minimum(c_clamp, 1.0 - 1e-6)

        # 2. Angle and Gravitational Field Calculation
        theta = tl.acos(c_clamp)
        exp_val = tl.exp(kappa_val * (c_clamp - 1.0))

        if RULE_IS_NEAR:
            attraction = exp_val
        else:
            attraction = 1.0 - exp_val

        # 3. Geodesic Pullback
        safe_lambda = tl.maximum(lambda_val, 1e-6)
        safe_lambda = tl.minimum(safe_lambda, 1.0 - 1e-4)

        effective_theta = theta * (1.0 - safe_lambda * attraction)
        
        # 4. Output generation
        out = scale * tl.cos(effective_theta) + bias

        # Store to HBM
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def manifold_linear_fuse_kernel_backward(
        grad_out_ptr, c_ptr, scale_ptr,
        grad_c_ptr, grad_k_ptr, grad_l_ptr, grad_s_ptr, grad_b_ptr,
        kappa_val, lambda_val,
        n_elements, N,
        BLOCK_SIZE: tl.constexpr, RULE_IS_NEAR: tl.constexpr
    ):
        '''
        Backward pass fusion kernel.
        Calculates exact analytical gradients without saving intermediate tensors.
        '''
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        feat_idx = offsets % N

        # Load gradients and saved tensors
        grad_out = tl.load(grad_out_ptr + offsets, mask=mask)
        c = tl.load(c_ptr + offsets, mask=mask)
        scale = tl.load(scale_ptr + feat_idx, mask=mask)

        # Recompute intermediates (Registers are incredibly fast compared to HBM loads)
        c_clamp = tl.maximum(c, -1.0 + 1e-6)
        c_clamp = tl.minimum(c_clamp, 1.0 - 1e-6)

        theta = tl.acos(c_clamp)
        exp_val = tl.exp(kappa_val * (c_clamp - 1.0))

        if RULE_IS_NEAR:
            attraction = exp_val
        else:
            attraction = 1.0 - exp_val

        safe_lambda = tl.maximum(lambda_val, 1e-6)
        safe_lambda = tl.minimum(safe_lambda, 1.0 - 1e-4)
        effective_theta = theta * (1.0 - safe_lambda * attraction)

        # -- Gradients for Scale and Bias --
        cos_teff = tl.cos(effective_theta)
        sin_teff = tl.sin(effective_theta)
        
        grad_s = grad_out * cos_teff
        grad_b = grad_out

        # -- Gradients for Theta_eff --
        g_teff = grad_out * (-scale * sin_teff)

        # -- Gradients for Lambda --
        grad_l = g_teff * (-theta * attraction)

        # -- Gradients for Kappa and Cosine (c) --
        d_teff_d_theta = 1.0 - safe_lambda * attraction
        d_teff_d_attr = -theta * safe_lambda

        # d(theta) / d(c_clamp)
        denom = tl.maximum(1.0 - c_clamp * c_clamp, 1e-7)
        d_theta_d_c = -1.0 / tl.sqrt(denom)

        # d(attraction) / d(...)
        if RULE_IS_NEAR:
            d_attr_d_k = exp_val * (c_clamp - 1.0)
            d_attr_d_c = exp_val * kappa_val
        else:
            d_attr_d_k = -exp_val * (c_clamp - 1.0)
            d_attr_d_c = -exp_val * kappa_val

        grad_k = g_teff * d_teff_d_attr * d_attr_d_k

        # Chain rule back to c_clamp
        d_teff_d_c = (d_teff_d_theta * d_theta_d_c) + (d_teff_d_attr * d_attr_d_c)
        grad_c = g_teff * d_teff_d_c

        # Enforce clamp gradient gating (only propagate if within bounds)
        in_bounds = (c > -1.0 + 1e-6) & (c < 1.0 - 1e-6)
        grad_c = tl.where(in_bounds, grad_c, 0.0)

        # Store gradients to HBM
        tl.store(grad_c_ptr + offsets, grad_c, mask=mask)
        tl.store(grad_k_ptr + offsets, grad_k, mask=mask)
        tl.store(grad_l_ptr + offsets, grad_l, mask=mask)
        tl.store(grad_s_ptr + offsets, grad_s, mask=mask)
        tl.store(grad_b_ptr + offsets, grad_b, mask=mask)


    class ManifoldLinearFuseFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, cosine, kappa, lambda_rate, scale, bias, rule):
            cosine = cosine.contiguous()
            scale = scale.contiguous()
            bias = bias.contiguous()

            out = torch.empty_like(cosine)
            n_elements = cosine.numel()
            N = cosine.size(-1)

            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            rule_is_near = (rule == 'near')

            manifold_linear_fuse_kernel_forward[grid](
                cosine, scale, bias, out,
                kappa.item(), lambda_rate.item(),
                n_elements, N,
                BLOCK_SIZE=1024, RULE_IS_NEAR=rule_is_near
            )

            ctx.save_for_backward(cosine, kappa, lambda_rate, scale, bias)
            ctx.rule = rule
            return out

        @staticmethod
        def backward(ctx, grad_out):
            cosine, kappa, lambda_rate, scale, bias = ctx.saved_tensors
            rule = ctx.rule
            grad_out = grad_out.contiguous()

            # Allocate gradient grids
            grad_c = torch.empty_like(cosine)
            grad_k_grid = torch.empty_like(cosine)
            grad_l_grid = torch.empty_like(cosine)
            grad_s_grid = torch.empty_like(cosine)
            grad_b_grid = torch.empty_like(cosine)

            n_elements = cosine.numel()
            N = cosine.size(-1)
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            rule_is_near = (rule == 'near')

            manifold_linear_fuse_kernel_backward[grid](
                grad_out, cosine, scale,
                grad_c, grad_k_grid, grad_l_grid, grad_s_grid, grad_b_grid,
                kappa.item(), lambda_rate.item(),
                n_elements, N,
                BLOCK_SIZE=1024, RULE_IS_NEAR=rule_is_near
            )

            # Python-level reduction
            grad_kappa = grad_k_grid.sum().view_as(kappa)
            grad_lambda = grad_l_grid.sum().view_as(lambda_rate)

            # Gate lambda gradients based on forward clamp bounds
            l_val = lambda_rate.item()
            if l_val <= 1e-6 or l_val >= 1.0 - 1e-4:
                grad_lambda.zero_()

            grad_scale = grad_s_grid.sum(dim=0).view_as(scale)
            grad_bias = grad_b_grid.sum(dim=0).view_as(bias)

            return grad_c, grad_kappa, grad_lambda, grad_scale, grad_bias, None