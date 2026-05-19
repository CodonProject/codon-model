import torch


def apply_fourier_mixing(x_v: torch.Tensor, x_g: torch.Tensor, seq_len: int) -> torch.Tensor:
    '''
    Perform causal mixing in the frequency domain via FFT.
    
    Args:
        x_v (torch.Tensor): Content stream tensor of shape [batch_size, n_heads, seq_len, d_head]
        x_g (torch.Tensor): Gate stream tensor of shape [batch_size, n_heads, seq_len, d_head]
        seq_len (int): Original sequence length L.
        
    Returns:
        torch.Tensor: Mixed sequence tensor truncated to original length to enforce causality,
                      cast back to the original input data type.
    '''
    orig_dtype = x_v.dtype
    
    N = 2 * seq_len
    
    x_v_T = x_v.transpose(-1, -2).contiguous().to(torch.float32)
    x_g_T = x_g.transpose(-1, -2).contiguous().to(torch.float32)
    
    V_fft = torch.fft.rfft(x_v_T, n=N, dim=-1)
    G_fft = torch.fft.rfft(x_g_T, n=N, dim=-1)
    
    X_fft = V_fft * G_fft
    
    x_mixed_T = torch.fft.irfft(X_fft, n=N, dim=-1)
    
    x_mixed = x_mixed_T.transpose(-1, -2)
    
    return x_mixed[:, :, :seq_len, :].to(orig_dtype)