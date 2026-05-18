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
    
    V_fft = torch.fft.rfft(x_v.to(torch.float32), n=N, dim=2)
    G_fft = torch.fft.rfft(x_g.to(torch.float32), n=N, dim=2)
    
    X_fft = V_fft * G_fft
    
    x_mixed = torch.fft.irfft(X_fft, n=N, dim=2)
    
    return x_mixed[:, :, :seq_len, :].to(orig_dtype)