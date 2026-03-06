import torch


def make_padding_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    '''
    Creates a padding mask.

    Args:
        src (torch.Tensor): The source sequence tensor. Shape is [B, L_src].
        pad_idx (int, optional): The index of the padding symbol. Defaults to 0.

    Returns:
        torch.Tensor: The padding mask. Shape is [B, 1, 1, L_src].
        True indicates the position is not padding and should be attended to.
    '''
    mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def make_lookahead_mask(size: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    '''
    Creates a lookahead mask (lower triangular matrix).

    Args:
        size (int): The sequence length.
        device (torch.device, optional): The device. Defaults to cpu.

    Returns:
        torch.Tensor: The lookahead mask. Shape is [size, size].
        True indicates allowed positions to attend to (lower triangular part).
    '''
    mask = torch.tril(torch.ones((size, size), device=device)).bool()
    return mask


def make_causal_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    '''
    Creates a causal mask (combining padding mask and lookahead mask).

    Args:
        tgt (torch.Tensor): The target sequence tensor. Shape is [B, L_tgt].
        pad_idx (int, optional): The index of the padding symbol. Defaults to 0.

    Returns:
        torch.Tensor: The causal mask. Shape is [B, 1, L_tgt, L_tgt].
    '''
    pad_mask = make_padding_mask(tgt, pad_idx)
    seq_len = tgt.size(1)
    lookahead_mask = make_lookahead_mask(seq_len, device=tgt.device)

    # pad_mask: [B, 1, 1, L]
    # lookahead_mask: [L, L]
    mask = pad_mask & lookahead_mask
    return mask


def make_sliding_window_mask(
    tensor: torch.Tensor, window_size: int, pad_idx: int = 0, causal: bool = True
) -> torch.Tensor:
    '''
    Creates a sliding window mask.

    Args:
        tensor (torch.Tensor): The input sequence tensor. Shape is [B, L].
        window_size (int): The window size (one-sided).
        pad_idx (int, optional): The index of the padding symbol. Defaults to 0.
        causal (bool, optional): Whether it is causal (unidirectional). Defaults to True.
            If True, position i can only attend to [i - window_size, i].
            If False, position i can attend to [i - window_size, i + window_size].

    Returns:
        torch.Tensor: The sliding window mask. Shape is [B, 1, L, L].
    '''
    pad_mask = make_padding_mask(tensor, pad_idx)  # [B, 1, 1, L]
    seq_len = tensor.size(1)

    ones = torch.ones((seq_len, seq_len), device=tensor.device, dtype=torch.bool)

    if causal:
        # j <= i AND j >= i - window_size
        window_mask = torch.tril(ones, diagonal=0) & torch.triu(
            ones, diagonal=-window_size
        )
    else:
        # j <= i + window_size AND j >= i - window_size
        window_mask = torch.tril(ones, diagonal=window_size) & torch.triu(
            ones, diagonal=-window_size
        )

    mask = pad_mask & window_mask
    return mask
