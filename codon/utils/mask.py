import torch

from tokenizers  import Tokenizer
from dataclasses import dataclass

from typing import Union



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


@dataclass
class MaskedContent:
    '''
    Result of the token masking process.

    Attributes:
        content (str): The original text content.
        tokenized (Union[list[int], torch.Tensor]): The list of token IDs or tensor.
        mask (Union[list[int], torch.Tensor]): The mask values (0 for masked, 1 for unmasked).
    '''
    content: str
    tokenized: Union[list[int], torch.Tensor]
    mask: Union[list[int], torch.Tensor]


class TokenMask:
    '''
    Handles token masking logic based on special tokens.
    '''

    def __init__(self, tokenizer: Tokenizer) -> None:
        '''
        Initializes the TokenMask.

        Args:
            tokenizer (Tokenizer): The configured tokenizer instance.
        '''
        self.tokenizer = tokenizer

    def mask(
        self,
        content: str,
        special_token: Union[str, int, list[Union[str, int]]],
        tensor_mask: bool = True
    ) -> MaskedContent:
        '''
        Tokenizes content and generates a mask based on the first found special token.

        The tokens before and including the special token are masked (0).
        The tokens after the special token are unmasked (1).
        If the special token is not found, all tokens are masked (0).

        Args:
            content (str): The text content to tokenize and mask.
            special_token (Union[str, int, list[Union[str, int]]]): The special token(s) to use as a separator.
                Can be a string, an integer ID, or a priority list of strings/integers.
            tensor_mask (bool, optional): Whether to return tensors instead of lists. Defaults to True.

        Returns:
            MaskedContent: Dataclass containing the original content, token IDs, and the generated mask.
        '''
        encoded = self.tokenizer.encode(content)
        ids = encoded.ids

        candidates = []
        if isinstance(special_token, list):
            candidates = special_token
        else:
            candidates = [special_token]

        split_index = -1

        for cand in candidates:
            tid = None
            if isinstance(cand, str):
                tid = self.tokenizer.token_to_id(cand)
            elif isinstance(cand, int):
                tid = cand

            if tid is not None:
                try:
                    split_index = ids.index(tid)
                    break
                except ValueError: continue

        mask = [0] * len(ids)
        if split_index != -1:
            for i in range(split_index + 1, len(ids)):
                mask[i] = 1

        if tensor_mask:
            ids = torch.tensor(ids)
            mask = torch.tensor(mask)

        return MaskedContent(content=content, tokenized=ids, mask=mask)
