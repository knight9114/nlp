# -------------------------------------------------------------------------
#   Transformer Utils
# -------------------------------------------------------------------------
# Imports
from typing import Tuple
import numpy as np
import torch


# -------------------------------------------------------------------------
#   Globals
# -------------------------------------------------------------------------
Tensor = torch.Tensor


# -------------------------------------------------------------------------
#   Masking Functions
# -------------------------------------------------------------------------
def create_padding_mask(
        seq:Tensor,
        pad_idx:int=0) -> Tensor:
    """
    """
    return (seq == pad_idx).type(torch.float32)[:, np.newaxis, np.newaxis, :]

def create_look_ahead_mask(
        seq:Tensor) -> Tensor:
    """
    """
    n = seq.shape[1]
    return torch.triu(torch.ones([n,n]), 1)

def create_masks(
        src:Tensor,
        tgt:Tensor,
        pad_idx:int=0) -> Tuple[Tensor, Tensor]:
    """
    """
    # Create Encoder Mask
    enc_mask = create_padding_mask(src, pad_idx)

    # Create Decoder Mask
    pad_mask = create_padding_mask(tgt, pad_idx)
    look_ahead_mask = create_look_ahead_mask(tgt)
    dec_mask = torch.max(pad_mask, look_ahead_mask)

    return enc_mask, dec_mask
