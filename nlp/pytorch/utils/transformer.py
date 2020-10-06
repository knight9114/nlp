# -------------------------------------------------------------------------
#   Transformer Utils
# -------------------------------------------------------------------------
# Imports
from typing import Tuple, Dict
import numpy as np
import torch
from nlp.pytorch.utils import constants


# -------------------------------------------------------------------------
#   Globals
# -------------------------------------------------------------------------
Tensor = torch.Tensor


# -------------------------------------------------------------------------
#   System Functions
# -------------------------------------------------------------------------
def get_device(device:str=constants.DEFAULT_DEVICE) -> torch.device:
    """
    """
    return torch.device(device)


# -------------------------------------------------------------------------
#   Masking Functions
# -------------------------------------------------------------------------
def create_padding_mask(
        seq:Tensor,
        pad_idx:int=constants.PAD_IDX) -> Tensor:
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
        pad_idx:int=constants.PAD_IDX) -> Tuple[Tensor, Tensor]:
    """
    """
    # Create Encoder Mask
    enc_mask = create_padding_mask(src, pad_idx).to(src.device)

    # Create Decoder Mask
    pad_mask = create_padding_mask(tgt, pad_idx).to(tgt.device)
    look_ahead_mask = create_look_ahead_mask(tgt).to(tgt.device)
    dec_mask = torch.max(pad_mask, look_ahead_mask)

    return enc_mask, dec_mask


# -------------------------------------------------------------------------
#   ONNX Functions
# -------------------------------------------------------------------------
def create_transformer_dynamic_axes() -> Dict[str, Dict[int, str]]:
    """
    """
    return {
            'src': {
                0: 'batch_size',
                1: 'src_seq_len'
                },
            'tgt': {
                0: 'batch_size',
                1: 'tgt_seq_len'
                },
            'enc_mask': {
                0: 'batch_size',
                3: 'src_seq_len'
                },
            'dec_mask': {
                0: 'batch_size',
                2: 'tgt_seq_len',
                3: 'tgt_seq_len'
                }
        }

def create_transformer_dummy_inputs(
        batch_size:int=64,
        src_seq_len:int=64,
        tgt_seq_len:int=64) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    """
    # Create Dummy Tokens
    src = torch.randint(0, 100, [batch_size, src_seq_len])
    tgt = torch.randint(0, 100, [batch_size, tgt_seq_len])

    # Create Dummy Masks
    enc_mask, dec_mask = create_masks(src, tgt)

    return src, tgt, enc_mask, dec_mask
