# -------------------------------------------------------------------------
#   Transformer Layers
# -------------------------------------------------------------------------
# Imports
from typing import Optional
import torch
from torch import nn
from .multihead_attention import MultiheadAttention
from .pointwise_feedforward import PointwiseFeedforward
from .positional_encoding import PositionalEncoding
from .processing import ProcessingLayer


# -------------------------------------------------------------------------
#   Globals
# -------------------------------------------------------------------------
Tensor = torch.Tensor
OptTensor = Optional[torch.Tensor]


# -------------------------------------------------------------------------
#   Create Encoder Layer
# -------------------------------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model:int,
            n_heads:int,
            d_ff:int,
            f_dropout:float=0.0,
            preprocess:str='',
            postprocess:str='drn'):
        """
        """
        # Initialize Module
        super().__init__()

        # Create Layers
        self.pre_self_mha_proc = ProcessingLayer(preprocess, d_model, f_dropout)
        self.self_mha = MultiheadAttention(d_model, n_heads)
        self.post_self_mha_proc = ProcessingLayer(postprocess, d_model, f_dropout)

        self.pre_pwff_proc = ProcessingLayer(preprocess, d_model, f_dropout)
        self.pwff = PointwiseFeedforward(d_model, d_ff)
        self.post_pwff_proc = ProcessingLayer(postprocess, d_model, f_dropout)

    def forward(
            self,
            x:Tensor,
            mask:OptTensor=None) -> Tensor:
        """
        """
        # Apply MHA
        x_pre = self.pre_self_mha_proc(x)
        attn, _ = self.self_mha(x_pre, x_pre, x_pre, mask)
        x_attn = self.post_self_mha_proc(attn, x)

        # Apply PWFF
        x_pre = self.pre_pwff_proc(x_attn)
        pwff = self.pwff(x_pre)
        x_pwff = self.post_pwff_proc(pwff, x_attn)

        return x_pwff


# -------------------------------------------------------------------------
#   Create Decoder Layer
# -------------------------------------------------------------------------
class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model:int,
            n_heads:int,
            d_ff:int,
            f_dropout:float=0.0,
            preprocess:str='',
            postprocess:str='drn'):
        """
        """
        # Initialize Module
        super().__init__()

        # Create Layers
        self.pre_self_mha_proc = ProcessingLayer(preprocess, d_model, f_dropout)
        self.self_mha = MultiheadAttention(d_model, n_heads)
        self.post_self_mha_proc = ProcessingLayer(postprocess, d_model, f_dropout)

        self.pre_pwff_proc = ProcessingLayer(preprocess, d_model, f_dropout)
        self.pwff = PointwiseFeedforward(d_model, d_ff)
        self.post_pwff_proc = ProcessingLayer(postprocess, d_model, f_dropout)

    def forward(
            self,
            x:Tensor,
            mask:OptTensor=None) -> Tensor:
        """
        """
        # Apply MHA
        x_pre = self.pre_self_mha_proc(x)
        attn, _ = self.self_mha(x_pre, x_pre, x_pre, mask)
        x_attn = self.post_self_mha_proc(attn, x)

        # Apply PWFF
        x_pre = self.pre_pwff_proc(x_attn)
        pwff = self.pwff(x_pre)
        x_pwff = self.post_pwff_proc(pwff, x_attn)

        return x_pwff
