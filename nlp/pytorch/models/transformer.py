# -------------------------------------------------------------------------
#   Transformer Model
# -------------------------------------------------------------------------
# Imports
from typing import Optional, Tuple, List
import torch
from torch import nn

from nlp.pytorch.layers import (
    TransformerEncoder,
    TransformerDecoder,
    PositionalEncoding
)


# -------------------------------------------------------------------------
#   Globals
# -------------------------------------------------------------------------
Tensor = torch.Tensor
OptTensor = Optional[torch.Tensor]


# -------------------------------------------------------------------------
#   Create Transformer
# -------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(
            self,
            n_src_vocab:int,
            n_tgt_vocab:int,
            n_layers:int,
            d_model:int,
            n_heads:int,
            d_ff:int,
            f_dropout:float=0.0,
            preprocess:str='',
            postprocess:str='drn',
            max_src_len:int=128,
            max_tgt_len:int=128):
        """
        """
        # Initialize Module
        super().__init__()

        # Create Embedding Layers
        self.src_token_embedding = nn.Embedding(n_src_vocab, d_model)
        self.src_positional_embedding = PositionalEncoding(d_model, max_src_len)
        self.tgt_token_embedding = nn.Embedding(n_tgt_vocab, d_model)
        self.tgt_positional_embedding = PositionalEncoding(d_model, max_tgt_len)

        # Create Encoder
        self.encoder = TransformerEncoder(
                n_layers, 
                d_model, 
                n_heads,
                d_ff,
                f_dropout,
                preprocess,
                postprocess)

        # Create Decoder
        self.decoder = TransformerDecoder(
                n_layers, 
                d_model, 
                n_heads,
                d_ff,
                f_dropout,
                preprocess,
                postprocess)

        # Create Projector
        self.projector = nn.Linear(d_model, n_tgt_vocab)

        # Create Dropout Layer
        self.dropout = nn.Dropout(f_dropout)

    def forward(
            self,
            src:Tensor,
            tgt:Tensor,
            enc_mask:OptTensor=None,
            dec_mask:OptTensor=None) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """
        """
        # Embed Inputs
        x_src = self.dropout(self.src_positional_embedding(self.src_token_embedding(src)))
        x_tgt = self.dropout(self.tgt_positional_embedding(self.tgt_token_embedding(tgt)))

        # Apply Encoder
        memory = self.encoder(x_src, enc_mask)

        # Apply Decoder
        decoded, self_attn, mem_attn = self.decoder(x_tgt, memory, enc_mask, dec_mask)

        # Apply Projector
        logits = self.projector(decoded)

        return logits, self_attn, mem_attn

