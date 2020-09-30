# -------------------------------------------------------------------------
#   Processing Layer
# -------------------------------------------------------------------------
# Imports
from typing import Optional
import torch
from torch import nn


# -------------------------------------------------------------------------
#   Globals
# -------------------------------------------------------------------------
Tensor = torch.Tensor
OptTensor = Optional[torch.Tensor]


# -------------------------------------------------------------------------
#   Create Layer
# -------------------------------------------------------------------------
class ProcessingLayer(nn.Module):
    def __init__(
            self,
            seq:str,
            d_model:int,
            f_dropout:float):
        """
        """
        # Initialize Module
        super().__init__()

        # Define Attributes
        self.d_model = d_model
        self.seq = seq

        # Create Layers
        if 'n' in self.seq:
            self.norm = nn.LayerNorm(d_model)

        if 'd' in self.seq:
            self.dropout = nn.Dropout(f_dropout)

    def forward(
            self,
            x:Tensor,
            residual:OptTensor=None) -> Tensor:
        """
        """
        # Shortcut
        if self.seq == '':
            return x

        # Apply Sequence
        for cmd in self.seq:
            if cmd == 'd':
                x = self.dropout(x)

            elif cmd == 'r':
                assert residual is not None
                x += residual

            elif cmd == 'n':
                x = self.norm(x)

        return x
