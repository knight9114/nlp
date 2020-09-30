# -------------------------------------------------------------------------
#   Positional Encoding
# -------------------------------------------------------------------------
# Imports
import numpy as np
import torch
from torch import nn


# -------------------------------------------------------------------------
#   Globals
# -------------------------------------------------------------------------
Tensor = torch.Tensor


# -------------------------------------------------------------------------
#   Create Layer
# -------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(
            self,
            d_model:int,
            max_seq_len:int):
        """
        """
        # Initialize Module
        super().__init__()

        # Define Attributes
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.f_model = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))

        # Create Encoding
        self.register_buffer('pe', self.create_encoding())

    def create_encoding(self):
        """
        """
        # Get Angles
        index = np.arange(self.d_model)[np.newaxis, :]
        positions = np.arange(self.max_seq_len)[:, np.newaxis]
        angle_rates = 1 / np.power(10000, (2 * (index // 2)) / np.float32(self.d_model))
        angle_rads = positions * angle_rates

        # Apply Sin and Cos
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        return torch.tensor(angle_rads[np.newaxis, ...])

    def forward(
            self,
            x:Tensor) -> Tensor:
        """
        """
        return (x * self.f_model) + self.pe[:, :x.shape[1], :]
