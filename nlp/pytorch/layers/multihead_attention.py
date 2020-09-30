# -------------------------------------------------------------------------
#   Multi-Head Attention
# -------------------------------------------------------------------------
# Imports
from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F


# -------------------------------------------------------------------------
#   Types
# -------------------------------------------------------------------------
OptTensor = Optional[torch.Tensor]
Tensor = torch.Tensor


# -------------------------------------------------------------------------
#   Create Layer
# -------------------------------------------------------------------------
class MultiheadAttention(nn.Module):
    def __init__(
            self, 
            d_model:int, 
            n_heads:int):
        """
        """
        # Assertions
        assert d_model % n_heads == 0

        # Initialize Layer
        super().__init__()

        # Define Attributes
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.f_model = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))

        # Create Layers
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.concat = nn.Linear(d_model, d_model)

    def split_heads(self, x:Tensor, batch_size:int) -> Tensor:
        """
        """
        return x.reshape([batch_size, -1, self.n_heads, self.d_head]).permute([0,2,1,3])

    def join_heads(self, x:Tensor, batch_size:int) -> Tensor:
        """
        """
        return x.permute([0,2,1,3]).reshape([batch_size, -1, self.d_model])

    def scaled_dot_product_attention(
            self, 
            q:Tensor, 
            k:Tensor, 
            v:Tensor, 
            mask:OptTensor=None) -> Tuple[Tensor, Tensor]:
        """
        """
        # Compute QK
        matmul_qk = torch.matmul(q, k.transpose(-1, -2))
        scaled_attn_logits = matmul_qk / self.f_model
        
        # Apply Mask
        if mask is not None:
            scaled_attn_logits += (mask * -1e9)

        # Get Probabilities
        attn_weights = F.softmax(scaled_attn_logits, dim=-1)

        # Compute Attention
        attn = torch.matmul(attn_weights, v)

        return attn, attn_weights

    def forward(
            self,
            q:Tensor,
            k:Tensor,
            v:Tensor,
            mask:OptTensor=None) -> Tuple[Tensor, Tensor]:
        """
        """
        # Prepare
        batch_size = q.shape[0]

        # Apply Linear Layers and Split
        q = self.split_heads(self.q_linear(q), batch_size)
        k = self.split_heads(self.k_linear(k), batch_size)
        v = self.split_heads(self.v_linear(v), batch_size)

        # Compute Attention
        scaled_attn, weights = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attn = self.join_heads(scaled_attn, batch_size)

        # Apply Concat Linear Layer
        concat_attn = self.concat(scaled_attn)

        return concat_attn, weights
