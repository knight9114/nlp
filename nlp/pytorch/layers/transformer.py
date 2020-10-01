# -------------------------------------------------------------------------
#   Transformer Layers
# -------------------------------------------------------------------------
# Imports
from typing import Optional, Tuple, List
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


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
            
# -------------------------------------------------------------------------
#   Create Encoder Block
# -------------------------------------------------------------------------
class TransformerEncoder(nn.Module):
    def __init__(
            self,
            n_layers:int,
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
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, n_heads, d_ff, f_dropout, preprocess, postprocess
            ) for _ in range(n_layers)
        ])

    def forward(
            self,
            x:Tensor,
            mask:OptTensor=None) -> Tensor:
        """
        """
        # Apply Encoder Layers
        for layer in self.layers:
            x = layer(x, mask)

        return x
        

# -------------------------------------------------------------------------
#   Create Decoder Block
# -------------------------------------------------------------------------
class TransformerDecoder(nn.Module):
    def __init__(
            self,
            n_layers:int,
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
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model, n_heads, d_ff, f_dropout, preprocess, postprocess
            ) for _ in range(n_layers)
        ])

    def forward(
            self,
            x:Tensor,
            memory:Tensor,
            enc_mask:OptTensor=None,
            dec_mask:OptTensor=None) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """
        """
        # Prepare
        self_attn_weights = []
        mem_attn_weights = []

        # Apply Decoder Layers
        for layer in self.layers:
            x, self_attn, mem_attn = layer(x, memory, enc_mask, dec_mask)
            self_attn_weights.append(self_attn)
            mem_attn_weights.append(mem_attn)

        return x, self_attn_weights, mem_attn_weights


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

        self.pre_mem_mha_proc = ProcessingLayer(preprocess, d_model, f_dropout)
        self.mem_mha = MultiheadAttention(d_model, n_heads)
        self.post_mem_mha_proc = ProcessingLayer(postprocess, d_model, f_dropout)

        self.pre_pwff_proc = ProcessingLayer(preprocess, d_model, f_dropout)
        self.pwff = PointwiseFeedforward(d_model, d_ff)
        self.post_pwff_proc = ProcessingLayer(postprocess, d_model, f_dropout)

    def forward(
            self,
            x:Tensor,
            memory:Tensor,
            enc_mask:OptTensor=None,
            dec_mask:OptTensor=None) -> Tuple[Tensor, Tensor, Tensor]: 
        """
        """
        # Apply Self MHA
        x_pre = self.pre_self_mha_proc(x)
        self_attn, self_attn_weight = self.self_mha(x_pre, x_pre, x_pre, dec_mask)
        x_self = self.post_self_mha_proc(self_attn, x)

        # Apply Memory MHA
        x_pre = self.pre_mem_mha_proc(x_self)
        mem_attn, mem_attn_weight = self.mem_mha(x_pre, memory, memory, enc_mask)
        x_mem = self.post_mem_mha_proc(mem_attn, x_self)

        # Apply PWFF
        x_pre = self.pre_pwff_proc(x_mem)
        pwff = self.pwff(x_pre)
        x_pwff = self.post_pwff_proc(pwff, x_mem)

        return x_pwff, self_attn_weight, mem_attn_weight


# -------------------------------------------------------------------------
#   Pointwise Feedforward Layer
# -------------------------------------------------------------------------
class PointwiseFeedforward(nn.Sequential):
    def __init__(
            self, 
            d_model:int, 
            d_ff:int):
        """
        """
        # Initialize Layer
        super().__init__(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
        )

# -------------------------------------------------------------------------
#   Multihead Attention Layer
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


# -------------------------------------------------------------------------
#   Processing Layer
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


# -------------------------------------------------------------------------
#   Positional Encoding Layer
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

        return torch.tensor(angle_rads[np.newaxis, ...], dtype=torch.float32)

    def forward(
            self,
            x:Tensor) -> Tensor:
        """
        """
        return (x * self.f_model) + self.pe[:, :x.shape[1], :]
