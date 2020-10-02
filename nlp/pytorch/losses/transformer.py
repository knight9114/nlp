# -------------------------------------------------------------------------
#   Transformer Losses
# -------------------------------------------------------------------------
# Imports
import torch
from torch.nn import functional as F


# -------------------------------------------------------------------------
#   Globals
# -------------------------------------------------------------------------
Tensor = torch.Tensor


# -------------------------------------------------------------------------
#   Smooth Cross-Entropy Loss
# -------------------------------------------------------------------------
def cross_entropy(
        targets:Tensor,
        x:Tensor,
        smoothing:float=0.,
        from_logits:bool=True) -> Tensor:
    """
    """
    # Label Smoothing
    n_vocab = x.shape[-1]
    y = F.one_hot(targets, n_vocab).type(torch.float32)
    y = (1. - smoothing) * y + (smoothing / n_vocab) * y

    # Apply Softmax
    if from_logits:
        x = F.log_softmax(x, -1)
    else:
        x = torch.log(x)

    # Compute Loss
    loss = F.kl_div(
            input=x,
            target=y,
            reduction='batchmean')

    return loss
