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


