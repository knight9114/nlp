# -------------------------------------------------------------------------
#   Point-Wise Feed-Forward
# -------------------------------------------------------------------------
# Imports
from torch import nn


# -------------------------------------------------------------------------
#   Create Layer
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
