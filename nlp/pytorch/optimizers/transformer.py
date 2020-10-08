# -------------------------------------------------------------------------
#   Transformer Optimizer
# -------------------------------------------------------------------------
# Imports
from typing import Iterable, Tuple, Optional, Callable
import torch
from torch import optim, nn


# -------------------------------------------------------------------------
#   Globals
# -------------------------------------------------------------------------
Tensor = torch.Tensor
OptClosure = Callable[..., Tensor]


# -------------------------------------------------------------------------
#   Create Optimizer
# -------------------------------------------------------------------------
class NoamOptimizer(optim.AdamW):
    def __init__(
            self,
            params:Iterable[Tensor],
            d_model:int,
            n_warmup_steps:int,
            betas:Tuple[float, float]=(.9, .98),
            epsilon:float=1e-9,
            decay:float=0.01,
            amsgrad:bool=False):
        """
        """
        # Define Attributes
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.f_model_rsqrt = d_model ** -0.5
        self.f_warmup = n_warmup_steps ** -1.5

        # Initialize Optimizer
        super().__init__(params, 1., betas, epsilon, decay, amsgrad)

        # Create Learning Rate Schedule
        self.schedule = optim.lr_scheduler.LambdaLR(
                optimizer=self,
                lr_lambda=self.create_schedule())

    def step_learning_rate(self):
        """
        """
        self.schedule.step()

    def create_schedule(self) -> Callable[[int], float]:
        """
        """
        # Define Closure
        def schedule(i:int) -> float:
            # Compute Pre/Post Values
            pre = (i + 1) ** -0.5
            post = i * self.f_warmup

            return self.f_model_rsqrt * min(pre, post)

        return schedule
