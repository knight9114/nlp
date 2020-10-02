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
class NoamOptimizer(optim.Optimizer):
    def __init__(
            self,
            params:Iterable[Tensor],
            d_model:int,
            n_warmup_steps:int,
            betas:Tuple[float, float]=(.9, .98),
            epsilon:float=1e-9,
            decay:float=0.01):
        """
        """
        # Prepare
        param_list = nn.ParameterList(params)

        # Define Attributes
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.f_model_rsqrt = d_model ** -0.5
        self.f_warmup = n_warmup_steps ** -1.5

        # Create Optimizer
        self.optimizer = optim.AdamW(
                params=param_list,
                lr=1.,
                betas=betas,
                eps=epsilon,
                weight_decay=decay)

        # Create Learning Rate Schedule
        self.schedule = optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=self.create_schedule())

        # Initialize Optimizer
        super().__init__(param_list, {})

    def step(
            self,
            closure:OptClosure=None):
        """
        """
        # Step Optimizer
        self.optimizer.step()

        # Step Schedule
        self.schedule.step()

    def zero_grad(self):
        """
        """
        self.optimizer.zero_grad()

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
