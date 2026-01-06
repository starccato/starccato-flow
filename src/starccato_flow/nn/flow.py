import torch
import torch.nn as nn
from torch import Tensor
from ..utils.defaults import SIGNAL_LENGTH, HIDDEN_DIM

class Flow(nn.Module):
    def __init__(self, dim: int = 2, signal_dim: int = SIGNAL_LENGTH, h: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1 + signal_dim, h), nn.ELU(),  # parameters + time + signal
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim))  # outputs parameter velocity
    
    def forward(self, x_t: Tensor, t: Tensor, h: Tensor) -> Tensor:
        # x_t: current parameter estimate, t: time, h: input signal
        return self.net(torch.cat((t, x_t, h), -1))
    
    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, h: Tensor) -> Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        # TODO: implement another class with different ODE solvers
        # Midpoint ODE solver, can use any other solver!
        return x_t + (t_end - t_start) * self(
            x_t + self(x_t, t_start, h) * (t_end - t_start) / 2,
            t_start + (t_end - t_start) / 2,
            h
        )