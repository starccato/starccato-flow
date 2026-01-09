# import torch
# import torch.nn as nn
# from torch import Tensor
# from ..utils.defaults import Y_LENGTH, HIDDEN_DIM

# class Flow(nn.Module):
#     def __init__(self, dim: int = 2, signal_dim: int = Y_LENGTH, h: int = HIDDEN_DIM):
#         super().__init__()
#         # Encode signal separately first
#         self.signal_encoder = nn.Sequential(
#             nn.Linear(signal_dim, h), nn.ELU(),
#             nn.Linear(h, h // 2), nn.ELU()
#         )
#         # Then combine with parameters
#         self.net = nn.Sequential(
#             nn.Linear(dim + 1 + h // 2, h), nn.ELU(),
#             nn.Linear(h, h), nn.ELU(),
#             nn.Linear(h, dim)
#         )
    
#     def forward(self, x_t: Tensor, t: Tensor, h: Tensor) -> Tensor:
#         h_encoded = self.signal_encoder(h)
#         return self.net(torch.cat((t, x_t, h_encoded), -1))
    
#     def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, h: Tensor) -> Tensor:
#         # Ensure t_start and t_end are on the same device as x_t
#         t_start = t_start.to(x_t.device).view(1, 1).expand(x_t.shape[0], 1)
#         t_end = t_end.to(x_t.device)
#         # TODO: implement another class with different ODE solvers
#         # Midpoint ODE solver, can use any other solver!
#         return x_t + (t_end - t_start) * self(
#             x_t + self(x_t, t_start, h) * (t_end - t_start) / 2,
#             t_start + (t_end - t_start) / 2,
#             h
#         )


import torch
import torch.nn as nn
from torch import Tensor
from ..utils.defaults import Y_LENGTH, HIDDEN_DIM

class Flow(nn.Module):
    def __init__(self, dim: int = 2, signal_dim: int = Y_LENGTH, h: int = HIDDEN_DIM):
        super().__init__()

        # Encode signal → FiLM parameters
        self.signal_encoder = nn.Sequential(
            nn.Linear(signal_dim, h),
            nn.ELU(),
            nn.Linear(h, 2 * h)  # gamma and beta
        )

        # Core vector field (no signal concatenation!)
        self.fc1 = nn.Linear(dim + 1, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, dim)

        self.act = nn.ELU()

    def forward(self, x_t: Tensor, t: Tensor, h: Tensor) -> Tensor:
        # Encode signal
        gamma, beta = self.signal_encoder(h).chunk(2, dim=-1)

        # Base input: parameters + time
        z = torch.cat((t, x_t), dim=-1)

        # Layer 1 with FiLM
        z = self.fc1(z)
        z = self.act(gamma * z + beta)

        # Layer 2 with FiLM
        z = self.fc2(z)
        z = self.act(gamma * z + beta)

        # Output
        return self.fc3(z)

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, h: Tensor) -> Tensor:
        t_start = t_start.to(x_t.device).view(1, 1).expand(x_t.shape[0], 1)
        t_end = t_end.to(x_t.device)

        dt = t_end - t_start

        k1 = self(x_t, t_start, h)
        k2 = self(x_t + 0.5 * dt * k1, t_start + 0.5 * dt, h)

        return x_t + dt * k2
