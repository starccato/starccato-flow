import torch
import torch.nn as nn
from torch import Tensor
from ..utils.defaults import Y_LENGTH, HIDDEN_DIM

class FlowFCL(nn.Module):
    """Fully Connected Layers version of Flow (original implementation)."""
    def __init__(self, dim: int = 8, signal_dim: int = 3 * Y_LENGTH, h: int = HIDDEN_DIM):
        super().__init__()
        # Encode signal separately first
        self.signal_encoder = nn.Sequential(
            nn.Linear(signal_dim, h), nn.GELU(),
            nn.Linear(h, h // 2), nn.GELU()
        )
        # Then combine with parameters
        self.net = nn.Sequential(
            nn.Linear(dim + 1 + h // 2, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, dim)
        )
    
    def forward(self, x_t: Tensor, t: Tensor, h: Tensor) -> Tensor:
        # Accept either flattened signals (B, 3*Y_LENGTH) or channel-first (B, 3, Y_LENGTH).
        if h.dim() == 3:
            h = h.view(h.size(0), -1)

        h_encoded = self.signal_encoder(h)
        return self.net(torch.cat((t, x_t, h_encoded), -1))
    
    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, h: Tensor) -> Tensor:
        # Ensure t_start and t_end are on the same device as x_t
        t_start = t_start.to(x_t.device).view(1, 1).expand(x_t.shape[0], 1)
        t_end = t_end.to(x_t.device).view(1, 1).expand(x_t.shape[0], 1)
        
        dt = t_end - t_start
        
        # RK4 (Runge-Kutta 4th order) ODE solver
        k1 = self(x_t, t_start, h)
        k2 = self(x_t + dt / 2 * k1, t_start + dt / 2, h)
        k3 = self(x_t + dt / 2 * k2, t_start + dt / 2, h)
        k4 = self(x_t + dt * k3, t_end, h)
        
        return x_t + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class FlowCNN(nn.Module):
    """Convolutional Neural Network version of Flow."""
    def __init__(self, dim: int = 8, signal_dim: int = 3 * Y_LENGTH, h: int = HIDDEN_DIM, num_channels: int = 3):
        super().__init__()
        self.num_channels = num_channels
        self.signal_length = signal_dim // num_channels
        
        # 1D Convolutional encoder for multi-channel signals
        self.signal_encoder = nn.Sequential(
            nn.Conv1d(num_channels, h // 2, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(h // 2, h // 4, kernel_size=3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        
        # Compute the flattened size after CNN
        cnn_out_dim = h // 4
        
        # Then combine with parameters
        self.net = nn.Sequential(
            nn.Linear(dim + 1 + cnn_out_dim, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, dim)
        )
    
    def forward(self, x_t: Tensor, t: Tensor, h: Tensor) -> Tensor:
        # Reshape to channel-first format (B, C, L) for Conv1d
        if h.dim() == 2:
            # If flattened (B, 3*Y_LENGTH), reshape to (B, 3, Y_LENGTH)
            h = h.view(h.size(0), self.num_channels, self.signal_length)
        
        # h should now be (B, C, L) for Conv1d
        h_encoded = self.signal_encoder(h).view(h.size(0), -1)  # Flatten after pooling
        return self.net(torch.cat((t, x_t, h_encoded), -1))
    
    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, h: Tensor) -> Tensor:
        # Ensure t_start and t_end are on the same device as x_t
        t_start = t_start.to(x_t.device).view(1, 1).expand(x_t.shape[0], 1)
        t_end = t_end.to(x_t.device).view(1, 1).expand(x_t.shape[0], 1)
        
        dt = t_end - t_start
        
        # RK4 (Runge-Kutta 4th order) ODE solver
        k1 = self(x_t, t_start, h)
        k2 = self(x_t + dt / 2 * k1, t_start + dt / 2, h)
        k3 = self(x_t + dt / 2 * k2, t_start + dt / 2, h)
        k4 = self(x_t + dt * k3, t_end, h)
        
        return x_t + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# Default to FCL for backward compatibility
class Flow(FlowFCL):
    """Default Flow class uses FCL implementation."""
    pass


# slightly older version kept for reference

# import torch
# import torch.nn as nn
# from torch import Tensor
# from ..utils.defaults import Y_LENGTH, HIDDEN_DIM

# class Flow(nn.Module):
#     def __init__(self, dim: int = 2, signal_dim: int = Y_LENGTH, h: int = HIDDEN_DIM):
#         super().__init__()

#         # Encode signal → FiLM parameters
#         self.signal_encoder = nn.Sequential(
#             nn.Linear(signal_dim, h),
#             nn.ELU(),
#             nn.Linear(h, 2 * h)  # gamma and beta
#         )

#         # Core vector field (no signal concatenation!)
#         self.fc1 = nn.Linear(dim + 1, h)
#         self.fc2 = nn.Linear(h, h)
#         self.fc3 = nn.Linear(h, dim)

#         self.act = nn.ELU()

#     def forward(self, x_t: Tensor, t: Tensor, h: Tensor) -> Tensor:
#         # Encode signal
#         gamma, beta = self.signal_encoder(h).chunk(2, dim=-1)

#         # Base input: parameters + time
#         z = torch.cat((t, x_t), dim=-1)

#         # Layer 1 with FiLM
#         z = self.fc1(z)
#         z = self.act(gamma * z + beta)

#         # Layer 2 with FiLM
#         z = self.fc2(z)
#         z = self.act(gamma * z + beta)

#         # Output
#         return self.fc3(z)

#     def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, h: Tensor) -> Tensor:
#         t_start = t_start.to(x_t.device).view(1, 1).expand(x_t.shape[0], 1)
#         t_end = t_end.to(x_t.device)

#         dt = t_end - t_start

#         k1 = self(x_t, t_start, h)
#         k2 = self(x_t + 0.5 * dt * k1, t_start + 0.5 * dt, h)

#         return x_t + dt * k2
