import torch
import torch.nn as nn
from torch import Tensor
from ..utils.defaults_general import Y_LENGTH, HIDDEN_DIM

class FlowFCL(nn.Module):
    """Fully Connected Layers version of Flow with batch normalization and deeper architecture.
    
    Improvements:
    - Batch normalization after each layer for training stability
    - Deeper network with more hidden layers
    - Better capacity for learning complex signal-parameter mappings
    """
    def __init__(self, dim: int = 8, signal_dim: int = 3 * Y_LENGTH, h: int = HIDDEN_DIM):
        super().__init__()
        # Deeper signal encoder with batch normalization
        self.signal_fc1 = nn.Linear(signal_dim, h)
        self.signal_bn1 = nn.BatchNorm1d(h)
        
        self.signal_fc2 = nn.Linear(h, h)
        self.signal_bn2 = nn.BatchNorm1d(h)
        
        self.signal_fc3 = nn.Linear(h, h // 2)
        self.signal_bn3 = nn.BatchNorm1d(h // 2)
        
        # Deeper main network with batch normalization
        self.fc1 = nn.Linear(dim + 1 + h // 2, h)
        self.bn1 = nn.BatchNorm1d(h)
        
        self.fc2 = nn.Linear(h, h)
        self.bn2 = nn.BatchNorm1d(h)
        
        self.fc3 = nn.Linear(h, h)
        self.bn3 = nn.BatchNorm1d(h)
        
        self.fc4 = nn.Linear(h, h // 2)
        self.bn4 = nn.BatchNorm1d(h // 2)
        
        self.fc5 = nn.Linear(h // 2, dim)
        
        self.act = nn.GELU()
    
    def forward(self, x_t: Tensor, t: Tensor, h: Tensor) -> Tensor:
        # Accept either flattened signals (B, 3*Y_LENGTH) or channel-first (B, 3, Y_LENGTH).
        if h.dim() == 3:
            h = h.view(h.size(0), -1)

        # Deeper signal encoder with batch norm
        h_encoded = self.signal_fc1(h)
        h_encoded = self.signal_bn1(h_encoded)
        h_encoded = self.act(h_encoded)
        
        h_encoded = self.signal_fc2(h_encoded)
        h_encoded = self.signal_bn2(h_encoded)
        h_encoded = self.act(h_encoded)
        
        h_encoded = self.signal_fc3(h_encoded)
        h_encoded = self.signal_bn3(h_encoded)
        h_encoded = self.act(h_encoded)
        
        # Deeper main network with batch norm
        combined = torch.cat((t, x_t, h_encoded), -1)
        
        out = self.fc1(combined)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.act(out)
        
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.act(out)
        
        out = self.fc4(out)
        out = self.bn4(out)
        out = self.act(out)
        
        out = self.fc5(out)
        
        return out
    
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


# class FlowCNN(nn.Module):
#     """Convolutional Neural Network version of Flow with batch normalization and deeper architecture.
#     
#     Improvements:
#     - Batch normalization after each layer for training stability
#     - Deeper convolutional encoder
#     - Deeper fully connected network
#     """
#     def __init__(self, dim: int = 8, signal_dim: int = 3 * Y_LENGTH, h: int = HIDDEN_DIM, num_channels: int = 3):
#         super().__init__()
#         self.num_channels = num_channels
#         self.signal_length = signal_dim // num_channels
        
#         # Deeper 1D Convolutional encoder with batch normalization
#         self.conv1 = nn.Conv1d(num_channels, h // 2, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(h // 2)
        
#         self.conv2 = nn.Conv1d(h // 2, h // 2, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(h // 2)
        
#         self.conv3 = nn.Conv1d(h // 2, h // 4, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm1d(h // 4)
#         
#         self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
#         cnn_out_dim = h // 4
        
#         # Deeper main network with batch normalization
#         self.fc1 = nn.Linear(dim + 1 + cnn_out_dim, h)
#         self.bn_fc1 = nn.BatchNorm1d(h)
        
#         self.fc2 = nn.Linear(h, h)
#         self.bn_fc2 = nn.BatchNorm1d(h)
        
#         self.fc3 = nn.Linear(h, h // 2)
#         self.bn_fc3 = nn.BatchNorm1d(h // 2)
        
#         self.fc4 = nn.Linear(h // 2, dim)
        
#         self.act = nn.GELU()
    
#     def forward(self, x_t: Tensor, t: Tensor, h: Tensor) -> Tensor:
#         # Reshape to channel-first format (B, C, L) for Conv1d
#         if h.dim() == 2:
#             # If flattened (B, 3*Y_LENGTH), reshape to (B, 3, Y_LENGTH)
#             h = h.view(h.size(0), self.num_channels, self.signal_length)
        
#         # Deeper conv encoder with batch norm
#         h = self.conv1(h)
#         h = self.bn1(h)
#         h = self.act(h)
        
#         h = self.conv2(h)
#         h = self.bn2(h)
#         h = self.act(h)
        
#         h = self.conv3(h)
#         h = self.bn3(h)
#         h = self.act(h)
        
#         h = self.pool(h)
#         h_encoded = h.view(h.size(0), -1)
        
#         # Deeper main network with batch norm
#         combined = torch.cat((t, x_t, h_encoded), -1)
        
#         out = self.fc1(combined)
#         out = self.bn_fc1(out)
#         out = self.act(out)
        
#         out = self.fc2(out)
#         out = self.bn_fc2(out)
#         out = self.act(out)
        
#         out = self.fc3(out)
#         out = self.bn_fc3(out)
#         out = self.act(out)
        
#         out = self.fc4(out)
        
#         return out
    
#     def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, h: Tensor) -> Tensor:
#         # Ensure t_start and t_end are on the same device as x_t
#         t_start = t_start.to(x_t.device).view(1, 1).expand(x_t.shape[0], 1)
#         t_end = t_end.to(x_t.device).view(1, 1).expand(x_t.shape[0], 1)
        
#         dt = t_end - t_start
        
#         # RK4 (Runge-Kutta 4th order) ODE solver
#         k1 = self(x_t, t_start, h)
#         k2 = self(x_t + dt / 2 * k1, t_start + dt / 2, h)
#         k3 = self(x_t + dt / 2 * k2, t_start + dt / 2, h)
#         k4 = self(x_t + dt * k3, t_end, h)
        
#         return x_t + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


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
