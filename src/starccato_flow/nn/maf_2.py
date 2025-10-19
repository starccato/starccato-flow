from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import ReLU
from starccato_flow.utils.defaults import DEVICE


class MaskedLinear(nn.Linear):
    """Linear transformation with masked elements. y = x.dot(mask*W.T) + b"""

    def __init__(self, n_in: int, n_out: int, bias: bool = True) -> None:
        super().__init__(n_in, n_out, bias)
        self.mask = None

    def initialise_mask(self, mask: Tensor):
        """Set mask tensor."""
        self.mask = mask.to(DEVICE)

    def forward(self, x: Tensor) -> Tensor:
        """Apply masked linear transformation."""
        print(self.mask.device)
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(
        self,
        n_in: int,
        hidden_dims: List[int],
        gaussian: bool = False,
        random_order: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """Initialise MADE model."""
        super().__init__()
        np.random.seed(seed)
        self.n_in = n_in
        self.n_out = 2 * n_in if gaussian else n_in
        self.hidden_dims = hidden_dims
        self.random_order = random_order
        self.gaussian = gaussian
        self.masks = {}
        self.mask_matrix = []
        self.layers = []

        # Build layer sizes list
        dim_list = [self.n_in, *hidden_dims, self.n_out]

        # Construct layers
        for i in range(len(dim_list) - 2):
            self.layers.append(MaskedLinear(dim_list[i], dim_list[i + 1]))
            self.layers.append(ReLU())
        self.layers.append(MaskedLinear(dim_list[-2], dim_list[-1]))

        self.model = nn.Sequential(*self.layers)

        # Create masks
        self._create_masks()

    def forward(self, x: Tensor) -> Tensor:
        if self.gaussian:
            return self.model(x)
        else:
            return torch.sigmoid(self.model(x))

    def _create_masks(self) -> None:
        """Create masks for the hidden layers."""
        L = len(self.hidden_dims)
        D = self.n_in

        # Input ordering
        self.masks[0] = np.random.permutation(D) if self.random_order else np.arange(D)

        # Hidden layers
        for l in range(L):
            low = self.masks[l].min()
            size = self.hidden_dims[l]
            self.masks[l + 1] = np.random.randint(low=low, high=D, size=size)

        # Output layer
        self.masks[L + 1] = self.masks[0]

        # Create mask matrices
        self.mask_matrix = []
        for i in range(len(self.masks) - 1):
            m, m_next = self.masks[i], self.masks[i + 1]
            M = torch.from_numpy((m_next[:, None] >= m[None, :]).astype(np.float32)).to(DEVICE)
            self.mask_matrix.append(M)

        # Double for Gaussian outputs
        if self.gaussian:
            m = self.mask_matrix.pop(-1)
            self.mask_matrix.append(torch.cat((m, m), dim=0))

        # Initialise MaskedLinear layers
        mask_iter = iter(self.mask_matrix)
        for module in self.model.modules():
            if isinstance(module, MaskedLinear):
                module.initialise_mask(next(mask_iter))


class MAFLayer(nn.Module):
    def __init__(self, dim: int, hidden_dims: List[int], reverse: bool = False):
        super().__init__()
        self.dim = dim
        self.made = MADE(dim, hidden_dims, gaussian=True)
        self.reverse = reverse

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.made(x.float())
        mu, logp = torch.chunk(out, 2, dim=1)
        u = (x - mu) * torch.exp(0.5 * logp)
        if self.reverse:
            u = u.flip(dims=(1,))
        log_det = 0.5 * torch.sum(logp, dim=1)
        return u, log_det

    def backward(self, u: Tensor) -> Tuple[Tensor, Tensor]:
        if self.reverse:
            u = u.flip(dims=(1,))
        x = torch.zeros_like(u)
        for dim in range(self.dim):
            out = self.made(x)
            mu, logp = torch.chunk(out, 2, dim=1)
            mod_logp = torch.clamp(-0.5 * logp, max=10)
            x[:, dim] = mu[:, dim] + u[:, dim] * torch.exp(mod_logp[:, dim])
        log_det = torch.sum(mod_logp, dim=1)
        return x, log_det


class MAF(nn.Module):
    def __init__(self, dim: int, n_layers: int, hidden_dims: List[int], use_reverse: bool = True):
        super().__init__()
        self.dim = dim
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(MAFLayer(dim, hidden_dims, reverse=use_reverse))
            self.layers.append(nn.BatchNorm1d(dim))  # 1D batch norm for 2D inputs

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        log_det_sum = torch.zeros(x.shape[0], device=DEVICE)
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_sum += log_det
        return x, log_det_sum

    def backward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        log_det_sum = torch.zeros(x.shape[0], device=DEVICE)
        for layer in reversed(self.layers):
            x, log_det = layer.backward(x)
            log_det_sum += log_det
        return x, log_det_sum
