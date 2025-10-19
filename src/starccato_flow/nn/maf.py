import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from starccato_flow.utils.defaults import DEVICE

class MaskedLinear(nn.Linear):
    """
    A masked linear layer ensures that each neuron can only depend on certain input dimensions, enforcing the autoregressive property. It turns a normal MLP into an autoregressive network.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        # mask will be set later. this will be fixed
        self.register_buffer('mask', torch.ones(out_features, in_features))
    def set_mask(self, mask):
        # mask: [out_features, in_features]
        assert mask.shape == (self.out_features, self.in_features)
        self.mask.data.copy_(mask)
    def forward(self, x):
        # weight matrix multiplication acts as a mask
        return F.linear(x, self.weight * self.mask, self.bias)

class MADE(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512,512], num_outputs_per_dim=2, natural_ordering=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_outputs_per_dim = num_outputs_per_dim
        
        # Create network layers
        hs = [input_dim] + hidden_dims + [input_dim * num_outputs_per_dim]
        self.net = nn.ModuleList([MaskedLinear(hs[i], hs[i+1]) for i in range(len(hs)-1)])
        self.natural_ordering = natural_ordering
        self.create_masks()

    def create_masks(self):
        rng = np.random.RandomState(0)
        
        # Calculate degrees for each layer
        L = len(self.net)
        degrees = []
        
        # Input layer degrees
        degrees.append(np.arange(1, self.input_dim + 1))
        
        # Hidden layer degrees
        for i in range(L - 1):
            min_prev_degree = min(degrees[-1])
            max_prev_degree = max(degrees[-1])
            layer_degrees = rng.randint(min_prev_degree, max_prev_degree + 1, 
                                      size=self.hidden_dims[i])
            degrees.append(layer_degrees)
        
        # Output layer degrees
        degrees.append(np.arange(1, self.input_dim + 1).repeat(self.num_outputs_per_dim))
        
        # Create masks
        masks = []
        for l in range(L):
            in_degrees = degrees[l]
            out_degrees = degrees[l+1]
            mask = (out_degrees[:, None] >= in_degrees[None, :]).astype(np.float32)
            mask = torch.from_numpy(mask)
            masks.append(mask)
            
        # Set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.net[:-1]):
            h = F.relu(layer(h))
        out = self.net[-1](h)  # shape [B, D * outputs_per_dim]
        out = out.view(x.shape[0], self.input_dim, self.num_outputs_per_dim)
        # return per-dim parameters
        return out  # [..., dim, 2]  e.g. mu, log_scale

class MAFLayer(nn.Module):
    def __init__(self, dim, hidden_dims=[512,512], permute=True):
        super().__init__()
        self.dim = dim
        self.autoreg = MADE(dim, hidden_dims=hidden_dims, num_outputs_per_dim=2, natural_ordering=True)
        self.permute = permute
        if permute:
            # a fixed random permutation; we can also learn or alternate reverse
            self.register_buffer('perm', torch.tensor(np.random.permutation(dim), dtype=torch.long))
            inv = np.argsort(self.perm.cpu().numpy())
            self.register_buffer('inv_perm', torch.tensor(inv, dtype=torch.long))

    def forward(self, x):
        """
        x -> z, return (z, log_det)
        """
        if self.permute:
            x = x[:, self.perm]  # permute dims
        params = self.autoreg(x)  # [B, D, 2]
        mu = params[..., 0]
        log_a = params[..., 1]  # log scale 'a'
        # Stability: clamp log_a
        log_a = torch.clamp(log_a, min=-3.0, max=3.0)
        z = (x - mu) * torch.exp(-log_a)
        # log det jacobian: -sum a
        log_det = -torch.sum(log_a, dim=1)
        if self.permute:
            # invert permutation on z so next layer sees canonical order (optional)
            z = z[:, self.inv_perm]
        return z, log_det

    def inverse(self, z):
        """
        Sample x given z (sequential because autoregressive).
        We need to reconstruct x dimension by dimension.
        """
        if self.permute:
            z = z[:, self.perm]  # apply same perm as forward (consistent)
        B, D = z.shape
        x = torch.zeros_like(z)
        for i in range(D):
            params = self.autoreg(x)  # note: MADE uses masked connections so outputs for dims > i may be computed but they don't depend on future x
            mu = params[:, i, 0]
            log_a = params[:, i, 1]
            log_a = torch.clamp(log_a, min=-3.0, max=3.0)
            x[:, i] = mu + torch.exp(log_a) * z[:, i]
        if self.permute:
            x = x[:, self.inv_perm]
        return x

class MaskedAutoregressiveFlow(nn.Module):
    def __init__(self, dim, n_layers=5, hidden_dims=[4,4]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # Alternate permutation: True for even layers, False for odd layers
            use_permute = (i % 2 == 0)
            self.layers.append(MAFLayer(dim, hidden_dims=hidden_dims, permute=use_permute))

        # base distribution: standard normal
        self.register_buffer('base_mu', torch.zeros(dim))
        self.register_buffer('base_var', torch.ones(dim))

    def forward(self, x):
        log_det_total = 0.0
        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total += log_det
        # log p_x = log p_z + log_det
        log_pz = -0.5 * torch.sum(z**2 + torch.log(2*torch.pi*torch.ones_like(z)), dim=1)
        log_px = log_pz + log_det_total
        return z, log_px

    def sample(self, num_samples, device=DEVICE):
        z = torch.randn(num_samples, self.base_mu.shape[0], device=device)
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x
