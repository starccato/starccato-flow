import torch
from torch import nn
from typing import Tuple

class ConditionalVAE(nn.Module):
    """Conditional Variational Autoencoder (CVAE) implementation in PyTorch.
    
    This CVAE conditions both encoding and decoding on physical parameters,
    enabling controlled generation of signals with specific parameter values.
    """
    
    def __init__(self, z_dim: int, hidden_dim: int, y_length: int, param_dim: int) -> None:
        """Initialize the Conditional VAE.
        
        Args:
            z_dim: Dimension of the latent space
            hidden_dim: Dimension of hidden layers
            y_length: Length of input/output signal
            param_dim: Dimension of conditioning parameters (e.g., 1 for beta, 4 for beta+omega+A+Ye)
        """
        super(ConditionalVAE, self).__init__()
        self.encoder = ConditionalEncoder(y_length=y_length, hidden_dim=hidden_dim, 
                                         z_dim=z_dim, param_dim=param_dim)
        self.decoder = ConditionalDecoder(z_dim=z_dim, hidden_dim=hidden_dim, 
                                         output_dim=y_length, param_dim=param_dim)
        self.param_dim = param_dim
        self.z_dim = z_dim
    
    @staticmethod
    def reparameterization(mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Apply the reparameterization trick to sample from a Gaussian.
        
        Args:
            mean: Mean of the Gaussian distribution
            var: Variance of the Gaussian distribution
            
        Returns:
            A sample from the Gaussian distribution
        """
        epsilon = torch.randn_like(var)  # sampling epsilon        
        z = mean + var * epsilon  # reparameterization trick
        return z
    
    def forward(self, d: torch.Tensor, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the Conditional VAE.
        
        Args:
            d: Input tensor of shape (batch_size, y_length)
            params: Conditioning parameters of shape (batch_size, param_dim)
            
        Returns:
            Tuple containing:
                - d_hat: Reconstructed input
                - mean: Mean of the latent distribution
                - log_var: Log variance of the latent distribution
        """
        mean, log_var = self.encoder(d, params)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        d_hat = self.decoder(z, params)
        return d_hat, mean, log_var

class ConditionalDecoder(nn.Module):
    """Conditional decoder network that maps latent vectors + parameters back to signal space."""
    
    def __init__(self, z_dim: int, hidden_dim: int, output_dim: int, param_dim: int) -> None:
        """Initialize the conditional decoder.
        
        Args:
            z_dim: Dimension of latent space input
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output signal
            param_dim: Dimension of conditioning parameters
        """
        super(ConditionalDecoder, self).__init__()
        
        # AGGRESSIVE parameter embedding - make parameters MORE important than latent
        # Use 2x latent dimension to force decoder to pay attention to params
        param_embed_dim = z_dim * 2
        self.param_embed = nn.Sequential(
            nn.Linear(param_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, param_embed_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Combine latent vector and embedded parameters
        self.FC_hidden = nn.Linear(z_dim + param_embed_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        # Dropout on latent to force decoder to rely on parameters
        self.z_dropout = nn.Dropout(0.2)
        
    def forward(self, z: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Forward pass through conditional decoder.
        
        Args:
            z: Latent vector from encoder of shape (batch_size, z_dim)
            params: Conditioning parameters of shape (batch_size, param_dim)
            
        Returns:
            Reconstructed signal of shape (batch_size, output_dim)
        """
        # Embed parameters to match latent dimension
        params_embedded = self.param_embed(params)
        
        # Apply dropout to latent during training to force use of parameters
        if self.training:
            z = self.z_dropout(z)
        
        # Concatenate latent vector with embedded parameters
        z_params = torch.cat([z, params_embedded], dim=-1)
        
        h = self.LeakyReLU(self.FC_hidden(z_params))
        h = self.LeakyReLU(self.FC_hidden2(h))
        d_hat = self.FC_output(h)
        return d_hat

class ConditionalEncoder(nn.Module):
    """Conditional encoder network that maps input signals + parameters to latent space parameters."""
    
    def __init__(self, y_length: int, hidden_dim: int, z_dim: int, param_dim: int) -> None:
        """Initialize the conditional encoder.
        
        Args:
            y_length: Length of input signal
            hidden_dim: Dimension of hidden layers
            z_dim: Dimension of latent space
            param_dim: Dimension of conditioning parameters
        """
        super(ConditionalEncoder, self).__init__()
        
        # Stronger parameter embedding in encoder
        # Make params equally important as signal features
        self.param_embed = nn.Sequential(
            nn.Linear(param_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Signal processing network
        self.signal_fc = nn.Sequential(
            nn.Linear(y_length, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Combine signal features and embedded parameters (now both hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, z_dim)
        self.FC_var = nn.Linear(hidden_dim, z_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        # Dropout on latent to prevent encoding everything in z
        self.latent_dropout = nn.Dropout(0.1)
        
    def forward(self, d: torch.Tensor, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through conditional encoder.
        
        Args:
            d: Input signal of shape (batch_size, y_length)
            params: Conditioning parameters of shape (batch_size, param_dim)
            
        Returns:
            Tuple containing:
                - mean: Mean of latent distribution
                - log_var: Log variance of latent distribution
        """
        # Process signal and parameters separately first
        signal_features = self.signal_fc(d)
        params_embedded = self.param_embed(params)
        
        # Concatenate processed features
        combined = torch.cat([signal_features, params_embedded], dim=-1)
        
        h_ = self.LeakyReLU(self.FC_input2(combined))
        
        # Apply dropout during training to prevent z from encoding everything
        if self.training:
            h_ = self.latent_dropout(h_)
        
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance 
        return mean, log_var
