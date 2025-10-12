import torch
from torch import nn
from typing import Tuple

class VAE(nn.Module):
    """Fully connected (Vanilla) VAE (Variational Autoencoder) implementation in PyTorch.
    
    This VAE maps input signals to a latent space and back, using reparameterization
    to enable backpropagation through the sampling process.
    """
    
    def __init__(self, z_dim: int, hidden_dim: int, y_length: int) -> None:
        """Initialize the VAE.
        
        Args:
            z_dim: Dimension of the latent space
            hidden_dim: Dimension of hidden layers
            y_length: Length of input/output signal
        """
        super(VAE, self).__init__()
        self.encoder = Encoder(y_length=y_length, hidden_dim=hidden_dim, z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim, hidden_dim=hidden_dim, output_dim=y_length)
    
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
    
    def forward(self, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE.
        
        Args:
            d: Input tensor of shape (batch_size, y_length)
            
        Returns:
            Tuple containing:
                - d_hat: Reconstructed input
                - mean: Mean of the latent distribution
                - log_var: Log variance of the latent distribution
        """
        mean, log_var = self.encoder(d)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        d_hat = self.decoder(z)
        return d_hat, mean, log_var

class Decoder(nn.Module):
    """Decoder network that maps latent vectors back to signal space."""
    
    def __init__(self, z_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Initialize the decoder.
        
        Args:
            z_dim: Dimension of latent space input
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output signal
        """
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(z_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder.
        
        Args:
            z: Latent vector from encoder of shape (batch_size, z_dim)
            
        Returns:
            Reconstructed signal of shape (batch_size, output_dim)
        """
        h = self.LeakyReLU(self.FC_hidden(z))
        h = self.LeakyReLU(self.FC_hidden2(h))
        d_hat = self.FC_output(h)
        return d_hat

class Encoder(nn.Module):
    """Encoder network that maps input signals to latent space parameters."""
    
    def __init__(self, y_length: int, hidden_dim: int, z_dim: int) -> None:
        """Initialize the encoder.
        
        Args:
            y_length: Length of input signal
            hidden_dim: Dimension of hidden layers
            z_dim: Dimension of latent space
        """
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(y_length, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, z_dim)
        self.FC_var = nn.Linear(hidden_dim, z_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder.
        
        Args:
            d: Input signal of shape (batch_size, y_length)
            
        Returns:
            Tuple containing:
                - mean: Mean of latent distribution
                - log_var: Log variance of latent distribution
        """
        h_ = self.LeakyReLU(self.FC_input(d))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance 
        return mean, log_var