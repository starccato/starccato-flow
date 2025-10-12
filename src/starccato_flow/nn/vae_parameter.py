from .vae import VAE, Encoder, Decoder
import torch
import torch.nn as nn

class VAE_PARAMETER(VAE):
    """VAE variant that maps signals to parameters instead of reconstructing signals"""
    
    def __init__(self, z_dim, hidden_dim, y_length, param_dim):
        """Initialize the parameter VAE.
        
        Args:
            z_dim (int): Dimension of latent space
            hidden_dim (int): Dimension of hidden layers
            y_length (int): Length of input signal
            param_dim (int): Dimension of parameter space (output)
        """
        # Initialize parent VAE but override decoder output_dim with param_dim
        super().__init__(z_dim=z_dim, hidden_dim=hidden_dim, y_length=y_length)
        # Replace decoder to output parameters instead of reconstructed signal
        self.decoder = Decoder(z_dim=z_dim, hidden_dim=hidden_dim, output_dim=param_dim)

    def forward(self, d):
        """Forward pass through VAE
        Args:
            d (torch.Tensor): Input signal of shape (batch_size, y_length)
        Returns:
            Tuple containing:
                - Reconstructed parameters of shape (batch_size, param_dim) 
                - Mean of latent distribution of shape (batch_size, z_dim)
                - Log variance of latent distribution of shape (batch_size, z_dim)
        """
        # Encode input signal to latent space
        mean, log_var = self.encoder(d)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var