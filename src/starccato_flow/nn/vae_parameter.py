from .vae import VAE, Encoder, Decoder

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