import torch
from torch import nn

''' Fully connected (Vanilla) VAE (Variational Autoencoder) implementation in PyTorch.'''

class VAE(nn.Module):
    def __init__(self, z_dim, hidden_dim, y_length):
        super(VAE, self).__init__()
        self.encoder = Encoder(y_length=y_length, hidden_dim=hidden_dim, z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim, hidden_dim=hidden_dim, output_dim=y_length)
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)  # sampling epsilon        
        z = mean + var * epsilon  # reparameterization trick
        return z
    
    def forward(self, d):
        mean, log_var = self.encoder(d)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        d_hat = self.decoder(z)
        return d_hat, mean, log_var

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(z_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, z):
        h = self.LeakyReLU(self.FC_hidden(z))
        h = self.LeakyReLU(self.FC_hidden2(h))
        d_hat = self.FC_output(h)
        return d_hat

class Encoder(nn.Module):
    def __init__(self, y_length, hidden_dim, z_dim):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(y_length, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, z_dim)
        self.FC_var = nn.Linear(hidden_dim, z_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, d):
        h_ = self.LeakyReLU(self.FC_input(d))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance 
        return mean, log_var