import pytest
import torch
from starccato_flow.nn.vae import VAE, Encoder, Decoder
from starccato_flow.nn.vae_parameter import VAE_PARAMETER

@pytest.fixture
def model_params():
    """Fixture for common model parameters"""
    return {
        'z_dim': 8,
        'hidden_dim': 128,
        'y_length': 256,
        'param_dim': 1,
        'batch_size': 32
    }

def test_vae_initialization(model_params):
    """Test VAE model initialization and basic properties"""
    vae = VAE(
        z_dim=model_params['z_dim'],
        hidden_dim=model_params['hidden_dim'],
        y_length=model_params['y_length']
    )
    
    assert isinstance(vae.encoder, Encoder)
    assert isinstance(vae.decoder, Decoder)
    
def test_vae_forward_pass(model_params):
    """Test VAE forward pass with dummy data"""
    vae = VAE(
        z_dim=model_params['z_dim'],
        hidden_dim=model_params['hidden_dim'],
        y_length=model_params['y_length']
    )
    
    # Create dummy input
    y = torch.randn(model_params['batch_size'], 1, model_params['y_length'])
    
    # Forward pass
    y_hat, mean, log_var = vae(y)
    
    # Check output shapes
    assert y_hat.shape == y.shape
    assert mean.shape == (model_params['batch_size'], 1, model_params['z_dim'])
    assert log_var.shape == (model_params['batch_size'], 1, model_params['z_dim'])

def test_vae_parameter_initialization(model_params):
    """Test VAE_PARAMETER model initialization and basic properties"""
    vae_param = VAE_PARAMETER(
        z_dim=model_params['z_dim'],
        hidden_dim=model_params['hidden_dim'],
        y_length=model_params['y_length'],
        param_dim=model_params['param_dim']
    )
    
    assert isinstance(vae_param.encoder, Encoder)
    assert isinstance(vae_param.decoder, Decoder)

def test_vae_parameter_forward_pass(model_params):
    """Test VAE_PARAMETER forward pass with dummy data"""
    vae_param = VAE_PARAMETER(
        z_dim=model_params['z_dim'],
        hidden_dim=model_params['hidden_dim'],
        y_length=model_params['y_length'],
        param_dim=model_params['param_dim']
    )
    
    # Create dummy input
    x = torch.randn(model_params['batch_size'], 1, model_params['y_length'])
    
    # Forward pass
    params, mean, log_var = vae_param(x)
    
    # Check output shapes
    assert params.shape == (model_params['batch_size'], 1, model_params['param_dim'])
    assert mean.shape == (model_params['batch_size'], 1, model_params['z_dim'])
    assert log_var.shape == (model_params['batch_size'], 1, model_params['z_dim'])

def test_reparameterization(model_params):
    """Test reparameterization trick"""
    vae = VAE(
        z_dim=model_params['z_dim'],
        hidden_dim=model_params['hidden_dim'],
        y_length=model_params['y_length']
    )
    
    mean = torch.zeros(model_params['batch_size'], model_params['z_dim'])
    log_var = torch.zeros(model_params['batch_size'], model_params['z_dim'])
    
    # Sample multiple times
    samples = [vae.reparameterization(mean, torch.exp(0.5 * log_var)) 
              for _ in range(100)]
    samples = torch.stack(samples)
    
    # Check that samples are different (stochastic)
    assert not torch.allclose(samples[0], samples[1])
    
    # Check sample statistics (should be close to N(0,1))
    sample_mean = samples.mean(dim=0)
    sample_std = samples.std(dim=0)
    
    assert torch.allclose(sample_mean, torch.zeros_like(sample_mean), atol=0.1)
    assert torch.allclose(sample_std, torch.ones_like(sample_std), atol=0.1)