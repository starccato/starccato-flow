import unittest
import numpy as np
import torch

from starccato_flow.nn.flow import Flow

def test_flow_step():
    """Test the step function of the Flow model."""
    # Initialize model
    model = Flow(dim=2, signal_dim=256, h=128)
    
    # Create dummy inputs
    batch_size = 4
    x_t = torch.randn(batch_size, 2)  # Current state
    t_start = torch.tensor([0.0])  # Start time
    t_end = torch.tensor([0.1])  # End time
    h = torch.randn(batch_size, 256)  # Conditioning signal

    # Perform a step
    x_t_next = model.step(x_t, t_start, t_end, h)
    
    # Check output shape
    assert x_t_next.shape == (batch_size, 2), "Output shape mismatch"
    
    print("Flow step function test passed.")

def test_flow_forward():
    """Test the forward function of the Flow model."""
    # Initialize model
    model = Flow(dim=2, signal_dim=256)
    
    # Create dummy inputs
    batch_size = 4
    x_t = torch.randn(batch_size, 2)  # Current state
    t = torch.randn(batch_size, 1)  # Time
    h = torch.randn(batch_size, 256)  # Conditioning signal

    # Perform a forward pass
    output = model(x_t, t, h)
    
    # Check output shape
    assert output.shape == (batch_size, 2), "Output shape mismatch"

