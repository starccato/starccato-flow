import signal
import pytest
import numpy as np
from starccato_flow.data.ccsn_data import CCSNData

def test_ccsn_data_initialization():
    """Test basic initialization and properties of CCSNData"""
    # Initialize dataset
    dataset = CCSNData(noise=False, curriculum=False)
    
    # Test basic properties
    assert dataset is not None
    assert dataset.signals.shape[1] == 256
    assert dataset.max_strain > 0
    
    # Test getting an item
    d, s, theta = dataset[0]

    assert d.shape[1] == 256  # Check signal dimension
    assert s.shape[1] == 256  # Check noisy signal dimension

def test_ccsn_signal_normalisation():
    """Test signal normalisation in CCSNData"""
    dataset = CCSNData(noise=False, curriculum=False)
    
    # Get a signal
    d, _, _ = dataset[0]
    
    # run normalisation function
    dataset.normalise_signals(d.numpy())

    assert np.max(np.abs(d.numpy())) <= 1.0, "Signal not properly normalised"
    
def test_ccsn_normalise_parameters():
    """Test parameter normalization in CCSNData"""
    dataset = CCSNData(noise=False, curriculum=False)
    
    # Get parameters
    _, _, theta = dataset[0]
    
    # run normalization function
    normalized_theta = dataset.normalize_parameters(theta.numpy())
    
    assert np.all(normalized_theta.numpy() >= -1.0) and np.all(normalized_theta.numpy() <= 1.0)