import pytest
import numpy as np
from starccato_flow.data.ccsn_data import CCSNData

def test_ccsn_data_initialization():
    """Test basic initialization and properties of CCSNData"""
    # Initialize dataset
    dataset = CCSNData(noise=False, curriculum=False)
    
    # Test basic properties
    assert dataset is not None
    assert dataset.signals.shape[0] == 256  # Check signal length
    assert dataset.max_strain > 0  # Check max strain is positive
    assert isinstance(dataset.signals, np.ndarray)  # Check signals is numpy array
    
    # Test getting an item
    signal, params = dataset[0]
    assert signal.shape[1] == 256  # Check signal dimension
