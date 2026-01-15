"""Base classes and utilities for dataset handling."""

import numpy as np


class BaseDataset:
    """Base class with shared functionality for all datasets."""
    
    def normalise_signals(self, signal):
        """Normalize signals by dividing by max strain.
        
        Args:
            signal: Signal array to normalize
            
        Returns:
            Normalized signal
        """
        return signal / self.max_strain
    
    def denormalise_signals(self, signal):
        """Denormalize signals by multiplying by max strain.
        
        Args:
            signal: Normalized signal array
            
        Returns:
            Denormalized signal in original units
        """
        return signal * self.max_strain
    
    def normalize_parameters(self, params):
        """Normalize parameters to [-1, 1] range using min-max normalization.
        
        Args:
            params: numpy array of shape (..., n_params) with parameter values
            
        Returns:
            Normalized parameters in [-1, 1] range
        """
        params_norm = params.copy()
        
        # Min-max normalization: (x - min) / (max - min) * 2 - 1
        param_range = self.max_parameter - self.min_parameter
        params_norm = 2 * (params - self.min_parameter) / param_range - 1
        
        return params_norm
    
    def denormalize_parameters(self, params_norm):
        """Denormalize parameters from [-1, 1] back to original ranges.
        
        Args:
            params_norm: numpy array of shape (..., n_params) with normalized params
            
        Returns:
            Denormalized parameters in original physical units
        """
        params = params_norm.copy()
        
        # Reverse normalization: x = (x_norm + 1) / 2 * (max - min) + min
        param_range = self.max_parameter - self.min_parameter
        params = (params_norm + 1) / 2 * param_range + self.min_parameter
        
        return params
