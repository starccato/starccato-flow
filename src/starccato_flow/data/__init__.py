"""Base classes and utilities for dataset handling."""


class BaseDataset:
    """Base class with shared functionality for all datasets."""

    def _get_parameter_bounds(self):
        """Return parameter normalization bounds across legacy/new attribute names."""
        if hasattr(self, "min_theta") and hasattr(self, "max_theta"):
            return self.min_theta, self.max_theta
        if hasattr(self, "min_parameter") and hasattr(self, "max_parameter"):
            return self.min_parameter, self.max_parameter
        raise AttributeError(
            "Dataset is missing parameter bounds. Expected either "
            "(min_theta, max_theta) or (min_parameter, max_parameter)."
        )
    
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
        
        min_params, max_params = self._get_parameter_bounds()

        # Min-max normalization: (x - min) / (max - min) * 2 - 1
        param_range = max_params - min_params
        params_norm = 2 * (params - min_params) / param_range - 1
        
        return params_norm
    
    def denormalize_parameters(self, params_norm):
        """Denormalize parameters from [-1, 1] back to original ranges.
        
        Args:
            params_norm: numpy array of shape (..., n_params) with normalized params
            
        Returns:
            Denormalized parameters in original physical units
        """
        params = params_norm.copy()
        
        min_params, max_params = self._get_parameter_bounds()

        # Reverse normalization: x = (x_norm + 1) / 2 * (max - min) + min
        param_range = max_params - min_params
        params = (params_norm + 1) / 2 * param_range + min_params
        
        return params


# Import main dataset classes for easier access
from .h_theta_multi import hThetaMulti
from .s_theta import sTheta

__all__ = ['BaseDataset', 'sTheta']
