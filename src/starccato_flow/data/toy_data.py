import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.datasets import make_moons

from ..utils.defaults import Y_LENGTH, TEN_KPC
from ..plotting.plotting import plot_signal_distribution

"""
This class generates synthetic time series data for testing purposes.
signals: y
parameters: x (two moons distribution)
"""

def _set_seed(seed: int):
    """Set the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return seed

class ToyData:
    def __init__(self, num_signals=1684, signal_length=Y_LENGTH, noise=True, curriculum=False, noise_level=0.1, 
                 shared_params=None, shared_min=None, shared_max=None, shared_max_strain=None):
        _set_seed(42)
        self.num_signals = num_signals
        self.signal_length = signal_length
        self.num_epochs = 256
        self.noise = noise
        self.start_snr = 200
        self.end_snr = 10
        self.curriculum = curriculum
        self.noise_level = noise_level

        # Use shared parameters/min/max if provided, otherwise generate new
        if shared_params is not None:
            self.parameters = shared_params
            self.parameter_min = shared_min
            self.parameter_max = shared_max
        else:
            self.parameters = self.generate_parameters()
            self.parameter_min = self.parameters.min(axis=0).astype(np.float32)
            self.parameter_max = self.parameters.max(axis=0).astype(np.float32)
        
        self.signals = self.generate_data()
        
        # Use shared max_strain if provided, otherwise compute from this subset
        if shared_max_strain is not None:
            self.max_strain = shared_max_strain
        else:
            self.max_strain = abs(self.signals).max()

    def generate_parameters(self):
        # Generate parameters from two moons distribution
        data = make_moons(self.num_signals, noise=0.05)
        parameters = data[0]  # Shape: (num_signals, 2)
        return parameters.astype(np.float32)

    def generate_data(self):
        """Generate sine wave signals based on 2D parameters from two moons"""
        t = np.linspace(0, 2*np.pi, self.signal_length)
        signals = []
        for i in range(self.num_signals):
            # Use parameters to modulate frequency and phase of sine wave
            freq = 1 + self.parameters[i, 0] * 0.5  # parameter controls frequency
            phase = self.parameters[i, 1]  # parameter controls phase
            # Generate clean signal
            signal = 1e-20 * np.sin(freq * t + phase)
            signals.append(signal)
        return np.array(signals)  # Shape: (num_signals, signal_length)
    
    def __len__(self):
        return self.num_signals
    
    @property
    def shape(self):
        return self.signals.shape

    def normalise_signals(self, signal):
        normalised_signal = signal / self.max_strain
        return normalised_signal
    
    def normalize_parameters(self, params):
        """Normalize toy data parameters to roughly [-1, 1] range.
        
        Two moons data is roughly in [-2, 2] range, so normalize to [-1, 1].
        
        Args:
            params: numpy array of shape (..., 2)
            
        Returns:
            Normalized parameters
        """
        # Two moons data is roughly [-1.5, 2.5] in x and [-1, 1.5] in y
        # Simple normalization to [-1, 1]
        params_norm = params.copy()

        # Min-max normalization: (x - min) / (max - min) * 2 - 1
        param_range = self.parameter_max - self.parameter_min
        params_norm = 2 * (params - self.parameter_min) / param_range - 1
        
        return params_norm
    
    def denormalize_parameters(self, params_norm):
        """Denormalize toy parameters back to original range.
        
        Args:
            params_norm: numpy array of shape (..., 2) with normalized params
            
        Returns:
            Denormalized parameters
        """
        params = params_norm.copy()

        # Reverse normalization: x = (x_norm + 1) / 2 * (max - min) + min
        param_range = self.parameter_max - self.parameter_min
        params = (params_norm + 1) / 2 * param_range + self.parameter_min

        return params

    def plot_signal_distribution(self, background=True, font_family="Serif", font_name="Times New Roman", fname=None):
        # Transpose signals to match expected shape (signal_length, num_signals)
        plot_signal_distribution(self.signals.T, generated=False, background=background, font_family=font_family, font_name=font_name, fname=fname)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        normalised_signal = self.normalise_signals(signal)
        
        # Add noise if enabled
        if self.noise:
            noisy_signal = signal + np.random.normal(0, self.noise_level, self.signal_length)
            normalised_noisy_signal = self.normalise_signals(noisy_signal)
        else:
            normalised_noisy_signal = normalised_signal
        
        parameters = self.parameters[idx].astype(np.float32)
        
        # Normalize parameters to [-1, 1]
        parameters = self.normalize_parameters(parameters)
        
        # Reshape for compatibility
        normalised_signal = normalised_signal.reshape(1, -1)
        normalised_noisy_signal = normalised_noisy_signal.reshape(1, -1)
        parameters = parameters.reshape(1, -1)
        
        # Return format: (clean_signal, noisy_signal, parameters) to match CCSNData
        return (
            torch.tensor(normalised_signal, dtype=torch.float32),
            torch.tensor(normalised_noisy_signal, dtype=torch.float32),
            torch.tensor(parameters, dtype=torch.float32)
        )
    
    def verify_alignment(self):
        """Verify that signals and parameters are properly aligned."""
        print("\nVerifying data alignment:")
        print(f"Number of signals: {self.signals.shape[0]}")
        print(f"Number of parameter sets: {len(self.parameters)}")
        # print(f"Parameter columns: {self.parameters.columns.tolist()}")
        # print(f"First few parameter values:\n{self.parameters.head()}")
        return True
    
    @property
    def current_epoch(self) -> int:
        """Get the current epoch number.

        Returns:
            int: Current epoch number
        """
        return self._current_epoch

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch number.

        Args:
            epoch (int): New epoch number
        """
        self._current_epoch = epoch
        self.rho_target = -1 * (epoch / self.num_epochs) * (abs(self.start_snr - self.end_snr)) + self.start_snr

    def get_loader(self, batch_size=32):
        return DataLoader(
            self, batch_size=batch_size, shuffle=True, num_workers=0
        )

    def get_signals_iterator(self, batch_size=32):
        return next(iter(self.get_loader(batch_size=batch_size)))