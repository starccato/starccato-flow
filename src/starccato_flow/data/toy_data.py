import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.datasets import make_moons

from ..utils.defaults import Y_LENGTH, TEN_KPC
from ..plotting.plotting import plot_signal_distribution
from . import BaseDataset

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

class ToyData(BaseDataset):
    def __init__(self, num_signals=1684, signal_length=Y_LENGTH, noise=True, curriculum=False, noise_level=0.1, 
                 start_snr=200, end_snr=10, rho_target=10,
                 shared_params=None, shared_min=None, shared_max=None, shared_max_strain=None):
        _set_seed(42)
        self.num_signals = num_signals
        self.signal_length = signal_length
        self.num_epochs = 256
        self.noise = noise
        self.start_snr = start_snr
        self.end_snr = end_snr
        self.rho_target = rho_target
        self.curriculum = curriculum
        self.noise_level = noise_level
        self._current_epoch = 0

        # Use shared parameters/min/max if provided, otherwise generate new
        if shared_params is not None:
            self.parameters = shared_params
            self.min_parameter = shared_min
            self.max_parameter = shared_max
        else:
            self.parameters = self.generate_parameters()
            self.min_parameter = self.parameters.min(axis=0).astype(np.float32)
            self.max_parameter = self.parameters.max(axis=0).astype(np.float32)
        
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

    @property
    def parameter_names(self):
        """Return parameter names for the toy dataset."""
        return ['x', 'y']  # Two moons 2D coordinates

    def plot_signal_distribution(self, background=True, font_family="Serif", font_name="Times New Roman", fname=None):
        # Transpose signals to match expected shape (signal_length, num_signals)
        plot_signal_distribution(self.signals.T, generated=False, background=background, font_family=font_family, font_name=font_name, fname=fname)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        normalised_signal = self.normalise_signals(signal)
        
        # Add noise if enabled with SNR control
        if self.noise:
            noise = np.random.normal(0, 1.0, self.signal_length)
            signal_power = np.sqrt(np.mean(signal**2))
            noise_power = np.sqrt(np.mean(noise**2))
            # Scale noise to achieve target SNR
            scaled_noise = noise * (signal_power / (self.rho_target * noise_power))
            noisy_signal = signal + scaled_noise
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
    
    def update_snr(self, snr):
        """Update the target SNR.
        
        Args:
            snr: Target signal-to-noise ratio
        """
        self.rho_target = snr
    
    def set_snr(self, snr):
        """Set the target SNR (alias for update_snr).
        
        Args:
            snr: Target signal-to-noise ratio
        """
        self.rho_target = snr

    @property
    def current_epoch(self) -> int:
        """Get the current epoch number.

        Returns:
            int: Current epoch number
        """
        return self._current_epoch

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch number and adjust SNR for curriculum learning.

        Args:
            epoch (int): New epoch number
        """
        self._current_epoch = epoch
        if self.curriculum:
            self.rho_target = -1 * (epoch / self.num_epochs) * (abs(self.start_snr - self.end_snr)) + self.start_snr

    def get_loader(self, batch_size=32):
        return DataLoader(
            self, batch_size=batch_size, shuffle=True, num_workers=0
        )

    def get_signals_iterator(self, batch_size=32):
        return next(iter(self.get_loader(batch_size=batch_size)))