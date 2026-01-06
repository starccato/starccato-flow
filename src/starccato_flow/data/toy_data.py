import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils.defaults import Y_LENGTH, TEN_KPC
from ..plotting.plotting import plot_signal_distribution

"""
This class generates synthetic time series data for testing purposes.
signals: y
parameters: x (frequencies and phases)
"""

def _set_seed(seed: int):
    """Set the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return seed

class ToyData:
    def __init__(self, num_signals=1684, signal_length=Y_LENGTH, noise = True, noise_level=0.1):
        _set_seed(42)
        self.num_signals = num_signals
        self.signal_length = signal_length
        self.noise = noise
        self.curriculum = False
        self.noise_level = noise_level

        self.parameters = self.generate_parameters()
        self.signals = self.generate_data()

        self.max_strain = abs(self.signals).max()

    def generate_parameters(self):
        # Generate synthetic parameters for each signal
        # frequencies = np.random.uni(loc=1.0, scale=0.2, size=self.num_signals)  # mean=1.0, std=0.2
        # generate uniform distribution for frequencies
        frequencies = np.random.uniform(low=0.0, high=1.0, size=self.num_signals)  # mean=1.0, std=0.2
        # phases = np.random.normal(loc=np.pi, scale=1.0, size=self.num_signals)      # mean=pi, std=1.0
        phases = np.random.uniform(low=-np.pi, high=np.pi, size=self.num_signals)  # mean=0, std=pi
        # Shape: (num_signals, 2)
        parameters = np.stack([frequencies, phases], axis=1)
        return parameters

    def generate_data(self):
        t = np.linspace(0, 1, self.signal_length)
        signals = []
        for i in range(self.num_signals):
            frequency = self.parameters[i, 0]
            phase = self.parameters[i, 1]
            # Generate clean signal without noise
            signal = 400 * np.sin(2 * np.pi * frequency * t + phase)
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

    def plot_signal_distribution(self, background=True, font_family="Serif", font_name="Times New Roman", fname=None):
        # Transpose signals to match expected shape (signal_length, num_signals)
        plot_signal_distribution(self.signals.T / TEN_KPC, generated=False, background=background, font_family=font_family, font_name=font_name, fname=fname)

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
        
        # Reshape for compatibility
        normalised_signal = normalised_signal.reshape(1, -1)
        normalised_noisy_signal = normalised_noisy_signal.reshape(1, -1)
        parameters = parameters.reshape(1, -1)
        
        # Return format: (clean_signal, noisy_signal, parameters) to match CCSNSNRData
        return (
            torch.tensor(normalised_signal, dtype=torch.float32),
            torch.tensor(normalised_noisy_signal, dtype=torch.float32),
            torch.tensor(parameters, dtype=torch.float32)
        )

    def get_loader(self, batch_size=32):
        return DataLoader(
            self, batch_size=batch_size, shuffle=True, num_workers=0
        )

    def get_signals_iterator(self, batch_size=32):
        return next(iter(self.get_loader(batch_size=batch_size)))