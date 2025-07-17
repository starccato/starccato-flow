import numpy as np
import torch
from torch.utils.data import DataLoader

'''This class generates synthetic time series data for testing purposes.'''

class ToyData:
    def __init__(self, indices, num_signals, signal_length, noise_level=0.1):
        self.indices = indices
        self.num_signals = num_signals
        self.signal_length = signal_length
        self.noise_level = noise_level

        self.parameters = self.generate_parameters()
        self.signals = self.generate_data()

        self.parameters = self.parameters[self.indices]
        self.signals = self.signals[self.indices]

        self.max_strain = abs(self.signals).max()

    def generate_parameters(self):
        # Generate synthetic parameters for each signal
        frequencies = np.random.uniform(0.5, 2.0, size=self.num_signals)
        phases = np.random.uniform(0, 2 * np.pi, size=self.num_signals)
        # Shape: (num_signals, 2)
        parameters = np.stack([frequencies, phases], axis=1)
        return parameters

    def generate_data(self):
        t = np.linspace(0, 1, self.signal_length)
        signals = []
        for i in range(self.num_signals):
            frequency = self.parameters[i, 0]
            phase = self.parameters[i, 1]
            signal = np.sin(2 * np.pi * frequency * t + phase) + np.random.normal(0, self.noise_level, self.signal_length)
            signals.append(signal)
        return np.array(signals)  # Shape: (num_signals, signal_length)

    ### overloads ###
    def __len__(self):
        return self.num_signals

    @property
    def shape(self):
        return self.signals.shape

    def get_indices(self):
        return self.indices

    def normalise_signals(self, signal):
        normalised_signal = signal / self.max_strain
        return normalised_signal

    def __getitem__(self, idx):
        signal = self.signals[idx]
        normalised_signal = self.normalise_signals(signal)
        parameters = self.parameters[idx].astype(np.float32)
        # Reshape for compatibility if needed
        normalised_signal = normalised_signal.reshape(1, -1)
        parameters = parameters.reshape(1, -1)
        return torch.tensor(normalised_signal, dtype=torch.float32), torch.tensor(parameters, dtype=torch.float32)

    def get_loader(self, batch_size=32):
        return DataLoader(
            self, batch_size=batch_size, shuffle=True, num_workers=0
        )

    def get_signals_iterator(self, batch_size=32):
        return next(iter(self.get_loader(batch_size=batch_size)))