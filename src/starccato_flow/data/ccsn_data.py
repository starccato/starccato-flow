import math
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from scipy.fft import fft, ifft

from ..plotting.plotting import plot_signal_distribution, plot_signal_grid
from ..utils.defaults import BATCH_SIZE, DEVICE
from ..utils.defaults import PARAMETERS_CSV, SIGNALS_CSV, TIME_CSV

"""This loads the signal data from the raw simulation outputs from Richers et al (20XX) ."""

class CCSNData(Dataset):
    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        num_epochs: int = 256,
        frac: float = 1.0,
        train: bool = True,
        noise: bool = True,
        curriculum: bool = True,
        indices: Optional[np.ndarray] = None,
        multi_param: bool = False
    ):
        """Initialize the CCSN dataset.
        
        Args:
            batch_size (int): Batch size for data loading
            frac (float): Fraction of data to use
            train (bool): Whether this is training data
            noise (bool): Whether to add noise
            curriculum (bool): Whether to use curriculum learning
            indices (Optional[np.ndarray]): Specific indices to use
            multi_param (bool): Whether to use multiple parameters
        """
        self._current_epoch = 0
        self.num_epochs = num_epochs
        self.parameters = pd.read_csv(PARAMETERS_CSV)
        self.signals = pd.read_csv(SIGNALS_CSV).astype("float32").T
        self.signals.index = [i for i in range(len(self.signals.index))]
        self.noise = noise
        self.curriculum = curriculum

        assert (
            self.signals.shape[0] == self.parameters.shape[0],
            "Signals and parameters must have the same number of rows (the number of signals)",
        )

        if frac < 1:
            init_shape = self.signals.shape
            n_signals = int(frac * self.signals.shape[0])
            # keep n_signals random signals columns
            self.signals = self.signals.sample(n=n_signals, axis=0)
            self.parameters = self.parameters.iloc[self.signals.index, :]
        
        # remove unusual parameters and corresponding signals
        keep_idx = self.parameters["beta1_IC_b"] > 0
        self.parameters = self.parameters[keep_idx]

        # parameter_set = ["beta1_IC_b", "A(km)", "EOS"]
        # parameter_set = ["beta1_IC_b"]

        if multi_param:
            parameter_set = ["beta1_IC_b", "A(km)", "EOS"]
        else: 
            parameter_set = ["beta1_IC_b"]

        # keep only the parameters we want
        self.parameters = self.parameters[parameter_set]

        # akm = pd.get_dummies(self.parameters["A(km)"], prefix="A")
        # self.parameters = pd.concat([self.parameters.drop(columns=["A(km)"]), akm], axis=1)

        # Equal frequency binning for beta1_IC_b
        if "beta1_IC_b" in parameter_set:
            self.parameters['beta1_IC_b'] = pd.qcut(
                self.parameters['beta1_IC_b'], q=3, labels=False
            )
            beta_bins = pd.get_dummies(self.parameters['beta1_IC_b'], prefix="beta_bin")
            self.parameters = pd.concat([self.parameters.drop(columns=["beta1_IC_b"]), beta_bins], axis=1)

        if multi_param:
            # one hot encode A(km)
            akm = pd.get_dummies(self.parameters["A(km)"], prefix="A")
            self.parameters = pd.concat([self.parameters.drop(columns=["A(km)"]), akm], axis=1)

            # one hot encode EOS
            eos = pd.get_dummies(self.parameters["EOS"], prefix="EOS")
            self.parameters = pd.concat([self.parameters.drop(columns=["EOS"]), eos], axis=1)

        self.signals = self.signals[keep_idx]
        self.signals = self.signals.values.T

        ### flatten signals and take last 256 timestamps
        temp_data = np.empty(shape=(256, 0)).astype("float32")

        for i in range(0, self.signals.shape[1]):
            signal = self.signals[:, i]
            signal = signal.reshape(1, -1)

            cut_signal = signal[:, int(len(signal[0]) - 256) : len(signal[0])]
            temp_data = np.insert(
                temp_data, temp_data.shape[1], cut_signal, axis=1
            )

        self.signals = temp_data

        if indices is not None:
            if train:
                self.signals = self.signals[:, indices]
                self.parameters = self.parameters.iloc[indices]
                self.indices = indices
            else:
                self.signals = self.signals[:, indices]
                self.parameters = self.parameters.iloc[indices]
                self.indices = indices

        self.batch_size = batch_size
        self.mean = self.signals.mean()
        self.std = np.std(self.signals, axis=None)
        self.scaling_factor = 5
        self.max_strain = abs(self.signals).max()
        self.ylim_signal = (self.signals[:, :].min(), self.signals[:, :].max())

    def plot_signal_distribution(self, background=True, font_family="Serif", font_name="Times New Roman", fname=None):
        plot_signal_distribution(self.signals, generated=False, background=background, font_family=font_family, font_name=font_name, fname=fname)

    def plot_signal_grid(self, n_signals=3, background=True, font_family="Serif", font_name="Times New Roman", fname=None):
        # Collect indices of the signals to plot
        selected_signals = []
        for i in range(n_signals):
            signal = self.__getitem__(i+100)[0].cpu().numpy().flatten()  # Flatten the signal
            selected_signals.append(signal)

        # Convert selected signals to a NumPy array for plotting
        selected_signals = np.array(selected_signals)

        plot_signal_grid(
            signals=selected_signals,
            max_value=self.max_strain,
            num_cols=3,
            num_rows=1,
            fname=fname,
            background=background,
            generated=False
        )
 
    def __str__(self):
        return f"TrainingData: {self.signals.shape}"

    def __repr__(self):
        return self.__str__()

    @property
    def raw_signals(self):
        return pd.read_csv(SIGNALS_CSV).astype("float32").T.values

    def summary(self):
        """Display summary stats about the data"""
        str = f"Signal Dataset mean: {self.mean:.3f} +/- {self.std:.3f}\n"
        str += f"Signal Dataset scaling factor (to match noise in generator): {self.scaling_factor}\n"
        str += f"Signal Dataset max value: {self.max_strain}\n"
        # str += f"Signal Dataset max parameter value: {self.max_parameter_value}\n"
        str += f"Signal Dataset shape: {self.signals.shape}\n"
        str += f"Parameter Dataset shape: {self.parameters.shape}\n"

    def add_aLIGO_noise(self, signal):
        """Add Advanced LIGO noise to the signal.
        
        Args:
            signal (np.ndarray): The gravitational wave signal
            
        Returns:
            np.ndarray: Signal with properly scaled aLIGO noise added
        """
        dataDeltaT = 1 / 4096  # Sampling rate: 4096 Hz
        dataSec = 256 / 4096   # Duration: 256 samples at 4096 Hz
        dataN = int(dataSec / dataDeltaT)  # Number of samples
        
        # Generate noise with proper PSD scaling
        noise = self.rnoise(
            N=dataN,
            delta_t=dataDeltaT,
            one_sided=True,  # Use one-sided spectrum as in R
            pad=1
        ).reshape(1, -1)  # shape (1, 256)

        # The noise is now properly scaled due to correct PSD implementation
        # No need for the 1000x scaling factor anymore
        noise = noise - noise.mean()  # Mean center as in R implementation

        # Convert strain to standard units (matches R implementation)
        signal = signal / 3.086e+22

        # Add noise with proper SNR if curriculum learning is enabled
        # if self.curriculum_learning:
        #     current_snr = self._get_current_snr()
        #     signal_power = np.mean(signal ** 2)
        #     noise_power = np.mean(noise ** 2)
        #     scaling = np.sqrt(signal_power / (noise_power * current_snr))
        #     noise = noise * scaling

        if self.curriculum:
            noise = noise * (1 + self._current_epoch / self.num_epochs) * 300
        else: 
            noise = noise * 300
            
        # Add scaled noise to signal
        aLIGO_signal = signal + noise 

        aLIGO_signal = aLIGO_signal * 3.086e+22

        return aLIGO_signal

    def normalise_signals(self, signal):
        normalised_signal = signal / self.max_strain
        return normalised_signal
    
    def normalise_parameters(self, parameters):
        normalised_parameters = parameters / self.max_parameter_value
        return normalised_parameters

    ### overloads ###
    def __len__(self):
        return self.signals.shape[1]

    @property
    def shape(self):
        return self.signals.shape
    
    def get_indices(self):
        return self.indices

    def __getitem__(self, idx):
        signal = self.signals[:, idx]
        signal = signal.reshape(1, -1)

        parameters = self.parameters.iloc[idx].values  # Extract parameter values as a NumPy array
        parameters = parameters.astype(np.float32)  # Ensure parameters are float32
        parameters = parameters.reshape(1, -1)

        if self.noise:
            signal = self.add_aLIGO_noise(signal)

        normalised_signal = self.normalise_signals(signal)

        return (
            torch.tensor(normalised_signal, dtype=torch.float32, device=DEVICE),
            torch.tensor(parameters, dtype=torch.float32, device=DEVICE)
        )

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
    
    def get_loader(
        self,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        snr: Optional[float] = None
    ) -> DataLoader:
        """Get data loader with optional noise at specified SNR.
        
        Args:
            batch_size (Optional[int]): Batch size
            shuffle (bool): Whether to shuffle data
            snr (Optional[float]): Signal-to-noise ratio
            
        Returns:
            DataLoader: PyTorch data loader
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Update SNR if provided
        if snr is not None:
            self.snr = snr
            # Reset noise cache for new SNR
            self.current_noise_profile = None
            
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )

    def AdvLIGOPsd(self, f):
        x = f / 215
        x2 = x * x
        psd = 1e-49 * (pow(x, - 4.14) - 5 / x2 + 111 * (1 - x2 + 0.5 * x2 * x2) / (1 + 0.5 * x2))
        # The upper bound is 2e10 times the minimum value
        cutoff = np.nanmin(psd) * 2e10
        psd[(psd > cutoff) | np.isnan(psd)] = cutoff
        return psd
        

    def rnoise(self, N: int, delta_t: float, one_sided: bool = True, pad: int = 1) -> np.ndarray:
        if pad < 1 or int(pad) != pad:
            raise ValueError("pad must be an integer >= 1")

        orig_N = N
        N *= pad
        is_even = (N % 2 == 0)
        half_N = N // 2 if is_even else (N - 1) // 2
        delta_f = 1 / (N * delta_t)
        fourier_freq = np.arange(half_N + 1) * delta_f

        # kappa & lambda as in R
        kappa = np.concatenate(([0], np.ones(half_N)))
        if is_even:
            kappa[-1] = 0
        lambda_factors = np.concatenate(([1], np.full(half_N - 1, 2), [1]))

        psd = self.AdvLIGOPsd(fourier_freq)
        psd[~np.isfinite(psd)] = 0
        psd[psd < 0] = 0

        if one_sided:
            sd_vec = np.sqrt(psd / ((1 + kappa) * lambda_factors))
        else:
            sd_vec = np.sqrt(psd / (1 + kappa))

        a = np.random.normal(0, sd_vec)
        b = np.random.normal(0, sd_vec) * kappa
        scale = np.sqrt(N / delta_t)
        real = scale * a
        imag = -scale * b

        # Mirror properly
        mirror_idx = np.where(kappa > 0)[0]
        real_full = np.concatenate([real, real[mirror_idx[::-1]]])
        imag_full = np.concatenate([imag, -imag[mirror_idx[::-1]]])

        noise_ft = real_full + 1j * imag_full
        noise = np.real(ifft(noise_ft)) / N

        if pad > 1:
            start = np.random.randint(0, orig_N * (pad - 1))
            noise = noise[start:start + orig_N]

        return noise
