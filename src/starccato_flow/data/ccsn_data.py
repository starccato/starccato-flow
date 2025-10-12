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
            noise = noise * (1 + self._current_epoch / self.num_epochs) * 500
        else: 
            noise = noise * 500
            
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

    def generate_aligo_noise(self, length: int = 256, log: bool = False) -> np.ndarray:
        """Generate Advanced LIGO noise using proper PSD.
        
        Args:
            length (int): Length of noise array
            log (bool): Whether to return the logarithm of the noise

        Returns:
            np.ndarray: Colored noise array
        """
        delta_t = 1.0 / 4096.0  # Time step (sampling rate of 4096 Hz)
        noise = self.rnoise(length, delta_t)

        if log:
            noise = np.log1p(noise)

        return noise

    def spec_adv(self, frequencies: np.ndarray, log: bool = False) -> np.ndarray:
        """Compute the Advanced LIGO power spectral density (PSD).
        
        Exact Python implementation of the R LALAdvLIGOPsd() function.
        The minimum is at 228.3 Hz and is bounded above at 2e10 times minimum value.

        Args:
            frequencies (np.ndarray): Array of frequencies
            log (bool): Whether to return the logarithm of the PSD

        Returns:
            np.ndarray: The PSD values for the given frequencies
        """
        # Upper bound cutoff calculation (same as R implementation)
        cutoff = -109.35 + np.log(2e10)
        
        # Normalize frequencies by 215 Hz
        x = frequencies / 215
        x2 = x * x  # More efficient than x**2
        
        # Handle zero frequency case
        mask_zero = (x == 0)
        x = np.where(mask_zero, 1e-10, x)  # Replace zeros with small value
        x2 = np.where(mask_zero, 1e-20, x2)
        
        # Calculate components safely
        seismic = x**(-4.14)
        thermal = 5/x2
        quantum = 111*(1-x2+0.5*x2*x2)/(1+0.5*x2)
        
        # Calculate log PSD with better numerical stability
        with np.errstate(divide='ignore', invalid='ignore'):
            log_psd = np.log(1e-49) + np.log(seismic - thermal + quantum)
        
        # Apply cutoff and handle invalid values
        log_psd = np.where(mask_zero, cutoff, log_psd)  # Set zero freq to cutoff
        log_psd = np.where((log_psd > cutoff) | (~np.isfinite(log_psd)), cutoff, log_psd)
        
        # Return either log or exponential form (matching R behavior)
        return log_psd if log else np.exp(log_psd)

    def rnoise(self, N: int, delta_t: float, one_sided: bool = True, pad: int = 1) -> np.ndarray:
        """Generate random noise in the Fourier domain based on aLIGO PSD.
        
        Exact Python implementation of the R rnoise() function.

        Args:
            N (int): Number of samples
            delta_t (float): Time step
            one_sided (bool): Whether to use one-sided spectrum (like R implementation)
            pad (int): Padding factor for better frequency resolution

        Returns:
            np.ndarray: Time-domain colored noise with proper scaling
        """
        # Input validation (matching R implementation)
        if pad < 1 or int(pad) != pad:
            raise ValueError("pad must be an integer >= 1")
            
        orig_N = N
        N *= pad
        
        # Setup frequency domain (matching R implementation)
        is_even = (N % 2 == 0)
        half_N = N//2 if is_even else (N-1)//2
        delta_f = 1 / (N * delta_t)
        fourier_freq = np.arange(half_N + 1) * delta_f
        
        # Setup kappa and lambda factors (exactly as in R)
        kappa = np.zeros(half_N + 1)
        kappa[1:] = 1
        if is_even:
            kappa[-1] = 0
            
        lambda_factors = np.ones(half_N + 1)
        if one_sided:
            lambda_factors[1:-1] = 2
            if not is_even:
                lambda_factors[-1] = 2
        
        # Generate random noise in Fourier domain with proper scaling
        psd = self.spec_adv(fourier_freq, log=False)
        
        # Handle improper PSD values (matching R behavior)
        psd[~np.isfinite(psd)] = 0
        psd[psd < 0] = 0
        
        # Calculate standard deviations with proper scaling
        if one_sided:
            sd_vec = np.sqrt(psd / ((1 + kappa) * lambda_factors))
        else:
            sd_vec = np.sqrt(psd / (1 + kappa))
            
        # Generate Fourier coefficients with correct scaling
        scale = np.sqrt(N / delta_t)  # Matches R implementation scaling
        a = np.random.normal(0, sd_vec)
        b = np.random.normal(0, sd_vec) * kappa
        
        # Build complex Fourier transform (matching R implementation)
        real = scale * a
        imag = -scale * b  # Note the negative sign as in R implementation
        
        # Create full Fourier transform
        real_full = np.concatenate([real, real[1:-1][::-1] if is_even else real[1:][::-1]])
        imag_full = np.concatenate([imag, -imag[1:-1][::-1] if is_even else -imag[1:][::-1]])
        noise_ft = real_full + 1j * imag_full
        
        # Transform to time domain with proper normalization
        noise = np.real(ifft(noise_ft)) / N  # Division by N matches R normalization
        
        # Handle padding exactly as in R implementation
        if pad > 1:
            start = np.random.randint(0, orig_N * (pad-1))
            noise = noise[start:start + orig_N]
            
        return noise

        return noise