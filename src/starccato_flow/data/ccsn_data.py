import math
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from scipy.fft import fft, ifft

from ..plotting.plotting import plot_candidate_signal, plot_signal_distribution, plot_signal_grid, plot_parameter_distribution, plot_parameter_distribution_grid
from ..utils.defaults import BATCH_SIZE, DEVICE, TEN_KPC
from ..utils.defaults import SAMPLING_RATE, Y_LENGTH
from ..utils.defaults import PARAMETERS_CSV, SIGNALS_CSV, TIME_CSV
from . import BaseDataset

"""This loads the signal data from the raw simulation outputs from Richers et al (20XX) ."""


is_even = (Y_LENGTH % 2 == 0)
half_N = Y_LENGTH // 2 if is_even else (Y_LENGTH - 1) // 2
delta_f = 1 / (Y_LENGTH * SAMPLING_RATE)
fourier_freq = np.arange(half_N + 1) * delta_f

class CCSNData(BaseDataset, Dataset):
    # Default LaTeX labels for parameters
    PARAMETER_LABELS = {
        'beta1_IC_b': r'$\beta_{IC,b}$',
        'omega_0(rad|s)': r'$\omega_0$',
        'A(km)': r'$\A$',
        'Ye_c_b': r'$Y_{e,c,b}$'
    }
    
    # Default parameter ranges for plotting (in physical units after denormalization)
    PARAMETER_RANGES = {
        'beta1_IC_b': (0, 0.25),
        'omega_0(rad|s)': (0, 16),
        'A(km)': (0, 10000),
        'Ye_c_b': (0, 0.3)
    }
    
    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        num_epochs: int = Y_LENGTH,
        noise: bool = True,
        curriculum: bool = True,
        snr: bool = True,
        start_snr: int = 100,
        end_snr: int = 10,
        rho_target: int = 10,
        indices: Optional[np.ndarray] = None,
        multi_param: bool = True,
        noise_realizations: int = 1,
        shared_min: Optional[np.ndarray] = None,
        shared_max: Optional[np.ndarray] = None,
        shared_max_strain: Optional[float] = None
    ):
        """Initialize the CCSN dataset.
        
        Args:
            batch_size (int): Batch size for data loading
            noise (bool): Whether to add noise
            curriculum (bool): Whether to use curriculum learning
            indices (Optional[np.ndarray]): Specific indices to use
            multi_param (bool): Whether to use multiple parameters
            noise_realizations (int): Number of different noise realizations per signal (multiplies dataset size)
        """
        self.batch_size = batch_size
        self._current_epoch = 0
        self.num_epochs = num_epochs
        
        # Load data from CSV files
        params_df = pd.read_csv(PARAMETERS_CSV)
        signals_df = pd.read_csv(SIGNALS_CSV).astype("float32").T
        
        self.noise = noise
        self.curriculum = curriculum
        self.snr = snr
        self.start_snr = start_snr
        self.end_snr = end_snr
        self.rho_target = rho_target
        self.noise_realizations = noise_realizations

        # Remove unusual parameters and corresponding signals
        keep_idx = params_df["beta1_IC_b"].values > 0
        # print(f"Removing {(~keep_idx).sum()} signals with beta1_IC_b <= 0")
        params_df = params_df[keep_idx]
        
        if multi_param:
            # parameter_set = ["beta1_IC_b", "omega_0(rad|s)", "A(km)", "Ye_c_b"]
            parameter_set = ["beta1_IC_b"]
        else: 
            parameter_set = ["beta1_IC_b"]

        # keep only the parameters we want
        params_df = params_df[parameter_set]
        
        # Store parameter names for reference
        self.parameter_names = parameter_set
        
        # Convert to numpy array and apply log transformations BEFORE computing min/max
        # This brings all parameters to similar scales for better training
        self.parameters = params_df.values.astype(np.float32)
        
        # Apply log transformations A (column 2)
        # if multi_param:
            # self.parameters[:, 2] = np.log(self.parameters[:, 2] + 1e-8)  # A(km)

        # Keep track of original indices
        signal_indices = np.where(keep_idx)[0]
        signals_df = signals_df[keep_idx]
        self.signals = signals_df.values.T

        # print(f"Processing {self.signals.shape[1]} signals")
        # print(f"Parameters shape: {self.parameters.shape}")
        assert self.signals.shape[1] == self.parameters.shape[0], "Signal and parameter counts don't match!"

        ### Take last Y_LENGTH timestamps from all signals (vectorized operation)
        self.signals = self.signals[-Y_LENGTH:, :]
        
        # Store original signal indices for verification
        self.original_indices = signal_indices.tolist()
        
        # Verify alignment
        assert self.signals.shape[1] == self.parameters.shape[0], "Signal and parameter counts don't match after processing!"

        if indices is not None:
            self.signals = self.signals[:, indices]
            self.parameters = self.parameters[indices]
            self.indices = indices

        # Use shared max_strain if provided, otherwise compute from this subset
        if shared_max_strain is not None:
            self.max_strain = shared_max_strain
        else:
            self.max_strain = abs(self.signals).max()
        
        # Use shared min/max if provided, otherwise compute from this subset
        if shared_min is not None and shared_max is not None:
            self.min_parameter = shared_min
            self.max_parameter = shared_max
        else:
            self.min_parameter = self.parameters.min(axis=0).astype(np.float32)
            self.max_parameter = self.parameters.max(axis=0).astype(np.float32)

        self.PSD = self.AdvLIGOPsd(fourier_freq)
        self.signal_rfft = np.fft.rfft(self.signals / TEN_KPC, axis=0)


    def plot_signal_distribution(self, background=True, font_family="Serif", font_name="Times New Roman", fname=None):
        plot_signal_distribution(self.signals/TEN_KPC, generated=False, background=background, font_family=font_family, font_name=font_name, fname=fname)

    def plot_signal_grid(self, n_signals=3, background=True, font_family="sans-serif", font_name="Avenir", fname=None):
        # Collect indices of the signals to plot
        selected_signals = []
        for i in range(n_signals):
            signal = self.__getitem__(i+110)[1].cpu().numpy().flatten()  # Flatten the signal
            selected_signals.append(signal)

        # Convert selected signals to a NumPy array for plotting
        selected_signals = np.array(selected_signals)

        plot_signal_grid(
            signals=selected_signals/TEN_KPC,
            noisy_signals=None,
            max_value=self.max_strain,
            num_cols=1,
            num_rows=1,
            fname=fname,
            background=background,
            generated=False,
            font_family=font_family,
            font_name=font_name
        )

    def plot_parameter_distribution(
        self, 
        param_name: str,
        param_label: Optional[str] = None,
        bins: int = 50,
        fname: Optional[str] = None,
        background: str = "white",
        font_family: str = "sans-serif",
        font_name: str = "Avenir",
        color: Optional[str] = None,
        alpha: float = 0.7,
        show_stats: bool = True
    ):
        """Plot the distribution of a parameter from the dataset.
        
        Args:
            param_name (str): Name of the parameter column to plot
            param_label (Optional[str]): Label for the parameter (LaTeX supported). If None, uses default from PARAMETER_LABELS
            bins (int): Number of histogram bins
            fname (Optional[str]): Filename to save plot
            background (str): Background color theme
            font_family (str): Font family to use
            font_name (str): Specific font name
            color (Optional[str]): Color for the histogram
            alpha (float): Transparency of the histogram bars
            show_stats (bool): Whether to display mean and std on the plot
        """
        if param_name not in self.parameter_names:
            raise ValueError(f"Parameter '{param_name}' not found. Available parameters: {self.parameter_names}")
        
        # Use default label if not provided
        if param_label is None:
            param_label = self.PARAMETER_LABELS.get(param_name, param_name)
        
        # Get parameter column index
        param_idx = self.parameter_names.index(param_name)
        values = self.parameters[:, param_idx]
        
        # Get default range for this parameter
        param_range = self.PARAMETER_RANGES.get(param_name, None)
        
        plot_parameter_distribution(
            values=values,
            param_name=param_name,
            param_label=param_label,
            bins=bins,
            fname=fname,
            background=background,
            font_family=font_family,
            font_name=font_name,
            color=color,
            alpha=alpha,
            show_stats=show_stats,
            param_range=param_range
        )

    def plot_all_parameter_distributions(
        self,
        bins: int = 50,
        fname_prefix: Optional[str] = None,
        background: str = "white",
        font_family: str = "sans-serif",
        font_name: str = "Avenir",
        color: Optional[str] = None,
        alpha: float = 0.7,
        show_stats: bool = True
    ):
        """Plot distributions for all parameters in the dataset.
        
        Args:
            bins (int): Number of histogram bins
            fname_prefix (Optional[str]): Prefix for saved plot filenames (e.g., 'plots/param_')
            background (str): Background color theme
            font_family (str): Font family to use
            font_name (str): Specific font name
            color (Optional[str]): Color for the histogram
            alpha (float): Transparency of the histogram bars
            show_stats (bool): Whether to display mean and std on the plot
        """
        for param_name in self.parameter_names:
            if fname_prefix:
                # Extract extension from prefix if it has one, otherwise default to .png
                import os
                prefix_base, prefix_ext = os.path.splitext(fname_prefix)
                ext = prefix_ext if prefix_ext else '.png'
                fname = f"{prefix_base}{param_name}{ext}"
            else:
                fname = None
            self.plot_parameter_distribution(
                param_name=param_name,
                param_label=None,  # Will use default
                bins=bins,
                fname=fname,
                background=background,
                font_family=font_family,
                font_name=font_name,
                color=color,
                alpha=alpha,
                show_stats=show_stats
            )

    # def plot_parameter_distributions_grid(
    #     self,
    #     bins: int = 25,
    #     fname: Optional[str] = None,
    #     background: str = "white",
    #     font_family: str = "sans-serif",
    #     font_name: str = "Avenir",
    #     color: Optional[str] = None,
    #     alpha: float = 0.8,
    #     figsize: Tuple[float, float] = (20, 5)
    # ):
    #     """Plot distributions for all parameters in a 1x4 grid (one row).
        
    #     Args:
    #         bins (int): Number of histogram bins
    #         fname (Optional[str]): Filename to save plot
    #         background (str): Background color theme
    #         font_family (str): Font family to use
    #         font_name (str): Specific font name
    #         color (Optional[str]): Color for the histogram
    #         alpha (float): Transparency of the histogram bars
    #         figsize (Tuple[float, float]): Figure size in inches
    #     """
    #     # Prepare parameters dictionary
    #     parameters_dict = {}
    #     for param_name in self.parameters.columns:
    #         values = self.parameters[param_name].values
    #         # Apply log transformation for A(km) parameter
    #         if param_name == "A(km)":
    #             values = np.log(values)
    #         parameters_dict[param_name] = values
        
    #     return plot_parameter_distribution_grid(
    #         parameters_dict=parameters_dict,
    #         labels_dict=self.PARAMETER_LABELS,
    #         ranges_dict=self.PARAMETER_RANGES,
    #         bins=bins,
    #         fname=fname,
    #         background=background,
    #         font_family=font_family,
    #         font_name=font_name,
    #         color=color,
    #         alpha=alpha,
    #         figsize=figsize
    #     )
 
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

    def update_snr(self, snr):
        self.rho_target = snr

    @staticmethod
    def calculate_snr(h, Sn, fs=SAMPLING_RATE):
        """Calculate SNR from signal and PSD.
        
        Args:
            h: signal in time domain or frequency domain (FFT)
            Sn: one-sided PSD as array over frequencies
            fs: sampling rate 
        """
        N = len(h)
        df = fs / N
        hf = np.fft.rfft(h)
        integrand = (np.abs(hf)**2) / Sn
        rho2 = 4 * np.sum(integrand) * df
        return np.sqrt(rho2)
    
    @staticmethod
    def calculate_snr_from_fft(hf, Sn, fs=SAMPLING_RATE, N=Y_LENGTH):
        """Calculate SNR directly from pre-computed FFT.
        
        Args:
            hf: signal FFT (already computed)
            Sn: one-sided PSD as array over frequencies
            fs: sampling rate
            N: length of original signal
        """
        df = fs / N
        integrand = (np.abs(hf)**2) / Sn
        rho2 = 4 * np.sum(integrand) * df
        return np.sqrt(rho2)

    def aLIGO_noise(self, seed_offset=0):
        """Add Advanced LIGO noise to the signal.
        
        Args:
            seed_offset (int): Offset for random seed to ensure different noise realizations
            
        Returns:
            np.ndarray: Signal with properly scaled aLIGO noise added
        """
        # Use seed_offset to ensure different noise for different realizations
        # This ensures reproducibility while varying across realizations
        if seed_offset > 0:
            random_state = np.random.RandomState(seed_offset + self._current_epoch * 10000)
            original_state = np.random.get_state()
            np.random.set_state(random_state.get_state())
        
        dataDeltaT = SAMPLING_RATE  # Sampling rate: 4096 Hz
        dataSec = Y_LENGTH * SAMPLING_RATE   # Duration: 256 samples at 4096 Hz
        dataN = int(dataSec / dataDeltaT)  # Number of samples
        
        # Generate noise with proper PSD scaling
        noise = self.rnoise(
            N=dataN,
            delta_t=dataDeltaT,
            one_sided=True,  # Use one-sided spectrum as in R
            pad=1
        ).reshape(1, -1)  # shape (1, Y_LENGTH)

        # Restore original random state if we changed it
        if seed_offset > 0:
            np.random.set_state(original_state)

        # The noise is now properly scaled due to correct PSD implementation
        noise = noise - noise.mean()  # Mean center as in R implementation

        return noise
    
    ### overloads ###
    def __len__(self):
        # Multiply dataset size by number of noise realizations
        return self.signals.shape[1] * self.noise_realizations

    @property
    def shape(self):
        return self.signals.shape
    
    def get_indices(self):
        return self.indices

    def verify_alignment(self):
        """Verify that signals and parameters are properly aligned."""
        print("\nVerifying data alignment:")
        print(f"Number of signals: {self.signals.shape[1]}")
        print(f"Number of parameter sets: {self.parameters.shape[0]}")
        print(f"Parameter names: {self.parameter_names}")
        print(f"First few parameter values:\n{self.parameters[:5]}")
        return True

    def __getitem__(self, idx):
        # Validate index is within bounds
        dataset_size = self.signals.shape[1] * self.noise_realizations
        if idx < 0 or idx >= dataset_size:
            raise IndexError(
                f"Index {idx} is out of range for dataset with {self.signals.shape[1]} base signals "
                f"and {self.noise_realizations} noise realizations (total size: {dataset_size})"
            )
        
        # Map the augmented index to the original signal index
        # If noise_realizations=3 and we have 100 signals:
        # idx 0-99 -> signal 0-99 (realization 0)
        # idx 100-199 -> signal 0-99 (realization 1)
        # idx 200-299 -> signal 0-99 (realization 2)
        original_idx = idx % self.signals.shape[1]
        noise_realization_idx = idx // self.signals.shape[1]
        
        s = self.signals[:, original_idx]
        s = s.reshape(1, -1)

        parameters = self.parameters[original_idx].copy()  # Extract parameter values as a NumPy array
        
        # Note: Log transformations already applied to parameters array in __init__
        # Parameters are: [beta, omega_0, log(A), Ye]
        
        # Normalize all parameters to [-1, 1]
        parameters = self.normalize_parameters(parameters)
        
        parameters = parameters.reshape(1, -1)

        # Use pre-computed FFT to calculate SNR (much faster!)
        hf = self.signal_rfft[:, original_idx]
        rho = self.calculate_snr_from_fft(hf, self.PSD)
        
        # Add different noise each time by using a unique seed based on noise_realization_idx
        n = self.aLIGO_noise(seed_offset=noise_realization_idx)
        
        s = s / 3.086e+22
        d = s + n * (rho / self.rho_target) * 100 # don't really get why it needs to scale by 100. Is there an issue with the noise units m vs. cm?
        s = s * 3.086e+22
        d = d * 3.086e+22

        s_star = self.normalise_signals(s)
        d_star = self.normalise_signals(d)

        return (
            torch.tensor(s_star, dtype=torch.float32, device=DEVICE),
            torch.tensor(d_star, dtype=torch.float32, device=DEVICE),
            torch.tensor(parameters, dtype=torch.float32, device=DEVICE)
        )
    
    def set_snr(self, snr):
        self.rho_target = snr

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
        # Avoid division by zero at f=0 by clipping to a small positive value
        f = np.clip(f, 1e-10, None)
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

        psd = self.PSD
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
