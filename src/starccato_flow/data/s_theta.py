from typing import Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from scipy.fft import ifft

import matplotlib.pyplot as plt

from starccato_flow.plotting.parameters import plot_parameter_distributions, plot_eos_ye_distribution


from ..plotting import plot_signal_distribution, plot_signal_grid, plot_parameter_distribution
from ..utils.defaults_general import BATCH_SIZE, DEVICE, TEN_KPC
from ..utils.defaults_general import SAMPLING_FREQ, Y_LENGTH
from ..utils.defaults_general import PARAMETERS_CSV, SIGNALS_CSV
from ..utils.defaults_plotting import PARAMETER_LABELS, PARAMETER_RANGES
from . import BaseDataset

"""This loads the signal data from the raw simulation outputs from Richers et al (2017) ."""

is_even = (Y_LENGTH % 2 == 0)
half_N = Y_LENGTH // 2 if is_even else (Y_LENGTH - 1) // 2
delta_f = 1 / (Y_LENGTH * SAMPLING_FREQ)
fourier_freq = np.arange(half_N + 1) * delta_f

class sTheta(BaseDataset, Dataset):
    # Import unified parameter labels and ranges from plotting_defaults
    PARAMETER_LABELS = PARAMETER_LABELS
    PARAMETER_RANGES = PARAMETER_RANGES
    
    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        num_epochs: int = Y_LENGTH,
        detector_noise_on: bool = True,
        curriculum: bool = True,
        snr: bool = True,
        start_snr: float = 200.0,
        end_snr: float = 8.0,
        indices: Optional[np.ndarray] = None,
        parameters: list = None,
        shared_min: Optional[np.ndarray] = None,
        shared_max: Optional[np.ndarray] = None,
        shared_max_strain: Optional[float] = None,
        generated: bool = False,
        params_df: pd.DataFrame = None,
        signals_df: pd.DataFrame = None,
        custom_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
        remove_erroneous: bool = True
    ):
        """Initialize the CCSN dataset.
        
        Args:
            batch_size (int): Batch size for data loading
            num_epochs (int): Total number of training epochs for curriculum learning
            detector_noise_on (bool): Whether to add detector noise
            curriculum (bool): Whether to use curriculum learning (SNR decreases from start_snr to end_snr)
            snr (bool): Whether to calculate SNR (for curriculum learning)
            start_snr (float): Starting SNR (Signal-to-Noise Ratio) in dB - used at epoch 0
            end_snr (float): Ending SNR in dB - used at final epoch (curriculum learning target)
            indices (Optional[np.ndarray]): Specific indices to use
            parameters (list): List of parameter names to include. Examples:
                ["beta1_IC_b", "omega_0(rad|s)", "A(km)", "Ye_c_b"] - all parameters
                ["beta1_IC_b", "A(km)"] - subset of parameters
                If None, defaults to ["beta1_IC_b", "omega_0(rad|s)", "A(km)", "Ye_c_b"]
            custom_data (Optional[tuple[np.ndarray, np.ndarray]]): Pre-generated (signals, parameters) arrays.
                signals: shape (signal_length, num_samples) or (num_samples, signal_length)
                parameters: shape (num_samples, num_params)
                If provided, overrides params_df and signals_df.
            remove_erroneous (bool): Whether to remove signals with beta1_IC_b <= 0 (default: True)
        """
        self.batch_size = batch_size
        self._current_epoch = 0
        self.num_epochs = num_epochs
        
        if custom_data is not None:
            # Use custom pre-generated data
            signals_array, params_array = custom_data
            
            # Ensure signals are in shape (signal_length, num_samples)
            # Check against expected signal length Y_LENGTH (typically 256)
            if signals_array.shape[0] == Y_LENGTH:
                # Already in correct format (signal_length, num_samples)
                pass
            elif signals_array.shape[1] == Y_LENGTH:
                # In (num_samples, signal_length) format, transpose it
                signals_array = signals_array.T
            else:
                # Fallback to heuristic: assume larger dimension is signal_length
                if signals_array.shape[0] > signals_array.shape[1]:
                    signals_array = signals_array.T
            
            # Create DataFrames for consistent processing
            signals_df = pd.DataFrame(signals_array.T)  # Transpose to (num_samples, signal_length)
            params_df = pd.DataFrame(params_array, columns=["beta1_IC_b", "omega_0(rad|s)", "A(km)", "Ye_c_b"])
            
        elif generated == False:
            # Load data from CSV files
            params_df = pd.read_csv(PARAMETERS_CSV)
            signals_df = pd.read_csv(SIGNALS_CSV).astype("float32").T
        else:
            # Use provided DataFrames
            params_df = params_df
            signals_df = signals_df
        
        self.detector_noise_on = detector_noise_on
        self.curriculum = curriculum
        self.snr = snr
        self.start_snr = start_snr
        self.end_snr = end_snr

        # Set default parameters if not provided
        if parameters is None:
            parameters = ["beta1_IC_b", "omega_0(rad|s)", "A(km)", "Ye_c_b"]
        self.parameter_names = parameters
        
        if remove_erroneous:
            beta_keep_idx = params_df["beta1_IC_b"].values >= 0
        else:
            beta_keep_idx = np.ones(len(params_df), dtype=bool)

        # keep only the parameters we want
        eos_df = params_df["EOS"]
        params_df = params_df[parameters]
        
        # Remove unusual parameters and corresponding signals using beta positivity.
        keep_idx = beta_keep_idx
        # print(f"Removing {(~keep_idx).sum()} signals with beta1_IC_b <= 0")
        params_df = params_df[keep_idx]
        eos_df = eos_df[keep_idx]
        
        # Convert to numpy array and apply log transformations BEFORE computing min/max
        # This brings all parameters to similar scales for better training
        self.parameters = params_df.values.astype(np.float32)
        self.eos = eos_df

        # Keep track of original indices
        signal_indices = np.where(keep_idx)[0]
        signals_df = signals_df.iloc[keep_idx]
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
            self.eos = self.eos.iloc[indices]
            self.indices = indices

        # Use shared max_strain if provided, otherwise compute from this subset
        if shared_max_strain is not None:
            self.shared_max_strain = shared_max_strain
        else:
            self.shared_max_strain = abs(self.signals).max()
        
        # Use shared min/max if provided, otherwise compute from this subset
        if shared_min is not None and shared_max is not None:
            self.shared_min_theta = shared_min
            self.shared_max_theta = shared_max
        else:
            self.shared_min_theta = self.parameters.min(axis=0).astype(np.float32)
            self.shared_max_theta = self.parameters.max(axis=0).astype(np.float32)

        # Print parameter bounds
        print(f"\n{'='*70}")
        print(f"sTheta Dataset - Parameter Bounds ({len(self.parameter_names)} parameters)")
        print(f"{'='*70}")
        for i, param_name in enumerate(self.parameter_names):
            print(f"{param_name:20s}: [{self.shared_min_theta[i]:12.6f}, {self.shared_max_theta[i]:12.6f}]")
        print(f"{'='*70}\n")

        self.PSD = self.AdvLIGOPsd(fourier_freq)
        self.signal_rfft = np.fft.rfft(self.signals / TEN_KPC, axis=0)

    @property
    def eos_values(self) -> Optional[np.ndarray]:
        """Extract EOS (Equation of State) values from parameters.
        
        Searches for a parameter containing 'EOS' (case-insensitive) and returns
        those values as strings.
        
        Returns:
            Optional[np.ndarray]: Array of EOS values as strings, or None if no EOS parameter found.
        """
        # Find parameter with "EOS" in the name (case-insensitive)
        eos_param_names = [p for p in self.parameter_names if "EOS" in p.upper()]
        
        if eos_param_names:
            eos_param_name = eos_param_names[0]  # Get first EOS parameter
            eos_idx = self.parameter_names.index(eos_param_name)
            return self.parameters[:, eos_idx].astype(str)
        
        return None

    def plot_signal_distribution(self, background=None, font_family="serif", font_name="Times New Roman", fname=None, beta_min=None, beta_max=None, figsize=tuple[float, float], axes: Optional[plt.Axes] = None):
        beta = self.parameters[:, 0]
        mask = np.ones(len(beta), dtype=bool)
        if beta_min is not None:
            mask &= beta >= beta_min
        if beta_max is not None:
            mask &= beta <= beta_max

        print(f"Plotting signal distribution for {mask.sum()} signals with beta in [{beta_min}, {beta_max}]")
        plot_signal_distribution(self.signals[:,mask]/TEN_KPC, generated=False, background=background, font_family=font_family, font_name=font_name, fname=fname, figsize=figsize, axes=axes)

    def plot_parameter_distributions(self, fname, font_family="sans-serif", font_name="Avenir"):
        params_dict = {
            param: self.parameters[:, self.parameter_names.index(param)] 
            for param in self.parameter_names
        }

        plot_parameter_distributions(
            parameters_dict=params_dict,
            fname=fname,
            font_family=font_family,
            font_name=font_name
        )

    def plot_random_signals_grid(self, n_signals, n_rows, n_cols, background="white", font_family="sans-serif", font_name="Avenir", fname=None, figsize=tuple[float, float]):
        np.random.seed(42)
        random_indices = np.random.choice(len(self), size=n_signals, replace=False)
        selected_signals = []
        for idx in random_indices:
            signal = self.signals[:,idx]
            selected_signals.append(signal)

        selected_signals = np.array(selected_signals)
        # Plot in 4x4 grid
        plot_signal_grid(
            signals=selected_signals/TEN_KPC,
            n_cols=n_rows,
            n_rows=n_cols,
            fname=fname,
            generated=False,
            background=background,
            font_family=font_family,
            font_name=font_name,
            figsize=figsize
        );

    # def plot_signal_grid(self, n_signals=3, background=True, font_family="sans-serif", font_name="Avenir", fname=None):
    #     # Collect indices of the signals to plot
    #     selected_signals = []
    #     for i in range(n_signals):
    #         signal = self.__getitem__(i+110)[1].cpu().numpy().flatten()  # Flatten the signal
    #         selected_signals.append(signal)

    #     # Convert selected signals to a NumPy array for plotting
    #     selected_signals = np.array(selected_signals)

    #     plot_signal_grid(
    #         signals=selected_signals/TEN_KPC,
    #         noisy_signals=None,
    #         max_value=self.shared_max_strain,
    #         num_cols=1,
    #         num_rows=1,
    #         fname=fname,
    #         background=background,
    #         generated=False,
    #         font_family=font_family,
    #         font_name=font_name
    #     )

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


    def plot_eos_ye_distribution(self, fname, font_family="sans-serif", font_name="Avenir"):
        plot_eos_ye_distribution(
            eos_values=self.eos,
            ye_values=self.parameters[:, self.parameter_names.index("Ye_c_b")],
            fname=fname,
            background="white",
            font_family=font_family,
            font_name=font_name,
            alpha=0.6,
            point_size=50
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
        str += f"Signal Dataset max value: {self.shared_max_strain}\n"
        # str += f"Signal Dataset max parameter value: {self.max_parameter_value}\n"
        str += f"Signal Dataset shape: {self.signals.shape}\n"
        str += f"Parameter Dataset shape: {self.parameters.shape}\n"

    def get_curriculum_snr(self) -> float:
        """Get the current SNR based on curriculum learning schedule.
        
        SNR linearly interpolates from start_snr (at epoch 0) to end_snr (at final epoch).
        If curriculum learning is disabled, returns end_snr.
        
        Returns:
            float: Current SNR target in dB
        """
        if not self.curriculum:
            return self.end_snr
        
        progress = self._current_epoch / max(self.num_epochs - 1, 1)
        current_snr = self.start_snr + progress * (self.end_snr - self.start_snr)
        return current_snr

    def update_snr(self, snr: float) -> None:
        """Update the SNR target (disables curriculum learning).
        
        Args:
            snr (float): Target SNR in dB
        """
        self.end_snr = snr
        self.curriculum = False

    @staticmethod
    def calculate_snr(h, Sn, fs=SAMPLING_FREQ):
        """Calculate SNR from signal and PSD.
        
        Args:
            h: signal in time domain or frequency domain (FFT)
            Sn: one-sided PSD as array over frequencies
            fs: sampling rate 
        """
        N = len(h)
        df = fs / N
        hf = np.fft.rfft(h)
        # Add small epsilon to avoid divide-by-zero warnings from near-zero PSD values
        Sn_safe = np.maximum(Sn, np.max(Sn) * 1e-10)
        integrand = (np.abs(hf)**2) / Sn_safe
        rho2 = 4 * np.sum(integrand) * df
        return np.sqrt(rho2)
    
    @staticmethod
    def calculate_snr_from_fft(hf, Sn, fs=SAMPLING_FREQ, N=Y_LENGTH):
        """Calculate SNR directly from pre-computed FFT.
        
        Args:
            hf: signal FFT (already computed)
            Sn: one-sided PSD as array over frequencies
            fs: sampling rate
            N: length of original signal
        """
        df = fs / N
        # Add small epsilon to avoid divide-by-zero warnings from near-zero PSD values
        Sn_safe = np.maximum(Sn, np.max(Sn) * 1e-10)
        integrand = (np.abs(hf)**2) / Sn_safe
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
        
        dataDeltaT = 1 / SAMPLING_FREQ  # Sampling rate: 4096 Hz
        dataSec = Y_LENGTH / SAMPLING_FREQ   # Duration: 256 samples at 4096 Hz
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
        return self.signals.shape[1]

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
        if idx < 0 or idx >= self.signals.shape[1]:
            raise IndexError(
                f"Index {idx} is out of range for dataset with {self.signals.shape[1]} signals"
            )
        
        s = self.signals[:, idx]
        s = s.reshape(1, -1)

        parameters = self.parameters[idx].copy()  # Extract parameter values as a NumPy array
        
        # Note: Log transformations already applied to parameters array in __init__
        # Parameters are: [beta, omega_0, log(A), Ye]
        
        # Normalize all parameters to [-1, 1]
        parameters = self.normalize_parameters(parameters)
        
        parameters = parameters.reshape(1, -1)

        # Use pre-computed FFT to calculate SNR (much faster!)
        hf = self.signal_rfft[:, idx]
        rho = self.calculate_snr_from_fft(hf, self.PSD)
        
        # Add noise only if self.detector_noise_on is True
        if self.detector_noise_on:
            # Add noise with a consistent seed per signal index
            n = self.aLIGO_noise(seed_offset=idx)
            
            # Get current SNR target (from curriculum learning)
            rho_target = self.get_curriculum_snr()
            
            s = s / 3.086e+22
            d = s + n * (rho / rho_target) * 100 # don't really get why it needs to scale by 100. Is there an issue with the noise units m vs. cm?
            s = s * 3.086e+22
            d = d * 3.086e+22
        else:
            # No noise: d = s (clean signal)
            d = s

        s_star = self.normalise_signals(s)
        d_star = self.normalise_signals(d)

        return (
            torch.tensor(s_star, dtype=torch.float32, device=DEVICE),
            torch.tensor(d_star, dtype=torch.float32, device=DEVICE),
            torch.tensor(parameters, dtype=torch.float32, device=DEVICE)
        )
    
    def set_snr(self, snr: float) -> None:
        """Set a fixed SNR target (disables curriculum learning).
        
        Args:
            snr (float): Target SNR in dB
        """
        self.end_snr = snr
        self.curriculum = False

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