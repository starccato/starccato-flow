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
from ..utils.defaults import SAMPLING_RATE, Y_LENGTH
from ..utils.defaults import PARAMETERS_CSV, SIGNALS_CSV, TIME_CSV

"""This loads the signal data from the raw simulation outputs from Richers et al (20XX) ."""

class CCSNSNRData(Dataset):
    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        num_epochs: int = 256,
        frac: float = 1.0,
        train: bool = True,
        noise: bool = True,
        curriculum: bool = True,
        snr: bool = True,
        start_snr: int = 100,
        end_snr: int = 10,
        rho_target: int = 10,
        indices: Optional[np.ndarray] = None,
        multi_param: bool = True,
        noise_realizations: int = 1
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
            noise_realizations (int): Number of different noise realizations per signal (multiplies dataset size)
        """
        self.batch_size = batch_size
        self._current_epoch = 0
        self.num_epochs = num_epochs
        self.parameters = pd.read_csv(PARAMETERS_CSV)
        self.signals = pd.read_csv(SIGNALS_CSV).astype("float32").T
        self.signals.index = [i for i in range(len(self.signals.index))]
        self.noise = noise
        self.curriculum = curriculum
        self.snr = snr
        self.start_snr = start_snr
        self.end_snr = end_snr
        self.rho_target = rho_target
        self.noise_realizations = noise_realizations

        assert (
            self.signals.shape[0] == self.parameters.shape[0],
            "Signals and parameters must have the same number of rows (the number of signals)",
        )

        # Sample a fraction of the data if requested
        if frac < 1:
            init_shape = self.signals.shape
            n_signals = int(frac * self.signals.shape[0])
            
            # Use the same random indices for both signals and parameters
            random_indices = np.random.choice(self.signals.shape[0], n_signals, replace=False)
            self.signals = self.signals.iloc[random_indices]
            self.parameters = self.parameters.iloc[random_indices]
            
            print(f"Sampled {n_signals} signals out of {init_shape[0]}")
        
        # Remove unusual parameters and corresponding signals
        keep_idx = self.parameters["beta1_IC_b"] > 0
        # print(f"Removing {(~keep_idx).sum()} signals with beta1_IC_b <= 0")
        self.parameters = self.parameters[keep_idx]

        parameter_set = ["beta1_IC_b", "A(km)"]
        # parameter_set = ["beta1_IC_b"]

        if multi_param:
            parameter_set = ["beta1_IC_b", "A(km)", "Ye_c_b", "omega_0(rad|s)"]
        else: 
            parameter_set = ["beta1_IC_b"]

        # keep only the parameters we want
        self.parameters = self.parameters[parameter_set]

        # akm = pd.get_dummies(self.parameters["A(km)"], prefix="A")
        # self.parameters = pd.concat([self.parameters.drop(columns=["A(km)"]), akm], axis=1)

        # Equal frequency binning for beta1_IC_b
        # if "beta1_IC_b" in parameter_set:
        #     self.parameters['beta1_IC_b'] = pd.qcut(
        #         self.parameters['beta1_IC_b'], q=3, labels=False
        #     )
        #     beta_bins = pd.get_dummies(self.parameters['beta1_IC_b'], prefix="beta_bin")
        #     self.parameters = pd.concat([self.parameters.drop(columns=["beta1_IC_b"]), beta_bins], axis=1)

        # if multi_param:
            # # one hot encode A(km)
            # akm = pd.get_dummies(self.parameters["A(km)"], prefix="A")
            # self.parameters = pd.concat([self.parameters.drop(columns=["A(km)"]), akm], axis=1)

            # one hot encode EOS
            # eos = pd.get_dummies(self.parameters["EOS"], prefix="EOS")
            # self.parameters = pd.concat([self.parameters.drop(columns=["EOS"]), eos], axis=1)

        # Keep track of original indices
        signal_indices = np.where(keep_idx)[0]
        self.signals = self.signals[keep_idx]
        self.signals = self.signals.values.T

        # print(f"Processing {self.signals.shape[1]} signals")
        # print(f"Parameters shape: {self.parameters.shape}")
        assert self.signals.shape[1] == len(self.parameters), "Signal and parameter counts don't match!"

        ### flatten signals and take last 256 timestamps
        temp_data = np.empty(shape=(256, 0)).astype("float32")

        # Store original signal indices for verification
        self.original_indices = []

        for i in range(0, self.signals.shape[1]):
            signal = self.signals[:, i]
            signal = signal.reshape(1, -1)

            cut_signal = signal[:, int(len(signal[0]) - 256) : len(signal[0])]
            temp_data = np.insert(
                temp_data, temp_data.shape[1], cut_signal, axis=1
            )
            self.original_indices.append(signal_indices[i])

        self.signals = temp_data
        
        # Verify alignment
        assert self.signals.shape[1] == len(self.parameters), "Signal and parameter counts don't match after processing!"

        if indices is not None:
            if train:
                self.signals = self.signals[:, indices]
                self.parameters = self.parameters.iloc[indices]
                self.indices = indices
            else:
                self.signals = self.signals[:, indices]
                self.parameters = self.parameters.iloc[indices]
                self.indices = indices

        self.max_strain = abs(self.signals).max()
        self.ylim_signal = (self.signals[:, :].min(), self.signals[:, :].max())

    def plot_signal_distribution(self, background=True, font_family="Serif", font_name="Times New Roman", fname=None):
        plot_signal_distribution(self.signals, generated=False, background=background, font_family=font_family, font_name=font_name, fname=fname)

    def plot_signal_grid(self, n_signals=3, background=True, font_family="Serif", font_name="Times New Roman", fname=None):
        # Collect indices of the signals to plot
        selected_signals = []
        for i in range(n_signals):
            signal = self.__getitem__(i+100)[1].cpu().numpy().flatten()  # Flatten the signal
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

    def update_snr(self, snr):
        self.rho_target = snr

    @staticmethod
    def calculate_snr(h, Sn, fs=SAMPLING_RATE):
        """
        Args:
            h: signal
            sn: one-sided PDS as array over frequencies
            fs: sampling rate 
        """
        N = len(h)
        df = fs / N
        hf = np.fft.rfft(h)
        freqs = np.fft.rfftfreq(N, 1/fs)
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

        # Restore original random state if we changed it
        if seed_offset > 0:
            np.random.set_state(original_state)

        # The noise is now properly scaled due to correct PSD implementation
        noise = noise - noise.mean()  # Mean center as in R implementation

        return noise

    def normalise_signals(self, signal):
        normalised_signal = signal / self.max_strain
        return normalised_signal
    
    def normalise_parameters(self, parameters):
        normalised_parameters = parameters / self.max_parameter_value
        return normalised_parameters

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
        print(f"Number of parameter sets: {len(self.parameters)}")
        print(f"Parameter columns: {self.parameters.columns.tolist()}")
        print(f"First few parameter values:\n{self.parameters.head()}")
        return True

    def __getitem__(self, idx):
        # Map the augmented index to the original signal index
        # If noise_realizations=3 and we have 100 signals:
        # idx 0-99 -> signal 0-99 (realization 0)
        # idx 100-199 -> signal 0-99 (realization 1)
        # idx 200-299 -> signal 0-99 (realization 2)
        original_idx = idx % self.signals.shape[1]
        noise_realization_idx = idx // self.signals.shape[1]
        
        s = self.signals[:, original_idx]
        s = s.reshape(1, -1)

        parameters = self.parameters.iloc[original_idx].values  # Extract parameter values as a NumPy array
        parameters = parameters.astype(np.float32)  # Ensure parameters are float32
        parameters = parameters.reshape(1, -1)


        is_even = (Y_LENGTH % 2 == 0)
        half_N = Y_LENGTH // 2 if is_even else (Y_LENGTH - 1) // 2
        delta_f = 1 / (Y_LENGTH * SAMPLING_RATE)
        fourier_freq = np.arange(half_N + 1) * delta_f

        Sn = self.AdvLIGOPsd(fourier_freq)

        # Ensure signal is a proper numpy array (not lazy object)
        s_array = np.asarray(s / 3.086e+22).flatten()
        rho = self.calculate_snr(s_array, Sn)
        
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
