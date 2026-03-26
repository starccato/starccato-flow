"""Multi-channel CCSN dataset for detector network analysis and sky localization.

This module only works with generated data (not raw CCSN CSV files).
"""

from typing import Optional, List, Tuple
import importlib
import numpy as np
import torch
from torch.utils.data import Dataset

from .s_theta_old import CCSNData
from ..localisation.supernovae import CCSN
from ..localisation.supernovae import CCSNLocations
from ..utils.defaults import DEVICE, Y_LENGTH, BATCH_SIZE, TEN_KPC, SAMPLING_RATE

class hThetaMulti(Dataset):
    """Multi-channel CCSN dataset for sky localization with generated data only.
    
    This class handles:
    - Multiple detector channels (H1, L1, V1, etc.)
    - Sky location parameters (RA, Dec, distance)
    - Antenna pattern projections for each detector
    - Detector-specific noise
    
    Note: This class only works with generated signals (custom_data), not raw CCSN CSV files.
    """
    
    def __init__(
        self,
        detectors: List[str] = ['H1', 'L1', 'V1'],
        batch_size: int = BATCH_SIZE,
        noise: bool = True,
        s: Optional[np.ndarray] = None,
        max_strain: Optional[float] = None,
        theta: Optional[np.ndarray] = None,
        min_theta: Optional[np.ndarray] = None,
        max_theta: Optional[np.ndarray] = None,
        ra: Optional[np.ndarray] = None,
        dec: Optional[np.ndarray] = None,
        d: Optional[np.ndarray] = None,
        random_polarization: bool = True,
        gps_time: float = 1457654242.0,
    ):
        """Initialize multi-channel CCSN dataset with generated data."""
        self.batch_size = batch_size
        self.noise = noise
        self.s = s
        self.max_strain = max_strain
        self.theta = theta
        self.min_theta = min_theta
        self.max_theta = max_theta
        self.min_theta = None
        self.max_theta = None
        self.ra = ra
        self.dec = dec
        self.d = d
        self.random_polarization = random_polarization
        self.gps_time = float(gps_time)
        
        # Multi-detector setup
        self.detectors = detectors
        self.num_detectors = len(detectors)
        bilby_detector = importlib.import_module("bilby.gw.detector")
        self.ifos = [bilby_detector.get_empty_interferometer(det_name) for det_name in detectors]
        
        # Set up PSD for noise generation
        is_even = (Y_LENGTH % 2 == 0)
        half_N = Y_LENGTH // 2 if is_even else (Y_LENGTH - 1) // 2
        delta_f = 1 / (Y_LENGTH * SAMPLING_RATE)
        fourier_freq = np.arange(half_N + 1) * delta_f
        self.PSD = self.AdvLIGOPsd(fourier_freq)
        self.signal_rfft = np.fft.rfft(self.signals / TEN_KPC, axis=0)
        
        # Project signals to multiple detectors
        self.multi_channel_signals = self._project_to_detectors()

        # If not externally provided, normalize using the multi-channel dynamic range.
        # if self.max_strain is None:
        #     self.max_strain = abs(self.multi_channel_signals).max()
        
        # Update parameter dimension if including sky params
        # if self.include_sky_params:
        #     # Concatenate physical params with sky params
        #     self.parameters = np.concatenate([self.parameters, self.sky_params], axis=1)
            
        #     # Update min/max for normalization
        #     if shared_min is not None and shared_max is not None:
        #         # Extend shared min/max with sky param ranges
        #         sky_min = self.sky_params.min(axis=0)
        #         sky_max = self.sky_params.max(axis=0)
        #         self.min_parameter = np.concatenate([self.min_parameter, sky_min])
        #         self.max_parameter = np.concatenate([self.max_parameter, sky_max])
        #     else:
        #         self.min_parameter = self.parameters.min(axis=0).astype(np.float32)
        #         self.max_parameter = self.parameters.max(axis=0).astype(np.float32)
        # else:
        #     # No sky params, just use physical params
        #     if shared_min is None or shared_max is None:
        #         self.min_parameter = self.parameters.min(axis=0).astype(np.float32)
        #         self.max_parameter = self.parameters.max(axis=0).astype(np.float32)
        
        self.param_dim = self.parameters.shape[1]
        
        print(f"\n=== Multi-Channel Dataset Info ===")
        print(f"Detectors: {', '.join(self.detectors)} ({self.num_detectors} channels)")
        print(f"Signals per channel: {self.signals.shape[1]}")
        print(f"Multi-channel shape: {self.multi_channel_signals.shape}")
        print(f"Parameter dimension: {self.param_dim}")
        if self.include_sky_params:
            print(f"Parameters: physical (4) + sky (3) = {self.param_dim}")
            print(f"Sky parameters included: RA, Dec, distance")
        print("=" * 50)
    
    def AdvLIGOPsd(self, f):
        """Advanced LIGO power spectral density."""
        # Avoid division by zero at f=0 by clipping to a small positive value
        f = np.clip(f, 1e-10, None)
        x = f / 215
        x2 = x * x
        psd = 1e-49 * (pow(x, - 4.14) - 5 / x2 + 111 * (1 - x2 + 0.5 * x2 * x2) / (1 + 0.5 * x2))
        # The upper bound is 2e10 times the minimum value
        cutoff = np.nanmin(psd) * 2e10
        psd[(psd > cutoff) | np.isnan(psd)] = cutoff
        return psd
    
    def aLIGO_noise(self, seed_offset: int = 0) -> np.ndarray:
        """Generate aLIGO noise realization."""
        np.random.seed(seed_offset)
        return self.rnoise(Y_LENGTH, SAMPLING_RATE, one_sided=True, pad=1)
    
    def rnoise(self, N: int, delta_t: float, one_sided: bool = True, pad: int = 1) -> np.ndarray:
        """Generate colored noise with given PSD."""
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
        from scipy.fft import ifft
        noise = np.real(ifft(noise_ft)) / N

        if pad > 1:
            start = np.random.randint(0, orig_N * (pad - 1))
            noise = noise[start:start + orig_N]

        return noise
    
    @staticmethod
    def calculate_snr_from_fft(hf, Sn, fs=SAMPLING_RATE, N=Y_LENGTH):
        """Calculate SNR directly from pre-computed FFT."""
        df = fs / N
        integrand = (np.abs(hf)**2) / Sn
        rho2 = 4 * np.sum(integrand) * df
        return np.sqrt(rho2)
    
    def normalise_signals(self, signal):
        """Normalize signals by dividing by max strain."""
        return signal / self.max_strain
    
    def denormalise_signals(self, signal):
        """Denormalize signals by multiplying by max strain."""
        return signal * self.max_strain
    
    def normalize_parameters(self, params):
        """Normalize parameters to [-1, 1] range."""
        params_norm = params.copy()
        param_range = self.max_parameter - self.min_parameter
        params_norm = 2 * (params - self.min_parameter) / param_range - 1
        return params_norm
    
    def denormalize_parameters(self, params_norm):
        """Denormalize parameters from [-1, 1] back to original ranges."""
        params = params_norm.copy()
        param_range = self.max_parameter - self.min_parameter
        params = (params_norm + 1) / 2 * param_range + self.min_parameter
        return params
    
    # def _generate_sky_params(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
    #     """Generate sky parameters (RA, Dec, distance) for n_samples.
        
    #     Args:
    #         n_samples: Number of samples to generate
    #         seed: Random seed
            
    #     Returns:
    #         Array of shape (n_samples, 3) with [RA, Dec, distance]
    #     """
    #     # Generate galactic locations
    #     # self.ccsn_locations.generate_locations(n_samples, seed=seed)
        
    #     # Get sky params
    #     sky_params = self.ccsn_locations.get_sky_params()
        
    #     print(f"✓ Generated {n_samples} sky locations")
    #     print(f"  RA range: [{np.rad2deg(sky_params[:, 0].min()):.1f}°, {np.rad2deg(sky_params[:, 0].max()):.1f}°]")
    #     print(f"  Dec range: [{np.rad2deg(sky_params[:, 1].min()):.1f}°, {np.rad2deg(sky_params[:, 1].max()):.1f}°]")
    #     print(f"  Distance range: [{sky_params[:, 2].min():.2f}, {sky_params[:, 2].max():.2f}] kpc")
        
    #     return sky_params
    
    def _project_to_detectors(self) -> np.ndarray:
        """Project single-channel signals to multiple detectors using antenna patterns.
        
        Returns:
            Array of shape (n_samples, num_detectors, signal_length)
        """
        n_samples = self.signals.shape[1]
        multi_channel = np.zeros((n_samples, self.num_detectors, Y_LENGTH), dtype=np.float32)
        t = np.arange(Y_LENGTH) * SAMPLING_RATE
        h_cross = np.zeros(Y_LENGTH, dtype=np.float32) # assume cross-polarization is zero for these templates
        rng = np.random.default_rng(self.seed)
        
        print(f"Projecting signals to {self.num_detectors} detectors...")
        
        for i in range(n_samples):
            h_plus = self.signals[:, i]  # Shape: (Y_LENGTH,)

            psi = rng.uniform(0, np.pi) if self.random_polarization else 0.0 # random polarisation angle in [0, pi]
            gps = self.gps_time

            # compute detector delays first; use relative stream delays.
            dts = np.array(
                [ifo.time_delay_from_geocenter(self.ra[i], self.dec[i], gps) for ifo in self.ifos],
                dtype=np.float64,
            )
            dt_min = dts.min()
            relative_dts = dts - dt_min

            # Distance scaling relative to 10 kpc reference waveforms.
            scale = 10.0 / max(self.d[i], 1e-8)
            
            for j, (ifo, dt_rel) in enumerate(zip(self.ifos, relative_dts)):
                # compute bilby antenna response patterns
                F_plus = ifo.antenna_response(
                    self.ra[i],
                    self.dec[i],
                    gps,
                    psi,
                    mode='plus',
                )
                F_cross = ifo.antenna_response(
                    self.ra[i],
                    self.dec[i],
                    gps,
                    psi,
                    mode='cross',
                )

                # Push later-arriving detectors right in stream (relative delay).
                h_plus_shifted = np.interp(t - dt_rel, t, h_plus, left=0.0, right=0.0)
                h_cross_shifted = np.interp(t - dt_rel, t, h_cross, left=0.0, right=0.0)

                # Combine + and x polarizations (x set to 0 by default template).
                h_signal = scale * (F_plus * h_plus_shifted + F_cross * h_cross_shifted)
                multi_channel[i, j, :] = h_signal.astype(np.float32)
        
        print(f"✓ Projected signals to {self.num_detectors} detectors")
        
        return multi_channel
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Get a single sample (multi-channel signal, parameters).
        
        Returns:
            Tuple of:
                - multi_channel_signal: Shape (num_detectors, signal_length)
                - parameters: Shape (param_dim,)
        """
        # Map idx to actual data index considering noise realizations
        noise_realization_idx = idx % self.noise_realizations
        original_idx = idx // self.noise_realizations
        
        # Get multi-channel signal (already projected)
        # Copy so normalization/noise operations do not mutate cached dataset arrays.
        multi_signal = self.multi_channel_signals[original_idx].copy()  # Shape: (num_detectors, Y_LENGTH)
        
        # Get parameters
        parameters = self.parameters[original_idx]
        
        # Add noise to each detector channel if enabled
        if self.noise:
            for j in range(self.num_detectors):
                # Get signal for this detector
                s = multi_signal[j:j+1, :]  # Shape: (1, Y_LENGTH)
                
                # Compute SNR (using base class method)
                s_normalized = s / TEN_KPC
                hf = np.fft.rfft(s_normalized, axis=1)[0]
                rho = self.calculate_snr_from_fft(hf, self.PSD)
                
                # Generate detector-specific noise
                n = self.aLIGO_noise(seed_offset=noise_realization_idx + j * 1000)
                
                # Add noise with target SNR
                s_normalized = s_normalized / 3.086e+22
                d_normalized = s_normalized + n * (rho / self.rho_target) * 100
                d = d_normalized * 3.086e+22
                
                # Normalize
                multi_signal[j:j+1, :] = self.normalise_signals(d)
        else:
            # No noise: normalize clean signals
            for j in range(self.num_detectors):
                s = multi_signal[j:j+1, :]
                multi_signal[j:j+1, :] = self.normalise_signals(s)
        
        # Normalize parameters
        params_normalized = self.normalize_parameters(parameters.reshape(1, -1))[0]
        
        return (
            torch.tensor(multi_signal, dtype=torch.float32, device=DEVICE),
            torch.tensor(params_normalized, dtype=torch.float32, device=DEVICE)
        )
    
    def get_single_detector_signal(self, idx: int, detector: str) -> torch.Tensor:
        """Get signal from a single detector.
        
        Args:
            idx: Sample index
            detector: Detector name (e.g., 'H1', 'L1', 'V1')
            
        Returns:
            Signal tensor of shape (signal_length,)
        """
        if detector not in self.detectors:
            raise ValueError(f"Detector {detector} not in {self.detectors}")
        
        det_idx = self.detectors.index(detector)
        original_idx = idx // self.noise_realizations
        
        signal = self.multi_channel_signals[original_idx, det_idx, :]
        signal_normalized = self.normalise_signals(signal.reshape(1, -1))
        
        return torch.tensor(signal_normalized, dtype=torch.float32, device=DEVICE).squeeze()
    
    def __len__(self) -> int:
        """Return total number of samples (including noise realizations)."""
        return self.signals.shape[1] * self.noise_realizations
    
    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch number for curriculum learning."""
        self._current_epoch = epoch
        if self.curriculum:
            self.rho_target = -1 * (epoch / self.num_epochs) * (abs(self.start_snr - self.end_snr)) + self.start_snr
    
    @property
    def current_epoch(self) -> int:
        """Get the current epoch number."""
        return self._current_epoch
    
    def __repr__(self) -> str:
        return (f"CCSNDataMultiChannel({self.signals.shape[1]} samples × {self.num_detectors} detectors)\n"
                f"  Detectors: {', '.join(self.detectors)}\n"
                f"  Multi-channel shape: {self.multi_channel_signals.shape}\n"
                f"  Parameters: {self.param_dim}D")
