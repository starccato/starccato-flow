"""Multi-channel CCSN dataset for detector network analysis and sky localization.

This module only works with generated data (not raw CCSN CSV files).
"""

from typing import Optional, List, Tuple
import importlib
import numpy as np
import torch
from torch.utils.data import Dataset

from .s_theta import sTheta
from ..localisation.supernovae import Supernovae
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

    # Fixed physical bounds for sky parameters: [ra, dec, distance_kpc, psi]
    SKY_MIN = np.array([-np.pi, -np.pi / 2.0, 0.0, 0.0], dtype=np.float32)
    SKY_MAX = np.array([np.pi, np.pi / 2.0, 20.0, np.pi], dtype=np.float32)
    
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
        seed: int = 99,
        noise_realizations: int = 1,
        rho_target: float = 10.0,
        curriculum: bool = False,
        num_epochs: int = 1,
        start_snr: float = 100.0,
        end_snr: float = 10.0,
    ):
        """Initialize multi-channel CCSN dataset with generated data."""
        self.batch_size = batch_size
        self.noise = noise
        self.s = self._coerce_signal_matrix(s)
        self.max_strain = max_strain
        self.theta = self._coerce_theta_matrix(theta)
        self.ra = np.asarray(ra, dtype=np.float32) if ra is not None else None
        self.dec = np.asarray(dec, dtype=np.float32) if dec is not None else None
        self.d = np.asarray(d, dtype=np.float32) if d is not None else None
        self.random_polarization = random_polarization
        self.gps_time = float(gps_time)
        self.seed = seed
        self.noise_realizations = noise_realizations
        self.rho_target = rho_target
        self.curriculum = curriculum
        self.num_epochs = num_epochs
        self.start_snr = start_snr
        self.end_snr = end_snr
        self._current_epoch = 0
        self.include_sky_params = True

        n_samples = self.s.shape[1]
        if self.ra is None or self.dec is None or self.d is None:
            raise ValueError("hThetaMulti requires ra, dec, and d arrays.")
        if self.ra.shape[0] != n_samples or self.dec.shape[0] != n_samples or self.d.shape[0] != n_samples:
            raise ValueError(
                "Sky parameter lengths must match number of signal samples. "
                f"Got s={n_samples}, ra={self.ra.shape[0]}, dec={self.dec.shape[0]}, d={self.d.shape[0]}."
            )

        rng = np.random.default_rng(self.seed)
        if self.random_polarization:
            self.polar_angle = rng.uniform(0.0, np.pi, size=n_samples).astype(np.float32)
        else:
            self.polar_angle = np.zeros(n_samples, dtype=np.float32)

        sky_params = np.column_stack([self.ra, self.dec, self.d, self.polar_angle]).astype(np.float32)
        self.parameters = np.concatenate([self.theta, sky_params], axis=1)

        theta_dim = self.theta.shape[1]
        if min_theta is not None and max_theta is not None:
            base_min = np.asarray(min_theta, dtype=np.float32)
            base_max = np.asarray(max_theta, dtype=np.float32)
        else:
            base_min = self.theta.min(axis=0).astype(np.float32)
            base_max = self.theta.max(axis=0).astype(np.float32)

        if base_min.shape[0] == self.parameters.shape[1] and base_max.shape[0] == self.parameters.shape[1]:
            self.min_theta = base_min
            self.max_theta = base_max
        elif base_min.shape[0] == theta_dim and base_max.shape[0] == theta_dim:
            # Append fixed sky bounds for consistent normalization across epochs/runs.
            self.min_theta = np.concatenate([base_min, self.SKY_MIN]).astype(np.float32)
            self.max_theta = np.concatenate([base_max, self.SKY_MAX]).astype(np.float32)
        else:
            raise ValueError(
                "min_theta/max_theta dimensions do not match either theta or combined parameter dimensions. "
                f"theta_dim={theta_dim}, combined_dim={self.parameters.shape[1]}, "
                f"min_dim={base_min.shape[0]}, max_dim={base_max.shape[0]}"
            )

        if self.max_strain is None:
            self.max_strain = np.max(np.abs(self.s))
        
        # Multi-detector setup
        self.detectors = detectors
        self.num_detectors = len(detectors)
        bilby_detector = importlib.import_module("bilby.gw.detector")
        self.ifos = [bilby_detector.get_empty_interferometer(det_name) for det_name in detectors]
        
        # Set up PSD for noise generation.
        is_even = (Y_LENGTH % 2 == 0)
        half_N = Y_LENGTH // 2 if is_even else (Y_LENGTH - 1) // 2
        delta_f = 1 / (Y_LENGTH * SAMPLING_RATE)
        fourier_freq = np.arange(half_N + 1) * delta_f
        self.PSD = self.AdvLIGOPsd(fourier_freq)
        self.signal_rfft = np.fft.rfft(self.s / TEN_KPC, axis=0)
        
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
        print(f"Signals per channel: {self.s.shape[1]}")
        print(f"Multi-channel shape: {self.multi_channel_signals.shape}")
        print(f"Parameter dimension: {self.param_dim}")
        if self.include_sky_params:
            print(f"Parameters include theta + sky: [ra, dec, d, polar_angle]")
        print("=" * 50)

    @staticmethod
    def _coerce_signal_matrix(s):
        """Convert incoming signals to shape (Y_LENGTH, n_samples)."""
        if s is None:
            raise ValueError("hThetaMulti requires non-empty 's' input.")

        if isinstance(s, list):
            if len(s) == 0:
                raise ValueError("hThetaMulti received an empty signal list.")
            s = torch.cat([
                item.detach().cpu() if isinstance(item, torch.Tensor) else torch.as_tensor(item)
                for item in s
            ], dim=0).numpy()
        elif isinstance(s, torch.Tensor):
            s = s.detach().cpu().numpy()
        else:
            s = np.asarray(s)

        if s.ndim != 2:
            raise ValueError(f"Expected 2D signal matrix, got shape {s.shape}.")

        if s.shape[0] != Y_LENGTH and s.shape[1] == Y_LENGTH:
            s = s.T
        if s.shape[0] != Y_LENGTH:
            raise ValueError(f"Signal matrix must have Y_LENGTH={Y_LENGTH} rows; got shape {s.shape}.")

        return s.astype(np.float32)

    @staticmethod
    def _coerce_theta_matrix(theta):
        """Convert incoming parameters to shape (n_samples, n_params)."""
        if theta is None:
            raise ValueError("hThetaMulti requires non-empty 'theta' input.")

        if isinstance(theta, list):
            # if len(theta) == 0:
            #     raise ValueError("hThetaMulti received an empty theta list.")
            theta = torch.cat([
                item.detach().cpu() if isinstance(item, torch.Tensor) else torch.as_tensor(item)
                for item in theta
            ], dim=0).numpy()
        elif isinstance(theta, torch.Tensor):
            theta = theta.detach().cpu().numpy()
        else:
            theta = np.asarray(theta)

        if theta.ndim != 2:
            raise ValueError(f"Expected 2D theta matrix, got shape {theta.shape}.")

        return theta.astype(np.float32)
    
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
        param_range = self.max_theta - self.min_theta
        params_norm = 2 * (params - self.min_theta) / param_range - 1
        return params_norm
    
    def denormalize_parameters(self, params_norm):
        """Denormalize parameters from [-1, 1] back to original ranges."""
        params = params_norm.copy()
        param_range = self.max_theta - self.min_theta
        params = (params_norm + 1) / 2 * param_range + self.min_theta
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
        n_samples = self.s.shape[1]
        multi_channel = np.zeros((n_samples, self.num_detectors, Y_LENGTH), dtype=np.float32)
        t = np.arange(Y_LENGTH) * SAMPLING_RATE
        h_cross = np.zeros(Y_LENGTH, dtype=np.float32) # assume cross-polarization is zero for these templates
        
        print(f"Projecting signals to {self.num_detectors} detectors...")
        
        for i in range(n_samples):
            h_plus = self.s[:, i]  # Shape: (Y_LENGTH,)

            psi = float(self.polar_angle[i])
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
        clean_signal = self.multi_channel_signals[original_idx].copy()  # Shape: (num_detectors, Y_LENGTH)
        noisy_signal = clean_signal.copy()
        
        # Get parameters
        parameters = self.parameters[original_idx].copy()
        
        # Add noise to each detector channel if enabled
        # if self.noise:
        #     for j in range(self.num_detectors):
        #         # Get signal for this detector
        #         s = clean_signal[j:j+1, :]  # Shape: (1, Y_LENGTH)
                
        #         # Compute SNR (using base class method)
        #         s_normalized = s / TEN_KPC
        #         hf = np.fft.rfft(s_normalized, axis=1)[0]
        #         rho = self.calculate_snr_from_fft(hf, self.PSD)
                
        #         # Generate detector-specific noise
        #         n = self.aLIGO_noise(seed_offset=noise_realization_idx + j * 1000)
                
        #         # Add noise with target SNR
        #         s_normalized = s_normalized / 3.086e+22
        #         d_normalized = s_normalized + n * (rho / self.rho_target) * 100
        #         d = d_normalized * 3.086e+22
                
        #         # Normalize
        #         noisy_signal[j:j+1, :] = self.normalise_signals(d)
        # else:
        
        noisy_signal = self.normalise_signals(noisy_signal)
        clean_signal = self.normalise_signals(clean_signal)
        
        # Normalize parameters
        params_normalized = self.normalize_parameters(parameters.reshape(1, -1))[0]
        
        return (
            torch.tensor(clean_signal, dtype=torch.float32, device=DEVICE),
            torch.tensor(noisy_signal, dtype=torch.float32, device=DEVICE),
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
        return self.s.shape[1] * self.noise_realizations
    
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
        return (f"CCSNDataMultiChannel({self.s.shape[1]} samples × {self.num_detectors} detectors)\n"
                f"  Detectors: {', '.join(self.detectors)}\n"
                f"  Multi-channel shape: {self.multi_channel_signals.shape}\n"
                f"  Parameters: {self.param_dim}D")
