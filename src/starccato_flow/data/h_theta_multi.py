"""Multi-channel CCSN dataset for detector network analysis and sky localization.

This module only works with generated data (not raw CCSN CSV files).
"""

from typing import Optional, List, Tuple
import importlib
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d

from ..utils.defaults import DEVICE, Y_LENGTH, BATCH_SIZE, TEN_KPC, SAMPLING_RATE, GPS_TIME
from ..utils.defaults import ALIGO_ASD_FILE, AVIRGO_ASD_FILE
from ..utils.plotting_defaults import PARAMETER_LABELS, PARAMETER_RANGES

class hThetaMulti(Dataset):
    """Multi-channel CCSN dataset for sky localization with generated data only.
    
    This class handles:
    - Multiple detector channels (H1, L1, V1, etc.)
    - Sky location parameters (RA, Dec, distance)
    - Antenna pattern projections for each detector
    - Detector-specific noise
    
    Note: This class only works with generated signals (custom_data), not raw CCSN CSV files.
    """

    # Import unified parameter labels and ranges from plotting_defaults
    PARAMETER_LABELS = PARAMETER_LABELS
    PARAMETER_RANGES = PARAMETER_RANGES
    
    # Fixed physical bounds for sky parameters: [ra, dec, distance_kpc, psi]
    SKY_MIN = np.array([-np.pi, -np.pi / 2.0, 0.0, 0.0], dtype=np.float32)
    SKY_MAX = np.array([np.pi, np.pi / 2.0, 20.0, np.pi], dtype=np.float32)
    
    def __init__(
        self,
        detectors: List[str] = ['H1', 'L1', 'V1'],
        batch_size: int = BATCH_SIZE,
        detector_noise_on: bool = True,
        s: Optional[np.ndarray] = None,
        max_strain: Optional[float] = None,
        theta: Optional[np.ndarray] = None,
        min_theta: Optional[np.ndarray] = None,
        max_theta: Optional[np.ndarray] = None,
        ra: Optional[np.ndarray] = None,
        dec: Optional[np.ndarray] = None,
        d: Optional[np.ndarray] = None,
        random_polarization: bool = True,
        gps_time: float = GPS_TIME,
        seed: int = 99,
        intrinsic_param_names: Optional[List[str]] = None,
    ):
        """Initialize multi-channel CCSN dataset with generated data."""
        self.batch_size = batch_size
        self.detector_noise_on = detector_noise_on
        self.s = self._coerce_signal_matrix(s)
        self.max_strain = max_strain
        self.theta = self._coerce_theta_matrix(theta)
        self.ra = np.asarray(ra, dtype=np.float32) if ra is not None else None
        self.dec = np.asarray(dec, dtype=np.float32) if dec is not None else None
        self.d = np.asarray(d, dtype=np.float32) if d is not None else None
        self.random_polarization = random_polarization
        self.gps_time = float(gps_time)
        self.seed = seed
        self.intrinsic_param_names = intrinsic_param_names if intrinsic_param_names is not None else []
        self._current_epoch = 0
        self.include_sky_params = True
        
        # Toggle between analytical and measured PSD for noise generation
        self.use_measured_psd = True
        
        # Cache for measured sensitivity curves (loaded on first use)
        self._ligo_freq_curve = None
        self._ligo_psd_curve = None
        self._virgo_freq_curve = None
        self._virgo_psd_curve = None

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

        # Print parameter bounds
        print(f"\n{'='*70}")
        print(f"hThetaMulti Dataset - Parameter Bounds ({len(self.min_theta)} parameters)")
        print(f"{'='*70}")
        print("INTRINSIC PARAMETERS:")
        # Use provided intrinsic parameter names, or fall back to generic labels
        if self.intrinsic_param_names:
            intrinsic_labels = self.intrinsic_param_names
        else:
            intrinsic_labels = ['beta1_IC_b', 'omega_0(rad|s)', 'A(km)', 'Ye_c_b']
        
        for i in range(theta_dim):
            param_label = intrinsic_labels[i] if i < len(intrinsic_labels) else f'theta_{i}'
            print(f"  {param_label:20s}: [{self.min_theta[i]:12.6f}, {self.max_theta[i]:12.6f}]")
        
        print("\nEXTRINSIC (SKY) PARAMETERS:")
        extrinsic_labels = ['ra', 'dec', 'd', 'psi']
        for i, label in enumerate(extrinsic_labels):
            idx = theta_dim + i
            if idx < len(self.min_theta):
                print(f"  {label:20s}: [{self.min_theta[idx]:12.6f}, {self.max_theta[idx]:12.6f}]")
        print(f"{'='*70}\n")

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

        if self.use_measured_psd:
            self.AdvLIGOPSD = self.AdvLIGOPsd_measured(fourier_freq)
            self.VirgoPSD = self.VirgoPsd_measured(fourier_freq)
        else:
            self.AdvLIGOPSD = self.AdvLIGOPsd(fourier_freq)
            self.VirgoPSD = self.VirgoPsd(fourier_freq)
        
        # Project signals to multiple detectors
        self.multi_channel_signals = self._project_to_detectors()        
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

    # need to check if this Adv Virgo or normal Virgo. Code is dated from 2021, same as the AdvLIGO psd.
    def VirgoPsd(self, f):
        """Virgo power spectral density."""
        # Avoid division by zero at f=0 by clipping to a small positive value
        f = np.clip(f, 1e-10, None)
        x = f / 500
        s0 = 10.2e-46
        psd = s0*(pow(7.87*x,-4.8) + 6./17./x + 1. + x*x);
        # The upper bound is 2e10 times the minimum value
        cutoff = np.nanmin(psd) * 2e10
        psd[(psd > cutoff) | np.isnan(psd)] = cutoff
        return psd
    
    @staticmethod
    def _load_sensitivity_curve(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load sensitivity curve from .txt file with frequency and ASD columns.
        
        Args:
            filepath: Path to .txt file with two columns: frequency (Hz) and ASD (strain/√Hz)
            
        Returns:
            Tuple of (frequencies, psd_values)
        """
        if not os.path.exists(AVIRGO_ASD_FILE):
            raise FileNotFoundError(f"Sensitivity curve file not found: {AVIRGO_ASD_FILE}")
        
        data = np.loadtxt(AVIRGO_ASD_FILE)
        frequencies = data[:, 0]
        asd = data[:, 1]
        psd = asd ** 2  # Convert ASD to PSD
        
        return frequencies, psd
    
    @staticmethod
    def _interpolate_psd(freq_query: np.ndarray, freq_curve: np.ndarray, psd_curve: np.ndarray) -> np.ndarray:
        """Interpolate PSD at query frequencies using log-log interpolation.
        
        Args:
            freq_query: Frequencies at which to get PSD values
            freq_curve: Frequency points from sensitivity curve
            psd_curve: PSD values from sensitivity curve
            
        Returns:
            PSD values at query frequencies
        """
        # Use log-log interpolation for smooth behavior across frequency range
        log_freq_curve = np.log10(np.clip(freq_curve, 1e-10, None))
        log_psd_curve = np.log10(np.clip(psd_curve, 1e-50, None))
        
        # Create interpolator
        interp_func = interp1d(
            log_freq_curve, 
            log_psd_curve, 
            kind='cubic', 
            fill_value='extrapolate',
            bounds_error=False
        )
        
        # Interpolate and convert back from log space
        log_freq_query = np.log10(np.clip(freq_query, 1e-10, None))
        log_psd_query = interp_func(log_freq_query)
        psd_query = 10.0 ** log_psd_query
        
        return psd_query
    
    def AdvLIGOPsd_measured(self, f: np.ndarray, asd_file: Optional[str] = None) -> np.ndarray:
        """Get Advanced LIGO PSD from measured sensitivity curve.
        
        Args:
            f: Frequencies (Hz)
            asd_file: Path to ASD file. If None, constructs path relative to module location.
            
        Returns:
            PSD values at frequencies f
        """        
        # Load and cache on first call
        if self._ligo_freq_curve is None or self._ligo_psd_curve is None:
            self._ligo_freq_curve, self._ligo_psd_curve = self._load_sensitivity_curve(ALIGO_ASD_FILE)
        
        return self._interpolate_psd(f, self._ligo_freq_curve, self._ligo_psd_curve)
    
    def VirgoPsd_measured(self, f: np.ndarray, asd_file: Optional[str] = None) -> np.ndarray:
        """Get Virgo PSD from measured sensitivity curve.
        
        Args:
            f: Frequencies (Hz)
            asd_file: Path to ASD file. If None, constructs path relative to module location.
            
        Returns:
            PSD values at frequencies f
        """
        # Load and cache on first call
        if self._virgo_freq_curve is None or self._virgo_psd_curve is None:
            self._virgo_freq_curve, self._virgo_psd_curve = self._load_sensitivity_curve(AVIRGO_ASD_FILE)
        
        return self._interpolate_psd(f, self._virgo_freq_curve, self._virgo_psd_curve)
    
    def detector_noise(self, seed_offset=0, detector=None):
        """Add detector noise to the signal.
        
        Args:
            seed_offset (int): Offset for random seed to ensure different noise realizations
            detector (str): The detector for which to generate noise ('LIGO' or 'Virgo')

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
            pad=1,
            detector=detector,
            use_measured_psd=self.use_measured_psd
        ).reshape(1, -1)  # shape (1, Y_LENGTH)

        # Restore original random state if we changed it
        if seed_offset > 0:
            np.random.set_state(original_state)

        # The noise is now properly scaled due to correct PSD implementation
        noise = noise - noise.mean()  # Mean center as in R implementation

        return noise
    
    def rnoise(self, N: int, delta_t: float, one_sided: bool = True, pad: int = 1, detector: str = None, use_measured_psd: bool = False) -> np.ndarray:
        """Generate colored noise with given PSD.
        
        Args:
            N: Number of samples
            delta_t: Time step (seconds)
            one_sided: Whether to use one-sided or two-sided spectral density
            pad: Padding factor (>=1)
            detector: Detector name ('H1', 'L1', or 'V1')
            use_measured_psd: If True, use measured sensitivity curves; if False, use analytical formulas
            
        Returns:
            Colored noise array
        """
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

        if detector == "H1" or detector == "L1":
            psd = self.AdvLIGOPSD
        elif detector == "V1":
            psd = self.VirgoPSD
        else:
            raise ValueError("Invalid detector specified. Please choose 'H1', 'L1', or 'V1'.")

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
    
    def calculate_snr_from_fft(self, idx=0, fs=SAMPLING_RATE, N=Y_LENGTH):
        """Calculate SNR directly from pre-computed FFT for a sample index.
        
        Args:
            idx (int): Index of the sample to calculate SNR for
            fs (float): Sampling frequency
            N (int): FFT length
            
        Returns:
            list: SNR values for each detector
        """
        df = fs / N
        snr = []

        # Get noisy signal tensor and move to CPU for numpy operations
        _, noisy_signal_tensor, _ = self.__getitem__(idx)
        noisy_signal = noisy_signal_tensor.cpu().numpy()  # Convert to numpy

        for j, detector in enumerate(self.detectors):
            if detector == "H1" or detector == "L1":
                psd = self.AdvLIGOPSD
            elif detector == "V1":
                psd = self.VirgoPSD
            else:
                raise ValueError("Invalid detector specified. Please choose 'LIGO' or 'Virgo'.")
            
            # Get signal for this detector
            signal = noisy_signal[j, :]  # Shape: (Y_LENGTH,)
            signal *= self.max_strain / TEN_KPC
            
            # Compute FFT
            hf = np.fft.rfft(signal, axis=0)[1]
            
            # Calculate SNR
            integrand = (np.abs(hf)**2) / psd
            rho2 = 4 * np.sum(integrand) * df
            snr.append(np.sqrt(rho2))

        return snr

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
                # h_signal = F_plus * h_plus_shifted + F_cross * h_cross_shifted
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
        # Get multi-channel signal (already projected)
        # Copy so normalization/noise operations do not mutate cached dataset arrays.
        clean_signal = self.multi_channel_signals[idx].copy()  # Shape: (num_detectors, Y_LENGTH)
        noisy_signal = clean_signal.copy()
        
        # Get parameters
        parameters = self.parameters[idx].copy()
        
        # Add noise to each detector channel if enabled
        if self.detector_noise_on:
            for j in range(self.num_detectors):
                # Get signal for this detector
                s = clean_signal[j:j+1, :].flatten()  # Shape: (1, Y_LENGTH)
                
                # Compute SNR (using base class method)
                s_normalized = s / TEN_KPC
                # hf = np.fft.rfft(s_normalized, axis=1)[0]
                # rho = self.calculate_snr_from_fft(hf)
                
                # Generate detector-specific noise
                n = self.detector_noise(seed_offset=j * 1000, detector=self.detectors[j]).flatten()  # Shape: (Y_LENGTH,)
                
                # Add noise with target SNR
                # d_normalized = s_normalized + n * (self.d[original_idx] / 10) * 100
                d_normalized = s_normalized + n * 100
                d = d_normalized * TEN_KPC
                
                noisy_signal[j:j+1, :] = d
        
        noisy_signal = self.normalise_signals(noisy_signal)
        clean_signal = self.normalise_signals(clean_signal)
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
        signal = self.multi_channel_signals[idx, det_idx, :]
        signal_normalized = self.normalise_signals(signal.reshape(1, -1))
        
        return torch.tensor(signal_normalized, dtype=torch.float32, device=DEVICE).squeeze()
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return self.s.shape[1]
    
    @property
    def current_epoch(self) -> int:
        """Get the current epoch number."""
        return self._current_epoch
    
    def __repr__(self) -> str:
        return (f"CCSNDataMultiChannel({self.s.shape[1]} samples × {self.num_detectors} detectors)\n"
                f"  Detectors: {', '.join(self.detectors)}\n"
                f"  Multi-channel shape: {self.multi_channel_signals.shape}\n"
                f"  Parameters: {self.param_dim}D")
