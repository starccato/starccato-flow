import os
import time
import csv
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import mean, nn
from torch.utils.data import DataLoader

from ..data.s_theta import sTheta
from ..data.h_theta_multi import hThetaMulti
from ..supernovae.supernovae import Supernovae
from tqdm.auto import trange
  
from ..plotting.sky import plot_galactic_supernovae_polar_hemispheres
from ..plotting.signals import plot_detector_signal_channels, plot_candidate_signal
from ..plotting.parameters import plot_eos_ye_posterior_distribution, plot_eos_ye_distribution, plot_epoch_sky_parameters, plot_corner, plot_pp_coverage
from ..plotting.losses import plot_loss

from ..utils.defaults import Y_LENGTH, HIDDEN_DIM, Z_DIM, BATCH_SIZE, DEVICE, TEN_KPC, VALIDATION_SPLIT, MAX_DISTANCE_KPC, SAMPLING_FREQ
from ..utils.plotting_defaults import PARAMETER_LABELS 
from ..nn.flow_multi import FlowFCL, FlowCNN

from . import create_train_val_split

def _set_seed(seed: int):
    """Set the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return seed

class FlowMatchingTrainerMulti:
    def __init__(
        self,
        y_length: int = Y_LENGTH,
        hidden_dim: int = HIDDEN_DIM,
        z_dim: int = Z_DIM,
        seed: int = 99,
        batch_size: int = BATCH_SIZE,
        num_epochs: int = 256,
        samples_per_epoch: int = 18000,
        validation_split: float = VALIDATION_SPLIT,
        lr_flow: float = 5e-4,
        checkpoint_interval: int = 16,
        outdir: str = "outdir",
        detector_noise_on: bool = True,
        toy: bool = False,
        max_grad_norm: float = 1.0,
        parameters: list = None,
        custom_data: tuple = None,  # (signals_array, params_array) for generated data
        train_data_path: str = None,  # Path to training data files (generated signals)
        val_data_path: str = None,  # Path to validation data files (real CVAE val set)
        use_physics_aware_norm: bool = True,  # Use parameter-specific normalization (Gabbard-inspired)
    ):
        """Initialize FlowMatchingTrainerMulti.
        
        Args:
            parameters: List of parameter names to estimate. Examples:
                ["beta1_IC_b", "ra", "dec", "d", "psi"] - estimate sky params + beta
                ["ra", "dec", "d", "psi"] - estimate only sky parameters
                If None, defaults to ["beta1_IC_b", "ra", "dec", "d", "psi"]
            custom_data: Optional tuple of (signals, parameters) arrays for using generated data.
                signals: shape (signal_length, num_samples)
                parameters: shape (num_samples, num_params)
                If provided, overrides toy/CCSN data loading and does train/val split.
            train_data_path: Path prefix for training data (e.g., 'outdir/generated').
                Will load {train_data_path}_signals.npy and {train_data_path}_parameters.npy
            val_data_path: Path prefix for validation data (e.g., 'outdir/cvae_val').
                Will load {val_data_path}_signals.npy and {val_data_path}_parameters.npy
            use_physics_aware_norm: If True, use parameter-specific normalization (Gabbard-inspired):
                - Cyclic parameters (RA, Dec, Psi): represented as (cos, sin) on 2D plane
                - Distance: log-space [0, 1] to ensure positive values
                - Intrinsic params: linear [-1, 1]
                If False, use original linear [-1, 1] normalization for all parameters.
                
        Note: If both train_data_path and val_data_path are provided, they take precedence
        over custom_data.
        """
        self.y_length = y_length
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.use_physics_aware_norm = use_physics_aware_norm
        self.seed = seed
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.samples_per_epoch = samples_per_epoch
        self.validation_split = validation_split
        self.lr_flow = lr_flow
        self.checkpoint_interval = checkpoint_interval
        
        # Construct absolute outdir path if not provided
        if outdir == "outdir" or (outdir and not os.path.isabs(outdir)):
            _module_dir = os.path.dirname(os.path.abspath(__file__))  # /src/starccato_flow/training/
            _starccato_flow_root = os.path.dirname(os.path.dirname(os.path.dirname(_module_dir)))  # /starccato-flow/
            outdir = os.path.join(_starccato_flow_root, "outdir")
        
        self.outdir = outdir
        self.toy = toy
        self.detector_noise_on = detector_noise_on
        self.max_grad_norm = max_grad_norm
        
        # Set default parameters if not provided
        self.parameters_to_estimate = parameters
        
        # Parameter mapping: parameter names -> (full name, index in hThetaMulti output)
        # hThetaMulti always produces: [beta1_IC_b, omega_0(rad|s), A(km), Ye_c_b, ra, dec, d, psi]
        self.parameter_mapping = {
            "beta1_IC_b": ("beta1_IC_b", 0),
            "beta_ic_b": ("beta1_IC_b", 0),  # Alias for backward compatibility
            "omega_0": ("omega_0(rad|s)", 1),
            "omega_0(rad|s)": ("omega_0(rad|s)", 1),  # Support full name too
            "A": ("A(km)", 2),
            "A(km)": ("A(km)", 2),  # Support full name
            "Ye_c_b": ("Ye_c_b", 3),
            "ra": ("ra", 4),
            "dec": ("dec", 5),
            "d": ("d", 6),
            "psi": ("psi", 7),
        }
        
        # Categorize parameters first (needed for extraction index calculation)
        intrinsic_param_names = {"beta1_IC_b", "omega_0", "A", "Ye_c_b"}
        self.intrinsic_params = [p for p in parameters if p in intrinsic_param_names]
        self.sky_params = [p for p in parameters if p not in intrinsic_param_names]
        self.sky_param_dim = len(self.sky_params)
        
        # Build extraction indices based on actual dataset structure
        # hThetaMulti concatenates: [intrinsic_params] + [sky_params]
        # Since we request specific intrinsic + sky params, hThetaMulti only outputs those in order
        # So extract_indices should just be sequential: [0, 1, 2, ..., n_total_params-1]
        n_total_params = len(self.intrinsic_params) + len(self.sky_params)
        self.param_extract_indices = list(range(n_total_params))
        
        print(f"\n=== Parameter Extraction Setup ===")
        print(f"Requested parameters: {parameters}")
        print(f"Intrinsic params: {self.intrinsic_params}")
        print(f"Sky params: {self.sky_params}")
        print(f"Extract indices (sequential from hThetaMulti): {self.param_extract_indices}")
        print(f"Final flow parameter dimension: {len(self.param_extract_indices)}")
        print(f"{'='*40}\n")

        # Construct absolute path to supernovae data file
        # __file__ is at src/starccato_flow/training/trainer_flow_matching_multi.py
        # We need to go up to starccato-flow root, then to data/supernovae/
        trainer_dir = os.path.dirname(os.path.abspath(__file__))
        starccato_flow_root = os.path.dirname(os.path.dirname(os.path.dirname(trainer_dir)))
        supernovae_file = os.path.join(starccato_flow_root, "..", "data", "supernovae", "exploded_supernovae_t100_sf5.csv")
        
        self.supernovae = Supernovae(
            locations_file=supernovae_file,
            rotation_offset=np.deg2rad(0.0),
        )

        # Load data from files if paths are provided
        if train_data_path is not None and val_data_path is not None:            
            print(f"\n=== Loading Data from Files ===")
            
            # Load training data (generated signals)
            train_signals = np.load(f"{train_data_path}_signals.npy")
            train_params = np.load(f"{train_data_path}_parameters.npy")
            print(f"Loaded training data: {train_signals.shape[1]} generated signals")
            
            # Load validation data (real CVAE validation set)
            val_signals = np.load(f"{val_data_path}_signals.npy")
            val_params = np.load(f"{val_data_path}_parameters.npy")
            print(f"Loaded validation data: {val_signals.shape[1]} real signals (CVAE held-out)")
            
            # Create training dataset
            # Note: sTheta should always have detector_noise_on=False (noise added only in hThetaMulti)
            self.training_dataset = sTheta(
                custom_data=(train_signals, train_params),
                detector_noise_on=False,
                num_epochs=num_epochs,
                batch_size=batch_size,
                parameters=parameters,
                intrinsic_param_names=self.intrinsic_params
            )
            
            # Create validation dataset sharing normalization from training
            # Note: sTheta should always have detector_noise_on=False (noise added only in hThetaMulti)
            self.validation_dataset = sTheta(
                custom_data=(val_signals, val_params),
                detector_noise_on=False,
                num_epochs=num_epochs,
                batch_size=batch_size,
                parameters=parameters,
                shared_min=self.training_dataset.shared_min_theta,
                shared_max=self.training_dataset.shared_max_theta,
                shared_max_strain=self.training_dataset.shared_max_strain,
            )
            
        elif custom_data is not None:
            # Use custom generated data from CVAE
            
            signals_array, params_array = custom_data
            num_samples = params_array.shape[0]
            
            # Split indices for train/val
            base_indices = list(range(num_samples))
            split = int(np.floor(validation_split * num_samples))
            
            rng = np.random.RandomState(seed)
            rng.shuffle(base_indices)
            train_indices = base_indices[split:]
            val_indices = base_indices[:split]
            
            print(f"\n=== Custom Data Split ===")
            print(f"Total samples: {num_samples}")
            print(f"Training samples: {len(train_indices)}")
            print(f"Validation samples: {len(val_indices)}")
            
            # Create training dataset with custom data
            # Note: sTheta should always have detector_noise_on=False (noise added only in hThetaMulti)
            self.training_dataset = sTheta(
                custom_data=(signals_array[:, train_indices], params_array[train_indices]),
                detector_noise_on=False,
                num_epochs=num_epochs,
                batch_size=batch_size,
                parameters=parameters,
            )
            
            # Create validation dataset with custom data
            # Note: sTheta should always have detector_noise_on=False (noise added only in hThetaMulti)
            self.validation_dataset = sTheta(
                custom_data=(signals_array[:, val_indices], params_array[val_indices]),
                detector_noise_on=False,
                num_epochs=num_epochs,
                batch_size=batch_size,
                parameters=parameters,
                shared_min=self.training_dataset.shared_min_theta,
                shared_max=self.training_dataset.shared_max_theta,
                shared_max_strain=self.training_dataset.shared_max_strain
            )
        else:
            # Use standard train/val split. Probably the most acceptable
            self.training_dataset, self.validation_dataset, self.val_indices = create_train_val_split(
                toy=self.toy,
                y_length=self.y_length,
                detector_noise_on=self.detector_noise_on,
                validation_split=self.validation_split,
                seed=self.seed,
                num_epochs=self.num_epochs,
                parameters=parameters,
            )

        # Create DataLoaders (datasets already have disjoint base signals via indices parameter)
        self.train_loader = DataLoader(
            self.training_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.validation_dataset, 
            batch_size=self.batch_size, 
            shuffle=False  # Don't shuffle validation for consistency
        )

        print(f"\n=== Dataset Sizes ===")
        print(f"Training samples: {len(self.training_dataset)}")
        print(f"Validation samples: {len(self.validation_dataset)}")
        print("=" * 50)

        # Print combined parameter bounds (intrinsic + sky)
        print(f"\n======================================================================")
        print(f"Combined Dataset - Parameter Bounds ({len(self.training_dataset.shared_min_theta)} parameters)")
        print(f"======================================================================")
        
        # Get parameter names
        param_names = self.intrinsic_params + ["ra", "dec", "d", "psi"]
        
        # Print bounds for each parameter
        for i, (name, min_val, max_val) in enumerate(zip(param_names, self.training_dataset.shared_min_theta, self.training_dataset.shared_max_theta)):
            # Format based on parameter type
            if name in ["ra", "dec", "psi"]:
                # Angular parameters - show in radians and degrees
                print(f"{name:20s}: [{min_val:12.6f}, {max_val:12.6f}] rad = [{np.degrees(min_val):8.2f}°, {np.degrees(max_val):8.2f}°]")
            elif name == "d":
                # Distance in kpc
                print(f"{name:20s}: [{min_val:12.6f}, {max_val:12.6f}] kpc")
            else:
                # Other parameters
                print(f"{name:20s}: [{min_val:12.6f}, {max_val:12.6f}]")
        print(f"======================================================================\n")

        if len(self.training_dataset) == 0:
            raise ValueError(
                "Training dataset is empty after filtering/splitting. "
                "Beta filtering is active; check data availability and split settings."
            )
        if len(self.validation_dataset) == 0:
            raise ValueError(
                "Validation dataset is empty after filtering/splitting. "
                "Increase dataset size or reduce validation_split."
            )

        self.checkpoint_interval = checkpoint_interval

        os.makedirs(outdir, exist_ok=True)
        _set_seed(self.seed)

        # setup Flow Matching model
        # Flow parameter dimension = number of parameters to estimate
        # With physics-aware normalization, cyclic parameters (RA, Dec, Psi) become 2D (cos, sin),
        # so the effective dimension is higher than the number of estimated parameters
        if self.use_physics_aware_norm:
            self.flow_param_dim = self._calculate_physics_aware_param_dim(self.parameters_to_estimate)
        else:
            self.flow_param_dim = len(self.parameters_to_estimate)
        
        self.flow_signal_dim = Y_LENGTH * 3
        self.flow = FlowFCL(dim=self.flow_param_dim, signal_dim=self.flow_signal_dim).to(DEVICE)
        # self.flow = FlowCNN(dim=self.flow_param_dim, signal_dim=self.flow_signal_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.lr_flow, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()
        
        # Initialize loss tracking lists (populated during training or loading)
        self.avg_mse_losses = []
        self.avg_mse_losses_val = []

    @staticmethod
    def _denormalize_with_bounds(params_norm: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
        """Denormalize parameters from [-1, 1] using explicit min/max bounds."""
        return (params_norm + 1.0) / 2.0 * (max_vals - min_vals) + min_vals
    
    @staticmethod
    def _calculate_physics_aware_param_dim(parameters_to_estimate: list) -> int:
        """Calculate effective dimension for physics-aware normalization.
        
        Cyclic parameters (ra, dec, psi) become 2D (cos, sin) in physics-aware space,
        intrinsic and distance remain 1D. So:
        - Each of [ra, dec, psi]: +1 (was 1 value, now 2 values in normalized space)
        - Each intrinsic or distance param: +0 (still 1 value)
        
        Args:
            parameters_to_estimate: List of parameter names
            
        Returns:
            Effective dimension in physics-aware normalized space
        """
        cyclic_params = {"ra", "dec", "psi"}
        base_dim = len(parameters_to_estimate)
        # Each cyclic parameter becomes 2D (cos, sin), adding 1 extra dimension per cyclic param
        extra_dims = sum(1 for p in parameters_to_estimate if p in cyclic_params)
        return base_dim + extra_dims
    
    def _denormalize_extracted_params(self, params_norm: np.ndarray, dataset) -> np.ndarray:
        """Denormalize extracted parameters using appropriate bounds.
        
        The input params_norm is in the reduced parameter space (e.g., 4D if we extracted 4 sky params).
        We need to extract bounds for each parameter IN THE ORDER THEY WERE REQUESTED.
        
        Note: When denormalizing from hThetaMulti datasets created with intrinsic_param_names,
        the dataset only contains the requested parameters, so we use sequential local indices.
        """
        # Build min/max bounds by looking up each requested parameter
        # This ensures correct ordering even if parameters are requested out of dataset order
        min_vals = []
        max_vals = []
        
        # When the dataset was created with specific intrinsic_param_names (e.g., hThetaMulti_val),
        # it only contains the requested parameters. Use sequential local indices.
        # Otherwise, use the global parameter_mapping indices.
        num_params_in_dataset = len(dataset.shared_min_theta)
        num_requested_params = len(self.parameters_to_estimate)
        
        if num_params_in_dataset == num_requested_params:
            # Dataset contains only the requested parameters (local indices)
            for local_idx in range(num_requested_params):
                min_vals.append(dataset.shared_min_theta[local_idx])
                max_vals.append(dataset.shared_max_theta[local_idx])
        else:
            # Dataset contains all parameters (use global indices from parameter_mapping)
            for param_name in self.parameters_to_estimate:
                if param_name in self.parameter_mapping:
                    _, dataset_idx = self.parameter_mapping[param_name]
                    min_vals.append(dataset.shared_min_theta[dataset_idx])
                    max_vals.append(dataset.shared_max_theta[dataset_idx])
                else:
                    raise ValueError(f"Parameter '{param_name}' not found in parameter_mapping")
        
        min_vals = np.array(min_vals, dtype=np.float32)
        max_vals = np.array(max_vals, dtype=np.float32)
        
        # Denormalize using those bounds
        return self._denormalize_with_bounds(params_norm, min_vals, max_vals)
    
    def _get_extracted_index(self, param_name: str) -> int:
        """Get the index of a parameter in the extracted parameter space.
        
        Args:
            param_name: Parameter name (e.g., "ra", "dec", "beta_ic_b")
            
        Returns:
            Index in the extracted parameter space, or -1 if not found
        """
        if param_name not in self.parameters_to_estimate:
            return -1
        return self.parameters_to_estimate.index(param_name)

    def _save_epoch_data_plots(self, epoch: int) -> None:
        """Save parameter distribution plots for the current epoch in a grid layout."""
        multi_dataset = getattr(self, "h_theta_multi_train", None)
        if multi_dataset is None:
            multi_dataset = getattr(self, "h_theta_multi", None)
        if multi_dataset is None:
            return

        epoch_dir = os.path.join(self.outdir, "flow_matching", "epoch_data")
        os.makedirs(epoch_dir, exist_ok=True)

        # Plot training set parameters
        if multi_dataset is not None:
            fname_train = os.path.join(epoch_dir, f"epoch_{epoch + 1:04d}_train_params.png")
            plot_epoch_sky_parameters(
                dataset=multi_dataset,
                sky_params=self.sky_params,
                fname=fname_train,
                background="black",
                color="#3498db",
                bins=40
            )

        # Plot validation set parameters
        val_multi_dataset = getattr(self, "h_theta_multi_val", None)
        if val_multi_dataset is not None:
            fname_val = os.path.join(epoch_dir, f"epoch_{epoch + 1:04d}_val_params.png")
            plot_epoch_sky_parameters(
                dataset=val_multi_dataset,
                sky_params=self.sky_params,
                fname=fname_val,
                background="black",
                color="#e74c3c",
                bins=40
            )

    def _sample_dataset_batches(self, dataset, n_samples: int):
        """Sample n_samples from a base dataset and return batched signal/parameter lists."""
        signals = []
        params = []
        base_size = int(dataset.parameters.shape[0])
        remaining = n_samples
        while remaining > 0:
            current_batch_size = min(self.batch_size, remaining)
            batch_indices = np.random.choice(
                base_size,
                current_batch_size,
                replace=base_size < current_batch_size,
            )

            # Pull raw (unnormalized) arrays directly so hThetaMulti performs
            # the only normalization step for combined theta+sky parameters.
            batch_signals = torch.tensor(
                dataset.signals[:, batch_indices].T,
                dtype=torch.float32,
            )
            batch_params = torch.tensor(
                dataset.parameters[batch_indices],
                dtype=torch.float32,
            )

            signals.append(batch_signals)
            params.append(batch_params)
            remaining -= current_batch_size
        return signals, params

    def run_parameter_estimation(self, signal_idx: int = None, d: float = None, ra: float = None, dec: float = None, epoch: int = None, export_on: bool = False, random_psi: bool = True, font_family: str = "Sans-serif", font_name: str = "Avenir", fname_signal: str = None, fname_posterior: str = None, fname_posterior_sky: str = None, fname_posterior_galactic: str = None, fname_eos_ye: str = None, background: str = "white", transparent: bool = False) -> None:
        """Run parameter estimation on a single signal and return the predicted parameters.
        
        Args:
            signal_idx: Index of the signal in the validation dataset
            d: Distance in kpc
            ra: Right ascension in radians (optional, random if None)
            dec: Declination in radians (optional, random if None)
            epoch: Epoch number (used for filenames when signal_idx is None)
            export_on: Whether to export signal channels as .txt files
            random_psi: Whether to use random polarization angle (True) or fixed psi=0 (False)
            font_family: Font family for plots
            font_name: Font name for plots
            fname_signal: Filename for the signal plot
            fname_posterior: Filename for the posterior plot
            fname_posterior_sky: Filename for the posterior sky plot
            fname_posterior_galactic: Filename for the posterior galactic plot
            fname_eos_ye: Filename for the EOS/Ye plot
            background: Background color for plots (e.g., "white", "black")
            transparent: Whether to save plots with transparent background
        """
        
        # Set up directory paths
        epoch_data_dir = os.path.join(self.outdir, "flow_matching", "epoch_data")
        os.makedirs(epoch_data_dir, exist_ok=True)
        
        # Create filename suffix based on provided parameters
        if signal_idx is not None and d is not None:
            if ra is not None and dec is not None:
                filename_suffix = f"signal_{signal_idx:04d}_ra_{np.degrees(ra):.1f}_dec_{np.degrees(dec):.1f}_d_{d:.1f}"
            else:
                filename_suffix = f"signal_{signal_idx:04d}_d_{d:.1f}"
        else:
            filename_suffix = f"epoch_{epoch+1:04d}" if epoch is not None else "epoch"
        
        if signal_idx is not None and d is not None:
            # Get RAW (unnormalized) signal and params from validation dataset
            # We need raw signals because hThetaMulti expects unnormalized input
            signal_raw = self.validation_dataset.signals[:, signal_idx:signal_idx+1]  # Raw signal, shape (Y_LENGTH, 1)
            params = self.validation_dataset.parameters[signal_idx]  # Raw params, shape (num_params,)
            EOS = self.validation_dataset.eos.iloc[signal_idx]

            # Use specified sky parameters or randomly select a supernova at the specified distance
            if ra is not None and dec is not None:
                # Use directly specified sky parameters
                sampled_ra = np.array([ra])
                sampled_dec = np.array([dec])
                sampled_d = np.array([d])
            else:
                # Find supernovae at the specified distance and randomly select one
                distance_mask = (
                    (self.supernovae.distances >= d - 0.25)
                    & (self.supernovae.distances <= d + 0.25)
                )
                candidate_indices = np.where(distance_mask)[0]

                # Randomly select one supernova at that distance
                candidate_index = np.random.choice(candidate_indices)
                sampled_ra = np.array([self.supernovae.ra[candidate_index]])
                sampled_dec = np.array([self.supernovae.dec[candidate_index]])
                sampled_d = np.array([self.supernovae.distances[candidate_index]])
            
            # Wrap raw signal and params in tensors for hThetaMulti
            signals = [torch.tensor(signal_raw, dtype=torch.float32)]
            params_np = np.asarray(params)
            if params_np.ndim == 1:
                params_np = params_np.reshape(1, -1)
            params_tensor = torch.tensor(params_np, dtype=torch.float32)
            
            # Create temporary multi-channel dataset with the specified sky parameters
            # Following the same pattern as train()
            temp_h_theta_multi_val = hThetaMulti(
                s=signals,  # List of tensors with RAW signals
                shared_max_strain=self.validation_dataset.shared_max_strain,
                theta=params_tensor,  # Tensor
                shared_min=self.validation_dataset.shared_min_theta,
                shared_max=self.validation_dataset.shared_max_theta,
                ra=sampled_ra,
                dec=sampled_dec,
                d=sampled_d,
                batch_size=self.batch_size,
                detector_noise_on=True,  # Add fresh detector noise, consistent with train()
                random_polarization=random_psi,
                seed=1,
                intrinsic_param_names=self.intrinsic_params,
                use_physics_aware_norm=self.use_physics_aware_norm
            )
            case = temp_h_theta_multi_val[0]
            # Use temp_h_theta_multi_val for plotting since it was used to create case
            active_h_theta_multi = temp_h_theta_multi_val
        else:
            random_signal_idx = np.random.randint(len(self.validation_dataset))
            case = self.h_theta_multi_val[random_signal_idx]
            active_h_theta_multi = self.h_theta_multi_val
        
        plot_detector_signal_channels(
            signals=case[0].detach().cpu().numpy() / TEN_KPC,
            noisy_signals=case[1].detach().cpu().numpy() / TEN_KPC,
            max_value=active_h_theta_multi.shared_max_strain,
            detector_labels=active_h_theta_multi.detectors,
            background="black",
            generated=False,
            fname=os.path.join(epoch_data_dir, f"{filename_suffix}_signal.png") if fname_signal is None else fname_signal,
            font_family=font_family,
            font_name=font_name,
            transparent=transparent,
            figsize_mm=(165, 190),
            fontsize_tick=12,
            fontsize_text=18,
            line_weight=1.4
        )
        # Generate posterior samples once and reuse for both plots
        posterior_samples_denorm, true_param_denorm = self._generate_posterior_samples(
            case, active_h_theta_multi, num_samples=3000, n_steps=20
        )
        
        # Clip posterior samples to valid parameter ranges
        # Use dataset bounds if available, otherwise skip clipping
        if (hasattr(self.training_dataset, 'shared_min_theta') and 
            hasattr(self.training_dataset, 'shared_max_theta') and
            len(self.training_dataset.shared_min_theta) > 0 and
            len(self.training_dataset.shared_max_theta) > 0):
            posterior_samples_denorm = np.clip(
                posterior_samples_denorm,
                self.training_dataset.shared_min_theta,
                self.training_dataset.shared_max_theta
            )
        
        self.plot_corner_sampled_signal(
            fname=os.path.join(epoch_data_dir, f"{filename_suffix}_corner.png") if fname_posterior is None else fname_posterior,
            posterior_samples_denorm=posterior_samples_denorm,
            true_param_denorm=true_param_denorm,
            background="black",
            font_family=font_family,
            font_name=font_name,
            transparent=transparent
        )
        # Debug: print true parameters being plotted
        print(f"\nTrue parameters to plot:")
        for i, param in enumerate(self.parameters_to_estimate):
            print(f"  {param}: {true_param_denorm[i]:.6f}")
            if param == 'ra':
                print(f"    RA in degrees: {np.rad2deg(true_param_denorm[i]):.2f}°")
            elif param == 'dec':
                print(f"    Dec in degrees: {np.rad2deg(true_param_denorm[i]):.2f}°")
        
        self.plot_sky_localisation_sampled_signal(
            fname=os.path.join(epoch_data_dir, f"{filename_suffix}_sky.png") if fname_posterior_sky is None else fname_posterior_sky,
            posterior_samples_denorm=posterior_samples_denorm,
            true_param_denorm=true_param_denorm,
            font_family=font_family,
            font_name=font_name,
            transparent=True
        )
        self.plot_galactic_distribution_with_posterior(
            fname=os.path.join(epoch_data_dir, f"{filename_suffix}_galactic.png") if fname_posterior_galactic is None else fname_posterior_galactic,
            posterior_samples_denorm=posterior_samples_denorm,
            true_param_denorm=true_param_denorm,
            font_family=font_family,
            font_name=font_name,
            transparent=transparent,
            background=background
        )
        # Plot zoomed version (10 kpc around sun) - transparent
        self.plot_galactic_distribution_with_posterior_zoom(
            fname=os.path.join(epoch_data_dir, f"{filename_suffix}_galactic_zoom.png") if fname_posterior_galactic is None else fname_posterior_galactic.replace(".svg", "_zoom.svg"),
            posterior_samples_denorm=posterior_samples_denorm,
            true_param_denorm=true_param_denorm,
            font_family=font_family,
            font_name=font_name,
            transparent=transparent,
            background=background
        )
        # Plot zoomed version (10 kpc around sun) - navy background
        self.plot_galactic_distribution_with_posterior_zoom(
            fname=os.path.join(epoch_data_dir, f"{filename_suffix}_galactic_zoom.png") if fname_posterior_galactic is None else fname_posterior_galactic.replace(".svg", "_zoom_navy.svg"),
            posterior_samples_denorm=posterior_samples_denorm,
            true_param_denorm=true_param_denorm,
            font_family=font_family,
            font_name=font_name,
            transparent=False,
            background="#00001e"
        )
        # export each channel of the signal as a separate .txt file for external analysis
        if export_on:
            export_dir = os.path.join(self.outdir, "exported_signals")
            os.makedirs(export_dir, exist_ok=True)
            detector_labels = active_h_theta_multi.detectors
            
            # Generate 4 seconds of zero signal at 4096 Hz = 16384 samples
            zero_duration_samples = int(4 * SAMPLING_FREQ)  # 4 seconds at 4096 Hz
            signal_length = case[0].shape[1]  # Y_LENGTH
            
            # Calculate middle position to embed the signal
            start_idx = (zero_duration_samples - signal_length) // 2
            end_idx = start_idx + signal_length
            
            # Collect combined signals for plotting
            combined_signals_for_plot = []
            
            for i in range(case[0].shape[0]):
                # Get clean signal and denormalize to physical units
                channel_signal = case[0][i].detach().cpu().numpy() / TEN_KPC * active_h_theta_multi.shared_max_strain
                
                # Create a zero array (silence)
                detector_name = detector_labels[i] if i < len(detector_labels) else f"channel_{i+1}"
                zero_signal = np.zeros(zero_duration_samples, dtype=np.float32)
                
                # Embed the signal in the middle of the zeros
                combined_signal = zero_signal.copy()
                combined_signal[start_idx:end_idx] = channel_signal
                
                combined_signals_for_plot.append(combined_signal)
                
                # Export combined signal (signal embedded in 4 seconds of zeros)
                np.savetxt(
                    os.path.join(export_dir, f"{filename_suffix}_{detector_name}.txt"),
                    combined_signal
                )
            
            print(f"✓ Exported signals to {export_dir}/")
            print(f"  Signal embedded in middle of 4-second zero (silence) window (samples {start_idx}-{end_idx})")
            print(f"  Signal time window: {start_idx / SAMPLING_FREQ:.3f}s - {end_idx / SAMPLING_FREQ:.3f}s")

        if "Ye_c_b" in self.parameters_to_estimate:
            # get true ye and corresponding eos values for the signal (using random_signal_idx if signal_idx is None)
            samples_ye = posterior_samples_denorm[:, self._get_extracted_index("Ye_c_b")]
            true_ye = true_param_denorm[self._get_extracted_index("Ye_c_b")]
            true_eos = self.validation_dataset.eos.iloc[signal_idx] if signal_idx is not None else self.validation_dataset.eos.iloc[random_signal_idx]

            # extract all ye and corresponding eos values from the training and validation datasets for comparison
            dataset_ye = [self.training_dataset.parameters[:, self._get_extracted_index("Ye_c_b")], self.validation_dataset.parameters[:, self._get_extracted_index("Ye_c_b")]]
            dataset_eos = [self.training_dataset.eos.values, self.validation_dataset.eos.values]

            plot_eos_ye_posterior_distribution(
                samples_ye=samples_ye,
                true_ye=true_ye,
                true_eos=str(true_eos),
                dataset_ye=dataset_ye,
                dataset_eos=dataset_eos,
                background=background,
                font_family=font_family,
                font_name=font_name,
                fname=os.path.join(epoch_data_dir, f"{filename_suffix}_eos_ye.png") if fname_eos_ye is None else fname_eos_ye,
            ) 

    def train(self):
        t0 = time.time()

        self.avg_mse_losses = []
        self.avg_mse_losses_val = []
        self.flow_gradient_norms = []  # Track gradient norms for visualization

        epoch_bar = trange(self.num_epochs, desc="Epochs", position=0, leave=True)
        for epoch in epoch_bar:
            self.flow.train()
            total_loss = 0
            total_samples = 0
            use_cuda = str(DEVICE).startswith("cuda")
            loader_kwargs = {
                "batch_size": self.batch_size,
                "num_workers": 0,
                "pin_memory": use_cuda,
                "persistent_workers": False,
            }

            sampled_ra, sampled_dec, sampled_d = self.supernovae.sample_supernovae_for_epoch(
                epoch,
                self.samples_per_epoch,
                self.num_epochs,
                exponential=True,
                epoch_dir=os.path.join(self.outdir, "flow_matching", "epoch_data"),
            )
            signals, params = self._sample_dataset_batches(self.training_dataset, self.samples_per_epoch)


            # create multi-channel signals
            self.h_theta_multi_train = hThetaMulti(
                s=signals,
                shared_max_strain=self.training_dataset.shared_max_strain,
                theta=params,
                shared_min=self.training_dataset.shared_min_theta,
                shared_max=self.training_dataset.shared_max_theta,
                ra=sampled_ra,
                dec=sampled_dec,
                d=sampled_d,
                batch_size=self.batch_size,
                detector_noise_on=True,
                random_polarization=True,
                seed=epoch,  # Vary seed by epoch for different psi values each epoch
                intrinsic_param_names=self.intrinsic_params,
                use_physics_aware_norm=self.use_physics_aware_norm
            )
            self.h_theta_multi_train_loader = DataLoader(
                self.h_theta_multi_train,
                shuffle=True,
                **loader_kwargs,
            )

            self._save_epoch_data_plots(epoch)

            for signal, noisy_signal, params in self.h_theta_multi_train_loader:
                signal = signal.view(signal.size(0), -1).to(DEVICE, non_blocking=use_cuda)
                noisy_signal = noisy_signal.view(noisy_signal.size(0), -1).to(DEVICE, non_blocking=use_cuda)
                params = params.view(params.size(0), -1).to(DEVICE, non_blocking=use_cuda)
                params_target = self._extract_params_to_estimate(params)

                if params_target.size(-1) != self.flow_param_dim:
                    raise ValueError(
                        f"Parameter dimension mismatch: expected {self.flow_param_dim}, got {params_target.size(-1)}"
                    )

                x_0 = torch.randn_like(params_target)  # noise in parameter space
                t = torch.rand(len(params_target), 1, device=DEVICE)  # random time values on correct device
                x_t = (1 - t) * x_0 + t * params_target  # interpolated parameters
                dx_t = params_target - x_0  # true velocity direction in parameter space

                self.optimizer.zero_grad()
                loss = self.loss_fn(self.flow(x_t, t, noisy_signal), dx_t)
                loss.backward()  # predict parameter velocity given signal
                
                # Clip gradients to prevent explosion
                grad_norm = torch.nn.utils.clip_grad_norm_(self.flow.parameters(), self.max_grad_norm)
                self.flow_gradient_norms.append(float(grad_norm))
                
                self.optimizer.step()

                total_loss += loss.item()
                total_samples += signal.size(0)

            avg_total_loss = total_loss / total_samples
            self.avg_mse_losses.append(avg_total_loss)

            # Validation
            self.flow.eval()
            val_total_loss = 0
            val_samples = 0
            with torch.no_grad():
                n_val_signals = 2000
                val_sampled_ra, val_sampled_dec, val_sampled_d = self.supernovae.sample_supernovae_for_epoch(
                    epoch,
                    n_val_signals,
                    self.num_epochs,
                    exponential=True,
                    epoch_dir=os.path.join(self.outdir, "flow_matching", "epoch_data"),
                )
                val_signals, val_params = self._sample_dataset_batches(self.validation_dataset, n_val_signals)                
                
                self.h_theta_multi_val = hThetaMulti(
                    s=val_signals,
                    shared_max_strain=self.validation_dataset.shared_max_strain,
                    theta=val_params,
                    shared_min=self.validation_dataset.shared_min_theta,
                    shared_max=self.validation_dataset.shared_max_theta,
                    ra=val_sampled_ra,
                    dec=val_sampled_dec,
                    d=val_sampled_d,
                    batch_size=self.batch_size,
                    detector_noise_on=True,
                    random_polarization=True,
                    seed=epoch + 1000,  # Different seed range for validation set
                    intrinsic_param_names=self.intrinsic_params,
                    use_physics_aware_norm=self.use_physics_aware_norm
                )
                self.h_theta_multi_val_loader = DataLoader(
                    self.h_theta_multi_val,
                    shuffle=False,
                    **loader_kwargs,
                )

                for val_signal, val_noisy_signal, val_params in self.h_theta_multi_val_loader:
                    val_noisy_signal = val_noisy_signal.view(val_noisy_signal.size(0), -1).to(DEVICE, non_blocking=use_cuda)
                    val_params = val_params.view(val_params.size(0), -1).to(DEVICE, non_blocking=use_cuda)
                    val_params_target = self._extract_params_to_estimate(val_params)

                    if val_params_target.size(-1) != self.flow_param_dim:
                        raise ValueError(
                            f"Validation parameter dimension mismatch: expected {self.flow_param_dim}, got {val_params_target.size(-1)}"
                        )

                    x_0 = torch.randn_like(val_params_target)
                    t = torch.rand(len(val_params_target), 1, device=DEVICE)
                    x_t = (1 - t) * x_0 + t * val_params_target
                    dx_t = val_params_target - x_0

                    loss = self.loss_fn(self.flow(x_t, t, val_noisy_signal), dx_t)
                    val_total_loss += loss.item()
                    val_samples += val_signal.size(0)
            
            avg_total_loss_val = val_total_loss / val_samples
            self.avg_mse_losses_val.append(avg_total_loss_val)
            epoch_bar.set_postfix(
                train_loss=f"{avg_total_loss:.4f}",
                val_loss=f"{avg_total_loss_val:.4f}",
            )

            corner_epoch_dir = os.path.join(self.outdir, "flow_matching", "epoch_data")
            os.makedirs(corner_epoch_dir, exist_ok=True)

            self.run_parameter_estimation(signal_idx=None, d=None, ra=None, dec=None, epoch=epoch, transparent=True) 

            print(f"Epoch {epoch+1}/{self.num_epochs} | Train MSE Loss: {avg_total_loss:.4f} | Val MSE Loss: {avg_total_loss_val:.4f}")

        runtime = (time.time() - t0) / 60
        print(f"Training Time: {runtime:.2f}min")
        print("Plotting training/validation loss curves...")
        self.save_models()
        self.save_losses()
        self.display_results(fname=os.path.join(self.outdir, "flow_matching", "training_validation_losses.png"))  

    def _plot_project_to_detectors_steps(self, signal_idx, f_name_h, f_name_h_delayed, f_name_h_delayed_rescaled, f_name_h_delayed_rescaled_noise=None, font_family="Serif", font_name="Times New Roman"):
        signal_raw = self.validation_dataset.signals[:, signal_idx:signal_idx+1]  # Raw signal, shape (Y_LENGTH, 1)
        params = self.validation_dataset.parameters[signal_idx]  # Raw params, shape (num_params,)
        d = 5 # kpc
        
        distance_mask = (
            (self.supernovae.distances >= d - 0.25)
            & (self.supernovae.distances <= d + 0.25)
        )
        candidate_indices = np.where(distance_mask)[0]

        # Randomly select one supernova at that distance
        candidate_index = np.random.choice(candidate_indices)
        sampled_ra = np.array([self.supernovae.ra[candidate_index]])
        sampled_dec = np.array([self.supernovae.dec[candidate_index]])
        sampled_d = np.array([self.supernovae.distances[candidate_index]])
        
        # Wrap raw signal and params in tensors for hThetaMulti
        signals = [torch.tensor(signal_raw, dtype=torch.float32)]
        params_np = np.asarray(params)
        if params_np.ndim == 1:
            params_np = params_np.reshape(1, -1)

        temp_h_theta_multi = hThetaMulti(
            s=signals,  # List of tensors with RAW signals
            shared_max_strain=self.validation_dataset.shared_max_strain,
            theta=params_np,  # Tensor
            shared_min=self.validation_dataset.shared_min_theta,
            shared_max=self.validation_dataset.shared_max_theta,
            ra=sampled_ra,
            dec=sampled_dec,
            d=sampled_d,
            batch_size=self.batch_size,
            detector_noise_on=True,  # Add fresh detector noise, consistent with train()
            random_polarization=True,
            seed=1,
            intrinsic_param_names=self.intrinsic_params,
            use_physics_aware_norm=self.use_physics_aware_norm
        )

        temp_h_theta_multi._plot_project_to_detectors_steps(
            signal_idx=0,
            f_name_h=f_name_h,
            f_name_h_delayed=f_name_h_delayed,
            f_name_h_delayed_rescaled=f_name_h_delayed_rescaled,
            f_name_h_delayed_rescaled_noise=f_name_h_delayed_rescaled_noise,
            font_family=font_family,
            font_name=font_name
        )


    def _generate_posterior_samples(self, sampled_case, h_theta_multi_dataset, num_samples=3000, n_steps=20):
        """Generate posterior samples for a given signal case.
        
        Args:
            sampled_case: Tuple of (signal, noisy_signal, params)
            h_theta_multi_dataset: Dataset containing normalization bounds
            num_samples: Number of posterior samples
            n_steps: Number of ODE solver steps
            
        Returns:
            Tuple of (samples_denorm, true_params_denorm) as numpy arrays
        """
        self.flow.eval()
        
        _, noisy_signal, params = sampled_case
        
        if noisy_signal.dim() == 2:
            noisy_signal = noisy_signal.unsqueeze(0)
        if params.dim() == 1:
            params = params.unsqueeze(0)
        
        noisy_signal = noisy_signal.view(noisy_signal.size(0), -1).to(DEVICE).float()
        params = params.view(params.size(0), -1).to(DEVICE).float()
        
        t0 = time.time()
        
        with torch.no_grad():
            posterior_samples = torch.randn(num_samples, self.flow_param_dim, device=DEVICE)
            repeated_signal = noisy_signal.repeat(num_samples, 1)
            
            time_steps = torch.linspace(0, 1.0, n_steps + 1)
            for i in range(n_steps):
                posterior_samples = self.flow.step(
                    posterior_samples,
                    time_steps[i],
                    time_steps[i + 1],
                    repeated_signal,
                )
            
            samples_cpu = posterior_samples.detach().cpu().numpy()
            true_params_norm = params.detach().cpu().numpy().flatten()
            
            # DEBUG: Check raw flow outputs (before denormalization)
            if self.use_physics_aware_norm:
                print(f"\n=== RAW FLOW OUTPUT (Physics-Aware Normalized Space) ===")
                print(f"Shape: {samples_cpu.shape} (11D: intrinsic params + 2D cyclic + 1D distance)")
                print(f"Overall range: [{samples_cpu.min():.4f}, {samples_cpu.max():.4f}]")
                print("Note: (cos, sin) pairs should be in [-1, 1], distance in [0, 1]")
            else:
                print(f"\n=== RAW FLOW OUTPUT (Normalized Space [-1, 1]) ===")
                print(f"Shape: {samples_cpu.shape}")
                for i, param_name in enumerate(self.parameters_to_estimate):
                    param_samples = samples_cpu[:, i]
                    print(f"  {param_name:10s}: min={param_samples.min():.4f}, max={param_samples.max():.4f}, "
                          f"mean={param_samples.mean():.4f}, std={param_samples.std():.4f}")
                    n_outside = np.sum((param_samples < -1.0) | (param_samples > 1.0))
                    if n_outside > 0:
                        print(f"             WARNING: {n_outside}/{len(param_samples)} samples OUTSIDE [-1, 1]!")
        
        # Denormalize parameters
        if self.toy:
            samples_denorm = samples_cpu
            true_params_denorm = true_params_norm
        else:
            if self.use_physics_aware_norm:
                # Physics-aware denormalization: full 11D → 8D physical
                # Samples are in physics-aware space (11D for all 8 params)
                samples_full_denorm = h_theta_multi_dataset.denormalize_parameters_physics_aware(samples_cpu)  # (N_samples, 8)
                true_params_full_denorm = h_theta_multi_dataset.denormalize_parameters_physics_aware(
                    true_params_norm.reshape(1, -1)
                ).flatten()  # (8,)
                
                # Extract only the requested parameters in physical space
                samples_denorm = samples_full_denorm[:, self.param_extract_indices]  # (N_samples, n_requested)
                true_params_denorm = true_params_full_denorm[self.param_extract_indices]  # (n_requested,)
                
                # DEBUG: Check denormalized samples
                print(f"\n=== DENORMALIZED OUTPUT (Physical Units) - Physics-Aware ===")
                for i, param_name in enumerate(self.parameters_to_estimate):
                    param_idx = self.param_extract_indices[i]
                    param_samples = samples_full_denorm[:, param_idx]
                    min_bound = h_theta_multi_dataset.shared_min_theta[param_idx]
                    max_bound = h_theta_multi_dataset.shared_max_theta[param_idx]
                    print(f"  {param_name:10s}: min={param_samples.min():.4f}, max={param_samples.max():.4f}")
                    print(f"             bounds: [{min_bound:.4f}, {max_bound:.4f}]")
                    n_outside = np.sum((param_samples < min_bound) | (param_samples > max_bound))
                    if n_outside > 0:
                        print(f"             WARNING: {n_outside}/{len(param_samples)} samples OUTSIDE bounds!")
            else:
                # Linear denormalization: extract then denormalize
                samples_denorm = self._denormalize_extracted_params(samples_cpu, h_theta_multi_dataset)
                
                # Extract relevant parameters from true_params if dataset has all 8 parameters
                num_params_in_dataset = len(h_theta_multi_dataset.shared_min_theta)
                if num_params_in_dataset == 8:
                    # Dataset has all parameters; extract only the requested ones
                    true_params_extracted = true_params_norm[self.param_extract_indices]
                else:
                    # Dataset has only requested parameters
                    true_params_extracted = true_params_norm
                
                true_params_denorm = self._denormalize_extracted_params(
                    true_params_extracted.reshape(1, -1), h_theta_multi_dataset
                ).flatten()
                
                # DEBUG: Check denormalized samples
                print(f"\n=== DENORMALIZED OUTPUT (Physical Units) ===")
                for i, param_name in enumerate(self.parameters_to_estimate):
                    param_samples = samples_denorm[:, i]
                    min_bound = h_theta_multi_dataset.shared_min_theta[i]
                    max_bound = h_theta_multi_dataset.shared_max_theta[i]
                    print(f"  {param_name:10s}: min={param_samples.min():.4f}, max={param_samples.max():.4f}")
                    print(f"             bounds: [{min_bound:.4f}, {max_bound:.4f}]")
                    n_outside = np.sum((param_samples < min_bound) | (param_samples > max_bound))
                    if n_outside > 0:
                        print(f"             WARNING: {n_outside}/{len(param_samples)} samples OUTSIDE bounds!")
        
        t1 = time.time()
        print(f"Posterior sampling and denormalisation took {(t1 - t0):.2f}s")
        
        return samples_denorm, true_params_denorm

    def plot_corner_sampled_signal(
        self,
        fname: str = "plots/corner_plot_sampled_signal.png",
        posterior_samples_denorm=None,
        true_param_denorm=None,
        background: str = "white",
        font_family: str = "Serif",
        font_name: str = "Times New Roman",
        transparent: bool = False
    ):
        """Generate a corner plot for posterior samples.

        Args:
            fname: Output filename for the plot
            posterior_samples_denorm: Posterior samples in denormalized parameter space
            true_param_denorm: True parameters in denormalized space
            background: Background color for plot
            font_family: Font family for plot text
            font_name: Font name for plot text
            transparent: Whether to save with transparent background
        """
        from starccato_flow.utils.plotting_defaults import PARAMETER_LABELS, PARAMETER_RANGES
        
        # Clip posterior samples to valid ranges for corner plot visualization only
        # This prevents histogram errors from out-of-bounds samples without affecting sky plots
        posterior_samples_clipped = posterior_samples_denorm.copy()
        for i, param in enumerate(self.parameters_to_estimate):
            if param in PARAMETER_RANGES:
                min_val, max_val = PARAMETER_RANGES[param]
                posterior_samples_clipped[:, i] = np.clip(posterior_samples_clipped[:, i], min_val, max_val)
        
        # Convert parameter names to LaTeX labels using plotting_defaults
        latex_labels = [PARAMETER_LABELS.get(param, param) for param in self.parameters_to_estimate]
        
        # Use plotting_defaults bounds for all parameters
        # This provides consistent, broader ranges for visualization across all runs
        ranges = []
        for param in self.parameters_to_estimate:
            if param in PARAMETER_RANGES:
                ranges.append(PARAMETER_RANGES[param])
            else:
                # Fallback: use data-driven bounds from the actual samples
                i = self.parameters_to_estimate.index(param)
                sample_min = np.nanmin(posterior_samples_clipped[:, i])
                sample_max = np.nanmax(posterior_samples_clipped[:, i])
                span = max(sample_max - sample_min, 1e-8)
                pad = 0.03 * span
                ranges.append((float(sample_min - pad), float(sample_max + pad)))
        
        # Debug: print ranges for each parameter
        print("\nPlot axis ranges (from plotting_defaults PARAMETER_RANGES):")
        for i, label in enumerate(self.parameters_to_estimate):
            print(f"  {label:20s}: {ranges[i]}")

        # Debug: validate ranges and samples before plotting
        print("\nDebug: Validating ranges and samples...")
        print(f"  Posterior samples shape: {posterior_samples_denorm.shape}")
        print(f"  True param shape: {true_param_denorm.shape}")
        print(f"  Number of ranges: {len(ranges)}")
        print(f"  Number of sample dimensions: {posterior_samples_denorm.shape[1] if posterior_samples_denorm.ndim > 1 else 1}")
        
        # Check for NaN values in samples
        nan_count = np.isnan(posterior_samples_denorm).sum()
        inf_count = np.isinf(posterior_samples_denorm).sum()
        print(f"  NaN values in samples: {nan_count}")
        print(f"  Inf values in samples: {inf_count}")
        
        # Validate ranges and check if samples fall within them
        print(f"  Validating ranges:")
        samples_in_range = True
        for i, (r_min, r_max) in enumerate(ranges):
            if r_min >= r_max:
                print(f"    ERROR: Range {i} ({r_min}, {r_max}) is invalid (min >= max)!")
            else:
                print(f"    Range {i}: ({r_min:.4f}, {r_max:.4f}) ✓")
            # Also check sample statistics for this dimension
            if posterior_samples_denorm.ndim > 1 and i < posterior_samples_denorm.shape[1]:
                sample_min = np.nanmin(posterior_samples_denorm[:, i])
                sample_max = np.nanmax(posterior_samples_denorm[:, i])
                print(f"      Sample range: [{sample_min:.4f}, {sample_max:.4f}]")
                
                # Check if any samples fall within the specified range
                in_range = np.sum((posterior_samples_denorm[:, i] >= r_min) & 
                                 (posterior_samples_denorm[:, i] <= r_max))
                pct_in_range = 100.0 * in_range / len(posterior_samples_denorm)
                print(f"      Samples in range: {in_range}/{len(posterior_samples_denorm)} ({pct_in_range:.1f}%)")
                
                if pct_in_range < 1:
                    print(f"      WARNING: <1% of samples fall in this range!")
                    samples_in_range = False
        
        if not samples_in_range:
            print("\nWARNING: Some parameters have <1% of samples in the specified range.")
            print("This may cause corner plot to fail. Consider expanding ranges or checking sample bounds.")
        
        plot_corner(
            samples_cpu=posterior_samples_clipped,
            true_param=true_param_denorm,
            fname=fname,
            labels=latex_labels,
            ranges=ranges,
            background=background,
            font_family=font_family,
            font_name=font_name,
        )

    def plot_sky_localisation_sampled_signal(
        self,
        fname: str = "plots/sky_localisation_sampled_signal.png",
        posterior_samples_denorm=None,
        true_param_denorm=None,
        font_family: str = "Serif",
        font_name: str = "Times New Roman",
        transparent: bool = False
    ):
        """Generate a sky-localisation (RA/Dec) posterior plot.
        
        Args:
            fname: Output filename for the plot
            posterior_samples_denorm: Posterior samples in denormalized parameter space
            true_param_denorm: True parameters in denormalized space
            font_family: Font family for plot text
            font_name: Font name for plot text
            transparent: Whether to save with transparent background
        """
        # Extract RA and Dec indices from the parameters_to_estimate list
        ra_idx = self._get_extracted_index("ra")
        dec_idx = self._get_extracted_index("dec")
        
        print(f"\nSky Localisation Plot Debug:")
        print(f"  ra_idx: {ra_idx}, dec_idx: {dec_idx}")
        print(f"  true_param_denorm shape: {true_param_denorm.shape}")
        print(f"  parameters_to_estimate: {self.parameters_to_estimate}")
        
        if ra_idx >= 0 and dec_idx >= 0:
            # Extract RA and Dec from the denormalized extracted parameters
            ra_samples = posterior_samples_denorm[:, ra_idx]
            dec_samples = posterior_samples_denorm[:, dec_idx]
            true_ra = true_param_denorm[ra_idx]
            true_dec = true_param_denorm[dec_idx]
            print(f"  true_ra: {true_ra:.6f} rad = {np.rad2deg(true_ra):.2f}°")
            print(f"  true_dec: {true_dec:.6f} rad = {np.rad2deg(true_dec):.2f}°")
            
            # For SVG output, use a reduced sample size to keep file size manageable
            # SVG files with many points can become very large; 500-800 points provides good density visualization
            if fname.endswith('.svg'):
                max_samples = min(800, len(ra_samples))
                sample_indices = np.linspace(0, len(ra_samples) - 1, max_samples, dtype=int)
                ra_samples = ra_samples[sample_indices]
                dec_samples = dec_samples[sample_indices]
                print(f"  SVG output: Using {len(ra_samples)} samples (downsampled from {posterior_samples_denorm.shape[0]})")
        else:
            # Fallback: assume they are at the end (shouldn't happen with proper setup)
            ra_samples = posterior_samples_denorm[:, -4]
            dec_samples = posterior_samples_denorm[:, -3]
            true_ra = true_param_denorm[-4]
            true_dec = true_param_denorm[-3]
            print(f"  FALLBACK: true_ra: {true_ra:.6f} rad = {np.rad2deg(true_ra):.2f}°")
            print(f"  FALLBACK: true_dec: {true_dec:.6f} rad = {np.rad2deg(true_dec):.2f}°")

        plot_galactic_supernovae_polar_hemispheres(
            ccsn=self.supernovae,
            fname=fname,
            posterior_ra_samples=ra_samples,
            posterior_dec_samples=dec_samples,
            true_ra_override=true_ra,
            true_dec_override=true_dec,
            show_constellation_borders=True,
            show_all_constellation_labels=False if fname.endswith('.svg') else True,  # Skip labels for SVG output
            dpi=150 if fname.endswith('.svg') else 300,  # Lower DPI for SVG to reduce file size
            background="black",
            font_family=font_family,
            font_name=font_name,
            red_blob_mode="density_peak",
            transparent=transparent,
        )
        
        # Compress SVG files to .svgz format for ~75% size reduction while keeping all visual elements
        if fname.endswith('.svg'):
            import gzip
            import shutil
            svgz_fname = fname.replace('.svg', '.svgz')
            with open(fname, 'rb') as f_in:
                with gzip.open(svgz_fname, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            import os
            svg_size = os.path.getsize(fname) / 1024  # KB
            svgz_size = os.path.getsize(svgz_fname) / 1024  # KB
            print(f"  SVG compression: {svg_size:.0f} KB → {svgz_size:.0f} KB ({100*svgz_size/svg_size:.0f}%)")


    def plot_galactic_distribution_with_posterior(
        self,
        fname: str = "plots/galactic_distribution_posterior.png",
        posterior_samples_denorm=None,
        true_param_denorm=None,
        font_family: str = "Serif",
        font_name: str = "Times New Roman",
        transparent: bool = False,
        background: str = "white"
    ):
        """Plot galactic distribution (X-Y plane) with posterior credible regions overlaid.
        
        This combines the background galactic supernova distribution with posterior density
        contours computed from RA/Dec/distance samples transformed to galactic Cartesian coordinates.
        
        Args:
            fname: Output filename for the plot
            posterior_samples_denorm: Posterior samples in denormalized parameter space
            true_param_denorm: True parameters in denormalized space
            font_family: Font family for plots
            font_name: Font name for plots
            transparent: Whether to save with transparent background
            background: Background color for plots
        """
        from ..plotting.analysis import plot_galactic_distribution_with_posterior
        
        # Extract RA, Dec, and distance indices
        ra_idx = self._get_extracted_index("ra")
        dec_idx = self._get_extracted_index("dec")
        d_idx = self._get_extracted_index("d")
        
        if ra_idx >= 0 and dec_idx >= 0 and d_idx >= 0:
            # Extract RA, Dec, distance from the denormalized extracted parameters
            ra_samples = posterior_samples_denorm[:, ra_idx]
            dec_samples = posterior_samples_denorm[:, dec_idx]
            d_samples = posterior_samples_denorm[:, d_idx]
            true_ra = true_param_denorm[ra_idx]
            true_dec = true_param_denorm[dec_idx]
            true_d = true_param_denorm[d_idx]
        else:
            # Fallback: assume they are at the end
            ra_samples = posterior_samples_denorm[:, -4]
            dec_samples = posterior_samples_denorm[:, -3]
            d_samples = posterior_samples_denorm[:, -2]
            true_ra = true_param_denorm[-4]
            true_dec = true_param_denorm[-3]
            true_d = true_param_denorm[-2]
        
        # Get galactic distribution coordinates
        galactic_coords = self.supernovae.galactic_coords
        sun_location = self.supernovae.SUN_LOCATION
        
        plot_galactic_distribution_with_posterior(
            galactic_coords=galactic_coords,
            posterior_ra=ra_samples,
            posterior_dec=dec_samples,
            posterior_distance=d_samples,
            true_ra=true_ra,
            true_dec=true_dec,
            true_distance=true_d,
            sun_location=sun_location,
            fname=fname,
            background=background,
            font_family=font_family,
            font_name=font_name,
            transparent=transparent,
        )

    def plot_galactic_distribution_with_posterior_zoom(
        self,
        fname: str = "plots/galactic_distribution_posterior_zoom.png",
        posterior_samples_denorm=None,
        true_param_denorm=None,
        font_family: str = "Serif",
        font_name: str = "Times New Roman",
        transparent: bool = False,
        background: str = "white"
    ):
        """Plot galactic distribution (X-Y plane) with posterior contours in 10 kpc zoom around sun.
        
        This is a zoomed version showing only the region within 10 kpc of the sun, with no legend,
        ticks, or axis markers.
        
        Args:
            fname: Output filename for the plot
            posterior_samples_denorm: Posterior samples in denormalized parameter space
            true_param_denorm: True parameters in denormalized space
            font_family: Font family for plots
            font_name: Font name for plots
            transparent: Whether to save with transparent background
            background: Background color for plots
        """
        from ..plotting.analysis import plot_galactic_distribution_with_posterior_zoom
        
        # Extract RA, Dec, and distance indices
        ra_idx = self._get_extracted_index("ra")
        dec_idx = self._get_extracted_index("dec")
        d_idx = self._get_extracted_index("d")
        
        if ra_idx >= 0 and dec_idx >= 0 and d_idx >= 0:
            # Extract RA, Dec, distance from the denormalized extracted parameters
            ra_samples = posterior_samples_denorm[:, ra_idx]
            dec_samples = posterior_samples_denorm[:, dec_idx]
            d_samples = posterior_samples_denorm[:, d_idx]
            true_ra = true_param_denorm[ra_idx]
            true_dec = true_param_denorm[dec_idx]
            true_d = true_param_denorm[d_idx]
        else:
            # Fallback: assume they are at the end
            ra_samples = posterior_samples_denorm[:, -4]
            dec_samples = posterior_samples_denorm[:, -3]
            d_samples = posterior_samples_denorm[:, -2]
            true_ra = true_param_denorm[-4]
            true_dec = true_param_denorm[-3]
            true_d = true_param_denorm[-2]
        
        # Get galactic distribution coordinates
        galactic_coords = self.supernovae.galactic_coords
        sun_location = self.supernovae.SUN_LOCATION
        
        plot_galactic_distribution_with_posterior_zoom(
            galactic_coords=galactic_coords,
            posterior_ra=ra_samples,
            posterior_dec=dec_samples,
            posterior_distance=d_samples,
            true_ra=true_ra,
            true_dec=true_dec,
            true_distance=true_d,
            sun_location=sun_location,
            fname=fname,
            figsize_mm=(125, 125),
            background=background,
            font_family=font_family,
            font_name=font_name,
            transparent=transparent,
        )

    def plot_candidate_signal(self, snr=100, background="white", index=0, fname="plots/candidate_signal.png"):
        """Plot a candidate signal with noise."""
        self.val_loader.dataset.update_snr(snr)
        signal, noisy_signal, _ = self.val_loader.dataset.__getitem__(index)
        signal_denorm = self.val_loader.dataset.denormalise_signals(signal) / TEN_KPC
        noisy_signal_denorm = self.val_loader.dataset.denormalise_signals(noisy_signal) / TEN_KPC
        plot_candidate_signal(
            signal=signal_denorm,
            noisy_signal=noisy_signal_denorm,
            max_value=self.val_loader.dataset.shared_max_strain,
            background=background,
            fname=fname
        )

    def plot_pp_coverage_validation(self, num_signals: int = 2000, num_samples: int = 3000, n_steps: int = 20, 
                                     fname: Optional[str] = None, background: str = "white", font_family: str = "Serif", font_name: str = "Times New Roman", transparent: bool = False) -> None:
        """Generate a p-p (credible interval coverage) plot using validation set signals.
        
        This plot shows empirical vs theoretical coverage of credible intervals for each parameter.
        Each parameter is represented as a separate line. Perfect calibration is shown as the diagonal.
        
        Args:
            num_signals (int): Number of validation signals to use for computing coverage
            num_samples (int): Number of posterior samples to draw per signal
            n_steps (int): Number of ODE solver steps for inference
            fname (Optional[str]): Filename to save plot. If None, saves to outdir/flow_matching/pp_coverage_validation.png
            background (str): Background color theme ("white" or "black")
            font_family (str): Font family for plot text
            font_name (str): Font name for plot text
            transparent (bool): Whether to make the plot background transparent
        """
        # Initialize validation dataset if not already present
        if not hasattr(self, 'h_theta_multi_val') or self.h_theta_multi_val is None:            
            # Sample validation signals and parameters
            val_signals, val_params = self._sample_dataset_batches(self.validation_dataset, num_signals)
            
            # Sample sky parameters for validation set
            # Use uniform sampling (not exponential) for representative calibration assessment
            val_sampled_ra, val_sampled_dec, val_sampled_d = self.supernovae.sample_supernovae_for_epoch(
                epoch=self.num_epochs,
                n_samples=num_signals,
                num_epochs=self.num_epochs,
                exponential=False,  # Uniform sampling for p-p plot calibration
                epoch_dir=os.path.join(self.outdir, "flow_matching", "epoch_data"),
            )
            
            # Create hThetaMulti validation dataset
            h_theta_multi_val = hThetaMulti(
                s=val_signals,
                shared_max_strain=self.validation_dataset.shared_max_strain,
                theta=val_params,
                shared_min=self.validation_dataset.shared_min_theta,
                shared_max=self.validation_dataset.shared_max_theta,
                ra=val_sampled_ra,
                dec=val_sampled_dec,
                d=val_sampled_d,
                batch_size=self.batch_size,
                detector_noise_on=True,
                random_polarization=True,
                seed=1000,
                intrinsic_param_names=self.intrinsic_params,
                use_physics_aware_norm=self.use_physics_aware_norm
            )
            print(f"✓ Created validation dataset with {len(h_theta_multi_val)} signals")
                
        self.flow.eval()
        
        posterior_samples_list = []
        true_params_list = []
        
        # Create DataLoader for efficient batch iteration
        use_cuda = str(DEVICE).startswith("cuda")
        loader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": 0,
            "pin_memory": use_cuda,
            "persistent_workers": False,
        }
        val_loader = DataLoader(
            h_theta_multi_val,
            shuffle=False,
            **loader_kwargs,
        )
        
        total_processed = 0
        with torch.no_grad():
            for signals_batch, noisy_signals_batch, true_params_batch in val_loader:
                # Determine how many samples to process from this batch
                num_to_process = min(len(signals_batch), num_signals - total_processed)
                if num_to_process <= 0:
                    break
                
                # Process each sample in the batch
                for i in range(num_to_process):
                    if (total_processed + 1) % max(1, num_signals // 5) == 0:
                        print(f"  Processed {total_processed + 1}/{num_signals} signals...")
                    
                    # Extract individual sample from batch
                    noisy_signal = noisy_signals_batch[i:i+1]
                    true_params = true_params_batch[i:i+1]
                    
                    if noisy_signal.dim() == 2:
                        noisy_signal = noisy_signal.unsqueeze(0)
                    if true_params.dim() == 1:
                        true_params = true_params.unsqueeze(0)
                    
                    noisy_signal = noisy_signal.view(noisy_signal.size(0), -1).to(DEVICE).float()
                    true_params = true_params.view(true_params.size(0), -1).to(DEVICE).float()
                    
                    # Generate posterior samples using flow
                    posterior_samples = torch.randn(num_samples, self.flow_param_dim, device=DEVICE)
                    repeated_signal = noisy_signal.repeat(num_samples, 1)
                    
                    time_steps = torch.linspace(0, 1.0, n_steps + 1)
                    for j in range(n_steps):
                        posterior_samples = self.flow.step(
                            posterior_samples,
                            time_steps[j],
                            time_steps[j + 1],
                            repeated_signal,
                        )
                    
                    # Denormalize parameters
                    if self.use_physics_aware_norm:
                        # Physics-aware denormalization
                        samples_full_denorm = h_theta_multi_val.denormalize_parameters_physics_aware(
                            posterior_samples.cpu().numpy()
                        )
                        samples_denorm = samples_full_denorm[:, self.param_extract_indices]
                        
                        true_params_full_denorm = h_theta_multi_val.denormalize_parameters_physics_aware(
                            true_params.cpu().numpy()
                        )
                        true_params_denorm = true_params_full_denorm[:, self.param_extract_indices]
                    else:
                        # Linear denormalization
                        samples_denorm = self._denormalize_extracted_params(
                            posterior_samples.cpu().numpy(), 
                            h_theta_multi_val
                        )
                        # Extract only the requested parameters from the full 8-parameter vector
                        true_params_extracted = true_params[:, self.param_extract_indices].cpu().numpy()
                        true_params_denorm = self._denormalize_extracted_params(
                            true_params_extracted,
                            h_theta_multi_val
                        )
                    
                    posterior_samples_list.append(samples_denorm)
                    true_params_list.append(true_params_denorm.flatten())
                    
                    total_processed += 1
        
        print(f"Generating p-p coverage plot...")
        
        # Create p-p coverage plot
        plot_pp_coverage(
            posterior_samples_list=posterior_samples_list,
            true_params_list=true_params_list,
            param_names=self.parameters_to_estimate,
            fname=fname,
            background=background,
            font_family=font_family,
            font_name=font_name,
            transparent=transparent
        )
        
        print(f"✓ Saved p-p coverage plot to {fname}")

    def _extract_params_to_estimate(self, full_params: torch.Tensor) -> torch.Tensor:
        """Extract only the parameters we're estimating from the full parameter tensor.
        
        For toy data, returns all parameters. For real data, extracts only requested parameters
        using the parameter_mapping indices. With physics-aware normalization, returns the full
        normalized parameters (which are already in the reduced space from __getitem__).
        """
        if self.toy:
            return full_params
        
        # With physics-aware normalization, the dataset __getitem__ already returns
        # the full normalized parameters in physics-aware space. We don't need to extract
        # since the dataset handles both full 8D and physics-aware normalization internally.
        # The network is trained on the full normalized space.
        if self.use_physics_aware_norm:
            return full_params
        
        # With linear normalization, extract only the requested parameter indices
        return full_params[:, self.param_extract_indices]
    
    def display_results(self, background="black", fname=None, font_family="sans-serif", font_name="Avenir") -> None:
        """Display training results."""
        plot_loss(self.avg_mse_losses, self.avg_mse_losses_val, loss_type="Mean Squared Error Loss", train_label="Training Mean Squared Error Loss", val_label="Validation Mean Squared Error Loss", background=background, fname=fname, font_family=font_family, font_name=font_name)        
        
    @property
    def save_fname(self):
        return f"{self.outdir}/flow_sky_weights_test.pt"

    def save_data(self):
        """Save flow model and training losses to disk (NPZ format for consistency with CVAE trainer)."""
        torch.save(self.flow.state_dict(), self.save_fname)
        print(f"Saved Flow model to {self.save_fname}")
        
        # Save losses to npz file (consistent with CVAE trainer)
        losses_path = f"{self.outdir}/flow_losses_test.npz"
        np.savez(
            losses_path,
            avg_mse_losses=np.array(self.avg_mse_losses),
            avg_mse_losses_val=np.array(self.avg_mse_losses_val)
        )
        print(f"Saved losses to {losses_path}")
    
    def save_models(self):
        """Save flow model (deprecated: use save_data() instead for combined model+losses saving)."""
        torch.save(self.flow.state_dict(), self.save_fname)
        print(f"Saved Flow model to {self.save_fname}")
    
    def save_losses(self, fname: str = None):
        """Save training and validation losses to a CSV file (optional, for compatibility).
        
        Args:
            fname: Output CSV filename. If None, saves to outdir/flow_matching/losses.csv
        """
        if fname is None:
            fname = os.path.join(self.outdir, "flow_matching", "losses.csv")
        
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        
        # Create DataFrame with losses
        data = {
            'epoch': np.arange(len(self.avg_mse_losses)),
            'train_loss': self.avg_mse_losses,
            'val_loss': self.avg_mse_losses_val
        }
        
        # Add gradient norms if available (pad with NaN if fewer entries)
        if hasattr(self, 'flow_gradient_norms') and len(self.flow_gradient_norms) > 0:
            # Average gradient norms per epoch (there are many batches per epoch)
            samples_per_epoch = self.samples_per_epoch
            batch_size = self.batch_size
            batches_per_epoch = max(1, samples_per_epoch // batch_size)
            
            epoch_avg_grads = []
            for epoch_idx in range(len(self.avg_mse_losses)):
                start_idx = epoch_idx * batches_per_epoch
                end_idx = min((epoch_idx + 1) * batches_per_epoch, len(self.flow_gradient_norms))
                if start_idx < len(self.flow_gradient_norms):
                    epoch_avg_grads.append(np.mean(self.flow_gradient_norms[start_idx:end_idx]))
                else:
                    epoch_avg_grads.append(np.nan)
            data['avg_gradient_norm'] = epoch_avg_grads
        
        df = pd.DataFrame(data)
        df.to_csv(fname, index=False)
        print(f"Saved losses to {fname}")
        return fname
    
    def load_losses(self, fname: str = None) -> dict:
        """Load training and validation losses from a CSV file.
        
        Args:
            fname: Input CSV filename. If None, loads from outdir/flow_matching/losses.csv
            
        Returns:
            Dictionary with keys 'train_loss' and 'val_loss' containing the loss arrays
        """
        if fname is None:
            fname = os.path.join(self.outdir, "flow_matching", "losses.csv")
        
        if not os.path.exists(fname):
            print(f"Losses file not found at {fname}")
            return None
        
        df = pd.read_csv(fname)
        self.avg_mse_losses = df['train_loss'].values.tolist()
        self.avg_mse_losses_val = df['val_loss'].values.tolist()
        
        # Load gradient norms if available
        if 'avg_gradient_norm' in df.columns:
            self.flow_gradient_norms = df['avg_gradient_norm'].dropna().values.tolist()
        
        print(f"Loaded {len(self.avg_mse_losses)} training loss entries from {fname}")
        
        return {
            'train_loss': self.avg_mse_losses,
            'val_loss': self.avg_mse_losses_val,
            'avg_gradient_norm': self.flow_gradient_norms if hasattr(self, 'flow_gradient_norms') else None
        }

    @classmethod
    def load_model(
        cls,
        model_path: str,
        param_dim: int = 5
    ) -> 'FlowFCL':
        """Load a trained Flow model from disk.
        
        Args:
            model_path: Path to the saved model weights (.pt file)
            param_dim: Number of physical parameters
            
        Returns:
            Loaded Flow model
        """
        # Reconstruct model architecture with correct parameter names
        signal_dim = Y_LENGTH * 3  # 3 detectors
        flow = FlowFCL(dim=param_dim, signal_dim=signal_dim).to(DEVICE)
        
        # Load saved weights
        flow.load_state_dict(torch.load(model_path, map_location=DEVICE))
        flow.eval()
        
        print(f"✓ Loaded Flow model from {model_path}")        
        return flow
    
    def load_model_instance(self, model_path: str) -> None:
        """Load a trained Flow model into this trainer instance.
        
        Args:
            model_path: Path to the saved model weights (.pt file)
        """
        self.flow = self.load_model(model_path, param_dim=self.flow_param_dim)
        print(f"✓ Loaded model into trainer from {model_path}")
    
    def load_pretrained(self, model_path: str) -> None:
        """Load pretrained weights and loss history into the trainer's model.
        
        Args:
            model_path: Path to the saved model weights (.pt file)
        """
        self.flow.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.flow.eval()
        print(f"✓ Loaded pretrained weights from {model_path}")
        
        # Try to load loss history from the same directory (NPZ format)
        losses_path = model_path.replace('flow_weights_final.pt', 'flow_losses.npz')
        losses_path = losses_path.replace('flow_weights.pt', 'flow_losses.npz')
        
        if os.path.exists(losses_path):
            losses = np.load(losses_path)
            self.avg_mse_losses = losses['avg_mse_losses'].tolist()
            self.avg_mse_losses_val = losses['avg_mse_losses_val'].tolist()
            print(f"✓ Loaded loss history from {losses_path}")
        else:
            # Initialize empty loss lists if file not found
            self.avg_mse_losses = []
            self.avg_mse_losses_val = []


    def export_strain_and_parameters(self, signal_idx: int, fname_prefix: str, ra: float = None, dec: float = None, d: float = None):
        """Export the signal and parameters for a specific index to CSV files (one per detector).
        
        Args:
            signal_idx: Index of the signal in the validation dataset
            fname_prefix: Prefix for output filenames
            ra: Optional RA coordinate (radians). If None, randomly selects a supernova
            dec: Optional Dec coordinate (radians). If None, randomly selects a supernova
            d: Optional distance (kpc). If None, randomly selects a supernova
        """
        if signal_idx < 0 or signal_idx >= self.validation_dataset.signals.shape[1]:
            raise IndexError(f"Signal index {signal_idx} is out of bounds for validation dataset with {self.validation_dataset.signals.shape[1]} signals.")
        
        # Get raw signal and parameters
        signal_raw = self.validation_dataset.signals[:, signal_idx:signal_idx+1]
        params = self.validation_dataset.parameters[signal_idx]
        
        # Default to random selection if sky parameters not specified
        if ra is None or dec is None or d is None:
            raise ValueError("Must specify ra, dec, and d for exporting multi-detector strains")
        
        # Wrap in tensors for hThetaMulti
        signals = [torch.tensor(signal_raw, dtype=torch.float32)]
        params_np = np.asarray(params)
        if params_np.ndim == 1:
            params_np = params_np.reshape(1, -1)
        params_tensor = torch.tensor(params_np, dtype=torch.float32)
        
        # Create multi-channel dataset
        sampled_ra = np.array([ra])
        sampled_dec = np.array([dec])
        sampled_d = np.array([d])
        
        temp_h_theta_multi = hThetaMulti(
            s=signals,
            shared_max_strain=self.validation_dataset.shared_max_strain,
            theta=params_tensor,
            shared_min=self.validation_dataset.shared_min_theta,
            shared_max=self.validation_dataset.shared_max_theta,
            ra=sampled_ra,
            dec=sampled_dec,
            d=sampled_d,
            batch_size=self.batch_size,
            detector_noise_on=True,
            random_polarization=True,
            seed=1,
            intrinsic_param_names=self.intrinsic_params,
            use_physics_aware_norm=self.use_physics_aware_norm
        )
        
        # Get the multi-channel signals
        clean_signal, noisy_signal, _ = temp_h_theta_multi[0]
        
        # Export each detector's strain separately
        detector_labels = temp_h_theta_multi.detectors
        for i, detector in enumerate(detector_labels):
            clean_strain = clean_signal[i].cpu().numpy() if isinstance(clean_signal, torch.Tensor) else clean_signal[i]
            noisy_strain = noisy_signal[i].cpu().numpy() if isinstance(noisy_signal, torch.Tensor) else noisy_signal[i]
            
            np.savetxt(f"{fname_prefix}_{detector}_clean.csv", clean_strain, delimiter=",")
            np.savetxt(f"{fname_prefix}_{detector}_noisy.csv", noisy_strain, delimiter=",")
        
        # Export parameters
        params_export = params.cpu().numpy() if isinstance(params, torch.Tensor) else params
        np.savetxt(f"{fname_prefix}_parameters.csv", params_export, delimiter=",")
        
        print(f"Exported multi-detector strains for signal index {signal_idx}:")
        for detector in detector_labels:
            print(f"  {fname_prefix}_{detector}_clean.csv")
            print(f"  {fname_prefix}_{detector}_noisy.csv")
        print(f"  {fname_prefix}_parameters.csv")

    def load_models(self):
        """Load pre-trained flow model weights from checkpoint."""
        if not os.path.exists(f"{self.outdir}/flow_weights_final.pt"):
            raise FileNotFoundError(f"Model checkpoint not found at {f'{self.outdir}/flow_weights_final.pt'}")
        
        state_dict = torch.load(f"{self.outdir}/flow_weights_final.pt", map_location=DEVICE)
        self.flow.load_state_dict(state_dict)
        self.flow.to(DEVICE)
        print(f"Loaded Flow model from {f'{self.outdir}/flow_weights_final.pt'}")