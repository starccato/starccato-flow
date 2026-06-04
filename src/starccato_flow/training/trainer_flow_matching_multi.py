import os
import time
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data.s_theta import sTheta
from ..data.h_theta_multi import hThetaMulti
from ..supernovae.supernovae import Supernovae
from tqdm.auto import trange
  
from ..plotting.sky import plot_galactic_supernovae_polar_hemispheres
from ..plotting.signals import plot_detector_signal_channels, plot_candidate_signal
from ..plotting.parameters import plot_epoch_sky_parameters, plot_corner
from ..plotting.losses import plot_loss

from ..utils.defaults import Y_LENGTH, HIDDEN_DIM, Z_DIM, BATCH_SIZE, DEVICE, TEN_KPC, VALIDATION_SPLIT 
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
        samples_per_epoch: int = 20000,
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
        val_data_path: str = None  # Path to validation data files (real CVAE val set)
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
                
        Note: If both train_data_path and val_data_path are provided, they take precedence
        over custom_data.
        """
        self.y_length = y_length
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.seed = seed
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.samples_per_epoch = samples_per_epoch
        self.validation_split = validation_split
        self.lr_flow = lr_flow
        self.checkpoint_interval = checkpoint_interval
        self.outdir = outdir
        self.toy = toy
        self.detector_noise_on = detector_noise_on
        self.max_grad_norm = max_grad_norm
        
        # Set default parameters if not provided
        if parameters is None:
            parameters = ["beta1_IC_b", "ra", "dec", "d", "psi"]
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
        # hThetaMulti always produces: [intrinsic_params..., ra, dec, d, psi]
        # We need to extract only the parameters we want in the order they appear
        self.param_extract_indices = []
        n_intrinsic = len(self.intrinsic_params)
        
        # Add indices for intrinsic parameters (they come first in hThetaMulti)
        for i in range(n_intrinsic):
            self.param_extract_indices.append(i)
        
        # Add indices for sky parameters
        # Sky parameters in hThetaMulti are always in order: [ra, dec, d, psi]
        sky_param_order = ["ra", "dec", "d", "psi"]
        for sky_param in self.sky_params:
            if sky_param in sky_param_order:
                sky_idx = sky_param_order.index(sky_param)
                self.param_extract_indices.append(n_intrinsic + sky_idx)
            else:
                raise ValueError(f"Unknown sky parameter '{sky_param}'. Available: {sky_param_order}")
        
        print(f"\n=== Parameter Extraction Setup ===")
        print(f"Requested parameters: {parameters}")
        print(f"Intrinsic params in dataset: {self.intrinsic_params} (indices 0-{n_intrinsic-1})")
        print(f"Sky params in dataset: {self.sky_params} (indices {n_intrinsic}-{len(self.param_extract_indices)-1})")
        print(f"Extract indices from hThetaMulti.parameters: {self.param_extract_indices}")
        print(f"Final flow parameter dimension: {len(self.param_extract_indices)}")
        print(f"{'='*40}\n")


        self.supernovae = Supernovae(
            locations_file='../../exploded_supernovae_t100_sf5.csv',
            rotation_offset=np.deg2rad(60.0),
        )  # locations of Galactic supernovae

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
                shared_min=self.training_dataset.min_theta,
                shared_max=self.training_dataset.max_theta,
                shared_max_strain=self.training_dataset.max_strain,
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
                shared_min=self.training_dataset.min_theta,
                shared_max=self.training_dataset.max_theta,
                shared_max_strain=self.training_dataset.max_strain
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
        self.flow_param_dim = len(self.parameters_to_estimate)
        self.flow_signal_dim = Y_LENGTH * 3 if not self.toy else Y_LENGTH
        self.flow = FlowFCL(dim=self.flow_param_dim, signal_dim=self.flow_signal_dim).to(DEVICE)
        # self.flow = FlowCNN(dim=self.flow_param_dim, signal_dim=self.flow_signal_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.lr_flow, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()

    @staticmethod
    def _denormalize_with_bounds(params_norm: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
        """Denormalize parameters from [-1, 1] using explicit min/max bounds."""
        return (params_norm + 1.0) / 2.0 * (max_vals - min_vals) + min_vals
    
    def _denormalize_extracted_params(self, params_norm: np.ndarray, dataset) -> np.ndarray:
        """Denormalize extracted parameters using appropriate bounds.
        
        The input params_norm is in the reduced parameter space (e.g., 5D if we extracted 5 params).
        The dataset (hThetaMulti) already contains only the extracted parameters,
        so we use its bounds directly without indexing.
        """
        # dataset.min_theta and dataset.max_theta already have only the extracted parameters
        # since they were built from training data with extracted parameters
        min_vals = dataset.min_theta
        max_vals = dataset.max_theta
        
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

    def run_parameter_estimation(self, signal_idx: int = None, d: float = None, ra: float = None, dec: float = None, export_on: bool = False, random_psi: bool = True):
        """Run parameter estimation on a single signal and return the predicted parameters.
        
        Args:
            signal_idx: Index of the signal in the validation dataset
            d: Distance in kpc
            ra: Right ascension in radians (optional, random if None)
            dec: Declination in radians (optional, random if None)
            export_on: Whether to export signal channels as .txt files
            random_psi: Whether to use random polarization angle (True) or fixed psi=0 (False)
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
            filename_suffix = "epoch"
        
        if signal_idx is not None and d is not None:
            # Get RAW (unnormalized) signal and params from validation dataset
            # We need raw signals because hThetaMulti expects unnormalized input
            signal_raw = self.validation_dataset.signals[:, signal_idx:signal_idx+1]  # Raw signal, shape (Y_LENGTH, 1)
            params = self.validation_dataset.parameters[signal_idx]  # Raw params, shape (num_params,)
            
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
                max_strain=self.validation_dataset.max_strain,
                theta=params_tensor,  # Tensor
                min_theta=self.validation_dataset.min_theta,
                max_theta=self.validation_dataset.max_theta,
                ra=sampled_ra,
                dec=sampled_dec,
                d=sampled_d,
                batch_size=self.batch_size,
                detector_noise_on=True,  # Add fresh detector noise, consistent with train()
                random_polarization=random_psi,
                seed=1,
                intrinsic_param_names=self.intrinsic_params
            )
            case = temp_h_theta_multi_val[0]
            # Use temp_h_theta_multi_val for plotting since it was used to create case
            active_h_theta_multi = temp_h_theta_multi_val
        else:
            case = self.h_theta_multi_val[np.random.randint(len(self.h_theta_multi_val))]
            active_h_theta_multi = self.h_theta_multi_val
        
        plot_detector_signal_channels(
            signals=case[0].detach().cpu().numpy() / TEN_KPC,
            noisy_signals=case[1].detach().cpu().numpy() / TEN_KPC,
            max_value=active_h_theta_multi.max_strain,
            detector_labels=active_h_theta_multi.detectors,
            background="black",
            generated=False,
            fname=os.path.join(epoch_data_dir, f"{filename_suffix}_signal.png"),
        )
        self.plot_corner_sampled_signal(
            num_samples=3000,
            n_steps=20,
            fname=os.path.join(epoch_data_dir, f"{filename_suffix}_corner.png"),
            sampled_case=case,
            h_theta_multi_dataset=active_h_theta_multi
        )
        self.plot_sky_localisation_sampled_signal(
            num_samples=3000,
            n_steps=20,
            fname=os.path.join(epoch_data_dir, f"{filename_suffix}_sky.png"),
            sampled_case=case,
            h_theta_multi_dataset=active_h_theta_multi,
        )

        # export each channel of the signal as a separate .txt file for external analysis
        if export_on:
            export_dir = os.path.join(self.outdir, "exported_signals")
            os.makedirs(export_dir, exist_ok=True)
            detector_labels = active_h_theta_multi.detectors
            for i in range(case[0].shape[0]):
                channel_signal = case[0][i].detach().cpu().numpy() / TEN_KPC  # Denormalize to physical units
                detector_name = detector_labels[i] if i < len(detector_labels) else f"channel_{i+1}"
                np.savetxt(os.path.join(export_dir, f"{filename_suffix}_{detector_name}.txt"), channel_signal)


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
                validation=False,
                epoch_dir=os.path.join(self.outdir, "flow_matching", "epoch_data"),
            )
            signals, params = self._sample_dataset_batches(self.training_dataset, self.samples_per_epoch)


            # create multi-channel signals
            self.h_theta_multi_train = hThetaMulti(
                s=signals,
                max_strain=self.training_dataset.max_strain,
                theta=params,
                min_theta=self.training_dataset.min_theta,
                max_theta=self.training_dataset.max_theta,
                ra=sampled_ra,
                dec=sampled_dec,
                d=sampled_d,
                batch_size=self.batch_size,
                detector_noise_on=True,
                random_polarization=True,
                seed=epoch,  # Vary seed by epoch for different psi values each epoch
                intrinsic_param_names=self.intrinsic_params
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
                n_val_signals = int(self.validation_dataset.signals.shape[1])
                val_ra, val_dec, val_d = self.supernovae.sample_supernovae_for_epoch(
                    epoch, 
                    n_val_signals, 
                    self.num_epochs, 
                    exponential=True, 
                    validation=True, 
                    epoch_dir=os.path.join(self.outdir, "flow_matching", "epoch_data")
                )
                signals_val, params_val = self.validation_dataset.signals, self.validation_dataset.parameters # use all the strain data from the validation set
                self.h_theta_multi_val = hThetaMulti(
                    s=signals_val,
                    max_strain=self.validation_dataset.max_strain,
                    theta=params_val,
                    min_theta=self.validation_dataset.min_theta,
                    max_theta=self.validation_dataset.max_theta,
                    ra=val_ra,
                    dec=val_dec,
                    d=val_d,
                    batch_size=self.batch_size,
                    detector_noise_on=True,
                    random_polarization=True,
                    seed=epoch + 1000,  # Different seed range for validation set
                    intrinsic_param_names=self.intrinsic_params
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

            self.run_parameter_estimation(signal_idx=None, d=None, ra=None, dec=None) 

            print(f"Epoch {epoch+1}/{self.num_epochs} | Train MSE Loss: {avg_total_loss:.4f} | Val MSE Loss: {avg_total_loss_val:.4f}")

        runtime = (time.time() - t0) / 60
        print(f"Training Time: {runtime:.2f}min")
        print("Plotting training/validation loss curves...")
        # Optionally: plot final results or save model
        self.save_models()
        self.save_losses()
        self.display_results(fname=os.path.join(self.outdir, "flow_matching", "training_validation_losses.png"))  

    def _plot_project_to_detectors_steps(self, signal_idx, f_name_h, f_name_h_delayed, f_name_h_rescaled_delayed):
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
            max_strain=self.validation_dataset.max_strain,
            theta=params_np,  # Tensor
            min_theta=self.validation_dataset.min_theta,
            max_theta=self.validation_dataset.max_theta,
            ra=sampled_ra,
            dec=sampled_dec,
            d=sampled_d,
            batch_size=self.batch_size,
            detector_noise_on=True,  # Add fresh detector noise, consistent with train()
            random_polarization=True,
            seed=1,
            intrinsic_param_names=self.intrinsic_params
        )

        temp_h_theta_multi._plot_project_to_detectors_steps(
            signal_idx=0,
            f_name_h=f_name_h,
            f_name_h_delayed=f_name_h_delayed,
            f_name_h_delayed_rescaled=f_name_h_rescaled_delayed,
            font_family="Serif",
            font_name="Times New Roman"
        )


    def plot_corner_sampled_signal(
        self,
        epoch: int = 0,
        num_samples: int = 5000,
        n_steps: int = 20,
        fname: str = "plots/corner_plot_sampled_signal.png",
        sampled_case=None,
        h_theta_multi_dataset=None,
    ):
        """Generate a corner plot for one sampled multi-channel validation signal.

        This samples one validation waveform and one sky location (RA/Dec/d), appends
        sky parameters (including psi), and plots the posterior over the full parameter vector.
        """
        self.flow.eval()
        
        # Use passed dataset or default to self.h_theta_multi_val
        if h_theta_multi_dataset is None:
            h_theta_multi_dataset = self.h_theta_multi_val

        t0 = time.time()

        # sample one signal from self.multi_channel_val dataset for corner plotting, ensuring it has the same sky parameters as the sky plot
        signal, noisy_signal, params = sampled_case

        if noisy_signal.dim() == 2:
            noisy_signal = noisy_signal.unsqueeze(0)
        if params.dim() == 1:
            params = params.unsqueeze(0)

        noisy_signal = noisy_signal.view(noisy_signal.size(0), -1).to(DEVICE).float()
        params = params.view(params.size(0), -1).to(DEVICE).float()

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

        # Denormalize extracted parameters
        if self.toy:
            # For toy data, we work in full parameter space so denormalize directly
            # (toy parameters are already in the format we need)
            samples_cpu = samples_cpu  # Already denormalized in toy case
            true_params = true_params_norm
            labels = self.parameters_to_estimate
        else:
            # For real data, always use extracted parameter denormalization
            # since we're operating in the extracted parameter space
            samples_cpu = self._denormalize_extracted_params(samples_cpu, h_theta_multi_dataset)
            true_params = self._denormalize_extracted_params(true_params_norm.reshape(1, -1), h_theta_multi_dataset).flatten()
            labels = self.parameters_to_estimate

        t1 = time.time()
        print(f"Corner plot sampling and denormalisation took {(t1 - t0):.2f}s")

        # Convert parameter names to LaTeX labels using plotting_defaults
        latex_labels = [PARAMETER_LABELS.get(param, param) for param in self.parameters_to_estimate]
        
        # Calculate axis ranges from extracted parameter bounds
        # Use the dataset bounds which are already in extracted parameter space
        mins = h_theta_multi_dataset.min_theta
        maxs = h_theta_multi_dataset.max_theta
        span = np.maximum(maxs - mins, 1e-8)
        pad = 0.03 * span
        ranges = [
            (float(mins[i] - pad[i]), float(maxs[i] + pad[i]))
            for i in range(len(mins))
        ]
        
        # Debug: print ranges for each parameter
        print("\nPlot axis ranges (extracted parameter space):")
        for i, label in enumerate(self.parameters_to_estimate):
            print(f"  {label:20s}: {ranges[i]}")

        # manually set limits on d if it's in the extracted parameters
        if 'd' in self.parameters_to_estimate:
            d_idx = self.parameters_to_estimate.index('d')
            ranges[d_idx] = (0.1, 20.0)
            print(f"  Override d range to: {ranges[d_idx]}")

        plot_corner(
            samples_cpu=samples_cpu,
            true_params=true_params,
            fname=fname,
            labels=latex_labels,
            ranges=ranges
        )

    def plot_sky_localisation_sampled_signal(
        self,
        epoch: int = 0,
        num_samples: int = 5000,
        n_steps: int = 20,
        fname: str = "plots/sky_localisation_sampled_signal.png",
        sampled_case=None,
        h_theta_multi_dataset=None,
    ):
        """Generate a sky-localisation (RA/Dec) posterior plot for one sampled signal."""
        self.flow.eval()
        
        # Use passed dataset or default to self.h_theta_multi_val
        if h_theta_multi_dataset is None:
            h_theta_multi_dataset = self.h_theta_multi_val

        if sampled_case is None:
            _, noisy_signal, params = self._sample_validation_plot_case(epoch)
        else:
            _, noisy_signal, params = sampled_case

        if noisy_signal.dim() == 2:
            noisy_signal = noisy_signal.unsqueeze(0)
        if params.dim() == 1:
            params = params.unsqueeze(0)

        noisy_signal = noisy_signal.view(noisy_signal.size(0), -1).to(DEVICE).float()
        params = params.view(params.size(0), -1).to(DEVICE).float()

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

        # Denormalize the extracted parameters
        if self.toy:
            # For toy data, assume last 4 params are [ra, dec, d, psi]
            samples_denorm = samples_cpu
            true_params_denorm = true_params_norm
        else:
            # For real data, denormalize in the extracted parameter space
            samples_denorm = self._denormalize_extracted_params(samples_cpu, h_theta_multi_dataset)
            true_params_denorm = self._denormalize_extracted_params(true_params_norm.reshape(1, -1), h_theta_multi_dataset).flatten()
        
        # Extract RA and Dec indices from the parameters_to_estimate list
        ra_idx = self._get_extracted_index("ra")
        dec_idx = self._get_extracted_index("dec")
        
        if ra_idx >= 0 and dec_idx >= 0:
            # Extract RA and Dec from the denormalized extracted parameters
            ra_samples = samples_denorm[:, ra_idx]
            dec_samples = samples_denorm[:, dec_idx]
            true_ra = true_params_denorm[ra_idx]
            true_dec = true_params_denorm[dec_idx]
        else:
            # Fallback: assume they are at the end (shouldn't happen with proper setup)
            ra_samples = samples_denorm[:, -4]
            dec_samples = samples_denorm[:, -3]
            true_ra = true_params_denorm[-4]
            true_dec = true_params_denorm[-3]

        plot_galactic_supernovae_polar_hemispheres(
            ccsn=self.supernovae,
            fname=fname,
            posterior_ra_samples=ra_samples,
            posterior_dec_samples=dec_samples,
            true_ra_override=true_ra,
            true_dec_override=true_dec,
            show_constellation_borders=True,
            show_important_constellation_labels=True,
            show=False,
            background="black",
            font_family="sans-serif",
            font_name="Avenir",
            red_blob_mode="density_peak",
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
            max_value=self.val_loader.dataset.max_strain,
            background=background,
            fname=fname
        )

    def _extract_params_to_estimate(self, full_params: torch.Tensor) -> torch.Tensor:
        """Extract only the parameters we're estimating from the full parameter tensor.
        
        For toy data, returns all parameters. For real data, extracts only requested parameters
        using the parameter_mapping indices.
        """
        if self.toy:
            return full_params
        
        # Extract only the requested parameter indices
        return full_params[:, self.param_extract_indices]
    
    def display_results(self, background="black", fname=None):
        """Display training results."""
        plot_loss(self.avg_mse_losses, self.avg_mse_losses_val, background=background, fname=fname)        
        
        # Plot Flow gradient norms if available
        if hasattr(self, 'flow_gradient_norms') and len(self.flow_gradient_norms) > 0:
            print("\nPlotting Flow Gradient Norms...")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.flow_gradient_norms, label='Flow Gradient Norm', color='#9b59b6', linewidth=2)
            ax.set_xlabel('Epoch', size=16)
            ax.set_ylabel('Gradient Norm', size=16)
            ax.set_title('Flow Gradient Norms During Training', size=18)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Clipping Threshold')
            
            # Set background color
            ax.set_facecolor(background)
            fig.patch.set_facecolor(background)
            ax.spines['bottom'].set_color('white' if background == 'black' else 'black')
            ax.spines['left'].set_color('white' if background == 'black' else 'black')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(colors='white' if background == 'black' else 'black')
            
            plt.tight_layout()
            
            # Save if fname provided
            if fname:
                grad_fname = fname.replace('.png', '_gradient_norms.png')
                plt.savefig(grad_fname, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
                print(f"Saved gradient norms plot to {grad_fname}")
            
            plt.show()
            plt.close()
        
    @property
    def save_fname(self):
        return f"{self.outdir}/flow_weights.pt"

    def save_models(self):
        torch.save(self.flow.state_dict(), self.save_fname)
        print(f"Saved Flow model to {self.save_fname}")
    
    def save_losses(self, fname: str = None):
        """Save training and validation losses to a CSV file.
        
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
            max_strain=self.validation_dataset.max_strain,
            theta=params_tensor,
            min_theta=self.validation_dataset.min_theta,
            max_theta=self.validation_dataset.max_theta,
            ra=sampled_ra,
            dec=sampled_dec,
            d=sampled_d,
            batch_size=self.batch_size,
            detector_noise_on=True,
            random_polarization=True,
            seed=1,
            intrinsic_param_names=self.intrinsic_params
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