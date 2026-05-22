import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from ..data.s_theta import sTheta
from ..data.h_theta_multi import hThetaMulti
from ..supernovae.supernovae import Supernovae
from tqdm.auto import trange

from ..plotting import (
    plot_corner,
    plot_galactic_supernovae_polar_hemispheres,
)
from ..plotting.signals import plot_detector_signal_channels, plot_candidate_signal
from ..plotting.parameters import plot_epoch_sky_parameters

from ..utils.defaults import Y_LENGTH, HIDDEN_DIM, Z_DIM, BATCH_SIZE, DEVICE, TEN_KPC, VALIDATION_SPLIT
from ..nn.flow_multi import FlowFCL, FlowCNN

from . import create_train_val_split, display_results_method

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

    def _sample_sky_params_for_epoch(
        self,
        epoch: int,
        n_samples: int,
        exponential: bool = True,
        validation: bool = False,
        epoch_dir: str = None,
    ):
        """Sample RA/Dec/d sky parameters for an epoch distance shell.

        When ``exponential`` is True, samples are weighted to favor larger
        distances in the shell (near ``max_kiloparsec``), with the weighting
        strength increasing over epochs.
        """
        min_kiloparsec = 0.0
        max_kiloparsec = min(20.0, (epoch / self.num_epochs) * 20.0 + 1.0)
        distance_mask = (
            (self.supernovae.distances >= min_kiloparsec)
            & (self.supernovae.distances <= max_kiloparsec)
        )
        candidate_indices = np.where(distance_mask)[0]
        if candidate_indices.size == 0:
            raise ValueError(
                f"No supernovae found in [{min_kiloparsec:.3f}, {max_kiloparsec:.3f}] kpc range."
            )

        sample_probs = None
        if exponential:
            candidate_distances = self.supernovae.distances[candidate_indices]
            shell_width = max(max_kiloparsec - min_kiloparsec, 1e-8)
            normalized_distance = np.clip((candidate_distances - min_kiloparsec) / shell_width, 0.0, 1.0)

            # Increase bias through training so later epochs concentrate more strongly
            # near the far edge of each shell.
            epoch_fraction = (epoch + 1) / max(self.num_epochs, 1)
            growth = 1.0 + 7.0 * epoch_fraction
            weights = np.exp(growth * normalized_distance)
            weight_sum = np.sum(weights)
            if np.isfinite(weight_sum) and weight_sum > 0.0:
                sample_probs = weights / weight_sum

        sampled_indices = np.random.choice(
            candidate_indices,
            size=n_samples,
            replace=candidate_indices.size < n_samples,
            p=sample_probs,
        )
        if epoch_dir is not None:
            os.makedirs(epoch_dir, exist_ok=True)
            if validation == True:
                filename_suffix = "validation"
            else:
                filename_suffix = "training"
            self.supernovae.plot_galactic_distribution(
                fname_xy=os.path.join(epoch_dir, f"epoch_{epoch + 1:04d}_{filename_suffix}_galactic_xy.png"),
                background="black",
                transparent=False,
                light_year=False,
                highlight_indices=sampled_indices,
                show=False,
                dpi=150,
            )
        sampled_sky_params = self.supernovae.get_sky_params(indices=sampled_indices)

        return sampled_sky_params[:, 0], sampled_sky_params[:, 1], sampled_sky_params[:, 2]

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

    def train(self):
        t0 = time.time()

        self.avg_mse_losses = []
        self.avg_mse_losses_val = []

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

            sampled_ra, sampled_dec, sampled_d = self._sample_sky_params_for_epoch(
                epoch,
                self.samples_per_epoch,
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
                torch.nn.utils.clip_grad_norm_(self.flow.parameters(), self.max_grad_norm)
                
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
                val_ra, val_dec, val_d = self._sample_sky_params_for_epoch(epoch, n_val_signals, exponential=True, validation=True, epoch_dir=os.path.join(self.outdir, "flow_matching", "epoch_data"))
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

            plot_case = self.h_theta_multi_val[np.random.randint(len(self.h_theta_multi_val))] # random sample
            # plot_case = self.h_theta_multi_val[100] # first sample for consistency across epochs

            # snr_case = self.h_theta_multi_train.calculate_snr_from_fft(idx=100) 
            # print("snr = ", snr_case)
            print(f"Plotting corner and sky localisation for epoch {epoch + 1} using validation sample with parameters: {plot_case[2].cpu().numpy()}")
            plot_detector_signal_channels(
                signals=plot_case[0].detach().cpu().numpy() / TEN_KPC,
                noisy_signals=plot_case[1].detach().cpu().numpy() / TEN_KPC,
                max_value=self.h_theta_multi_val.max_strain,
                detector_labels=self.h_theta_multi_val.detectors,
                background="black",
                generated=False,
            )
            self.plot_corner_sampled_signal(
                num_samples=3000,
                n_steps=20,
                fname=os.path.join(corner_epoch_dir, f"epoch_{epoch + 1:04d}_corner.png"),
                sampled_case=plot_case,
            )
            self.plot_sky_localisation_sampled_signal(
                num_samples=3000,
                n_steps=20,
                fname=os.path.join(corner_epoch_dir, f"epoch_{epoch + 1:04d}_sky.png"),
                sampled_case=plot_case,
            )

            print(f"Epoch {epoch+1}/{self.num_epochs} | Train MSE Loss: {avg_total_loss:.4f} | Val MSE Loss: {avg_total_loss_val:.4f}")

        runtime = (time.time() - t0) / 60
        print(f"Training Time: {runtime:.2f}min")
        print("Plotting training/validation loss curves...")
        display_results_method(self.avg_mse_losses, self.avg_mse_losses_val, background="black")
        # Optionally: plot final results or save model
        # self.save_models()

    def plot_corner_sampled_signal(
        self,
        epoch: int = 0,
        num_samples: int = 5000,
        n_steps: int = 20,
        fname: str = "plots/corner_plot_sampled_signal.png",
        sampled_case=None,
    ):
        """Generate a corner plot for one sampled multi-channel validation signal.

        This samples one validation waveform and one sky location (RA/Dec/d), appends
        sky parameters (including psi), and plots the posterior over the full parameter vector.
        """
        self.flow.eval()

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
            samples_cpu = self._denormalize_extracted_params(samples_cpu, self.h_theta_multi_val)
            true_params = self._denormalize_extracted_params(true_params_norm.reshape(1, -1), self.h_theta_multi_val).flatten()
            labels = self.parameters_to_estimate

        t1 = time.time()
        print(f"Corner plot sampling and denormalisation took {(t1 - t0):.2f}s")

        # Calculate axis ranges from extracted parameter bounds
        # Use the dataset bounds which are already in extracted parameter space
        mins = self.h_theta_multi_val.min_theta
        maxs = self.h_theta_multi_val.max_theta
        span = np.maximum(maxs - mins, 1e-8)
        pad = 0.03 * span
        ranges = [
            (float(mins[i] - pad[i]), float(maxs[i] + pad[i]))
            for i in range(len(mins))
        ]
        
        # Debug: print ranges for each parameter
        print("\nPlot axis ranges (extracted parameter space):")
        for i, label in enumerate(labels):
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
            labels=labels,
            ranges=ranges
        )

    def plot_sky_localisation_sampled_signal(
        self,
        epoch: int = 0,
        num_samples: int = 5000,
        n_steps: int = 20,
        fname: str = "plots/sky_localisation_sampled_signal.png",
        sampled_case=None,
    ):
        """Generate a sky-localisation (RA/Dec) posterior plot for one sampled signal."""
        self.flow.eval()

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
            samples_denorm = self._denormalize_extracted_params(samples_cpu, self.h_theta_multi_val)
            true_params_denorm = self._denormalize_extracted_params(true_params_norm.reshape(1, -1), self.h_theta_multi_val).flatten()
        
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
    
    def display_results(self, background="black"):
        """Display training results."""
        display_results_method(self.avg_mse_losses, self.avg_mse_losses_val, background=background)
        
        # # Plot VAE gradient norms if available
        # if hasattr(self, 'vae_gradient_norms'):
        #     if len(self.vae_gradient_norms) > 0:
        #         print("\nPlotting VAE Gradient Norms...")
        #         fig, ax = plt.subplots(figsize=(10, 6))
        #         ax.plot(self.vae_gradient_norms, label='VAE Gradient Norm', color='#3498db', linewidth=2)
        #         ax.set_xlabel('Epoch', size=16)
        #         ax.set_ylabel('Gradient Norm', size=16)
        #         ax.set_title('VAE Gradient Norms During Training', size=18)
        #         ax.legend(fontsize=12)
        #         ax.grid(True, alpha=0.3)
        #         ax.axhline(y=self.max_grad_norm, color='red', linestyle='--', alpha=0.5, label=f'Clipping Threshold ({self.max_grad_norm})')
        #         ax.legend(fontsize=12)
        #         plt.tight_layout()
        #         plt.show()
        
        # # Plot Flow NLL losses if available
        # if hasattr(self, 'flow_train_nll_losses') and hasattr(self, 'flow_val_nll_losses'):
        #     if len(self.flow_train_nll_losses) > 0:
        #         print("\nPlotting Flow NLL Losses...")
        #         plot_loss(
        #             train_losses=self.flow_train_nll_losses, 
        #             val_losses=self.flow_val_nll_losses,
        #             background="black",
        #             fname="plots/flow_loss_curve.svg"
        #         )
        
        # # Plot Flow gradient norms if available
        # if hasattr(self, 'flow_gradient_norms'):
        #     if len(self.flow_gradient_norms) > 0:
        #         print("\nPlotting Flow Gradient Norms...")
        #         fig, ax = plt.subplots(figsize=(10, 6))
        #         ax.plot(self.flow_gradient_norms, label='Flow Gradient Norm', color='#9b59b6', linewidth=2)
        #         ax.set_xlabel('Epoch', size=16)
        #         ax.set_ylabel('Gradient Norm', size=16)
        #         ax.set_title('Flow Gradient Norms During Training', size=18)
        #         ax.legend(fontsize=12)
        #         ax.grid(True, alpha=0.3)
        #         ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Clipping Threshold')
        #         plt.tight_layout()
        #         plt.show()
        
    @property
    def save_fname(self):
        return f"{self.outdir}/generator_weights.pt"

    def save_models(self):
        torch.save(self.vae.state_dict(), self.save_fname)
        print(f"Saved VAE model to {self.save_fname}")