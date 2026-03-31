import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from ..data.s_theta import sTheta
from ..data.h_theta_multi import hThetaMulti
from ..localisation.supernovae import Supernovae
from ..localisation.supernovae import Supernovae
from tqdm.auto import trange

from ..plotting import (
    plot_corner,
    plot_galactic_supernovae_polar_hemispheres,
)

from ..utils.defaults import Y_LENGTH, HIDDEN_DIM, Z_DIM, BATCH_SIZE, DEVICE, SAMPLING_RATE
from ..nn.flow_multi import Flow

from . import create_train_val_split, plot_candidate_signal_method, display_results_method

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
        samples_per_epoch: int = 30000,
        validation_split: float = 0.1,
        lr_flow: float = 5e-4,
        checkpoint_interval: int = 16,
        outdir: str = "outdir",
        noise: bool = True,
        curriculum: bool = True,
        toy: bool = True,
        max_grad_norm: float = 1.0,  # Maximum gradient norm for clipping
        start_snr: int = 100,
        end_snr: int = 10,
        noise_realizations: int = 1,  # Number of noise realizations per signal
        multi_param: bool = True,
        include_beta: bool = True,
        estimate_intrinsic_params: bool = False,
        custom_data: tuple = None,  # (signals_array, params_array) for generated data
        train_data_path: str = None,  # Path to training data files (generated signals)
        val_data_path: str = None  # Path to validation data files (real CVAE val set)
    ):
        """Initialize FlowMatchingTrainerMulti.
        
        Args:
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
        self.noise = noise
        self.curriculum = curriculum
        self.max_grad_norm = max_grad_norm
        self.start_snr = start_snr
        self.end_snr = end_snr
        self.noise_realizations = noise_realizations
        self.multi_param = multi_param
        self.include_beta = include_beta
        self.estimate_intrinsic_params = estimate_intrinsic_params
        self.sky_param_dim = 4


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
            self.training_dataset = sTheta(
                custom_data=(train_signals, train_params),
                noise=noise,
                curriculum=curriculum,
                num_epochs=num_epochs,
                start_snr=start_snr,
                end_snr=end_snr,
                noise_realizations=noise_realizations,
                batch_size=batch_size,
                multi_param=multi_param,
                include_beta=include_beta,
                generated=True
            )
            
            # Create validation dataset sharing normalization from training
            self.validation_dataset = sTheta(
                custom_data=(val_signals, val_params),
                noise=noise,
                curriculum=False,  # No curriculum for validation
                num_epochs=num_epochs,
                start_snr=end_snr,
                end_snr=end_snr,
                noise_realizations=1,  # Single realization for validation
                batch_size=batch_size,
                multi_param=multi_param,
                include_beta=include_beta,
                shared_min=self.training_dataset.min_theta,
                shared_max=self.training_dataset.max_theta,
                shared_max_strain=self.training_dataset.max_strain,
                generated=True
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
            self.training_dataset = sTheta(
                custom_data=(signals_array[:, train_indices], params_array[train_indices]),
                noise=noise,
                curriculum=curriculum,
                num_epochs=num_epochs,
                start_snr=start_snr,
                end_snr=end_snr,
                noise_realizations=noise_realizations,
                batch_size=batch_size,
                multi_param=multi_param,
                include_beta=include_beta,
            )
            
            # Create validation dataset with custom data
            self.validation_dataset = sTheta(
                custom_data=(signals_array[:, val_indices], params_array[val_indices]),
                noise=noise,
                curriculum=False,  # No curriculum for validation
                num_epochs=num_epochs,
                start_snr=end_snr,
                end_snr=end_snr,
                noise_realizations=1,  # Single realization for validation
                batch_size=batch_size,
                multi_param=multi_param,
                include_beta=include_beta,
                shared_min=self.training_dataset.min_theta,
                shared_max=self.training_dataset.max_theta,
                shared_max_strain=self.training_dataset.max_strain
            )
        else:
            # Use standard train/val split. Probably the most acceptable
            self.training_dataset, self.validation_dataset, self.val_indices = create_train_val_split(
                toy=self.toy,
                y_length=self.y_length,
                noise=self.noise,
                validation_split=self.validation_split,
                seed=self.seed,
                num_epochs=self.num_epochs,
                start_snr=start_snr,
                end_snr=end_snr,
                curriculum=self.curriculum,
                noise_realizations=self.noise_realizations,
                multi_param=self.multi_param,
                include_beta=self.include_beta,
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
        # Multi-detector setup appends [ra, dec, d, polar_angle] to theta.
        # By default we estimate only sky parameters while keeping intrinsic params in the dataset.
        base_param_dim = self.training_dataset.parameters.shape[1]
        if self.toy:
            self.flow_param_dim = base_param_dim
        elif self.estimate_intrinsic_params:
            self.flow_param_dim = base_param_dim + self.sky_param_dim
        else:
            self.flow_param_dim = self.sky_param_dim
        self.flow_signal_dim = Y_LENGTH * 3 if not self.toy else Y_LENGTH
        self.flow = Flow(dim=self.flow_param_dim, signal_dim=self.flow_signal_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.lr_flow, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()

    @staticmethod
    def _denormalize_with_bounds(params_norm: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
        """Denormalize parameters from [-1, 1] using explicit min/max bounds."""
        return (params_norm + 1.0) / 2.0 * (max_vals - min_vals) + min_vals

    def _save_epoch_data_plots(self, epoch: int) -> None:
        """Save signal and parameter snapshots for the current epoch."""
        if not hasattr(self, "h_theta_multi"):
            return

        epoch_dir = os.path.join(self.outdir, "flow_matching", "epoch_data")
        os.makedirs(epoch_dir, exist_ok=True)

        # Plot 1: first generated multi-channel signal for this epoch.
        signals = self.h_theta_multi.multi_channel_signals
        time_axis = np.arange(Y_LENGTH) * SAMPLING_RATE
        fig_sig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        detector_labels = getattr(self.h_theta_multi, "detectors", ["H1", "L1", "V1"])
        first_signal = signals[0]

        for i, ax in enumerate(axes):
            ax.plot(time_axis, first_signal[i], lw=1.5)
            ax.set_ylabel("h")
            ax.set_title(f"{detector_labels[i]}")
            ax.grid(alpha=0.2)

        axes[-1].set_xlabel("time (s)")
        fig_sig.suptitle(f"Epoch {epoch + 1}: First Multi-Channel Sample")
        fig_sig.tight_layout()
        fig_sig.savefig(
            os.path.join(epoch_dir, f"epoch_{epoch + 1:04d}_signals.png"),
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig_sig)

        # Plot 2: distribution snapshots for the last four params [ra, dec, d, psi].
        params = self.h_theta_multi.parameters
        fig_par, axes = plt.subplots(2, 2, figsize=(10, 8))
        param_labels = ["ra", "dec", "d", "psi"]

        for i, ax in enumerate(axes.flatten()):
            ax.hist(params[:, -4 + i], bins=40, alpha=0.85)
            ax.set_title(param_labels[i])
            ax.grid(alpha=0.2)

        fig_par.suptitle(f"Epoch {epoch + 1}: Sky-Parameter Distributions")
        fig_par.tight_layout()
        fig_par.savefig(
            os.path.join(epoch_dir, f"epoch_{epoch + 1:04d}_params.png"),
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig_par)

    def _sample_sky_params_for_epoch(self, epoch: int, n_samples: int):
        """Sample RA/Dec/d sky parameters for an epoch distance shell."""
        min_kiloparsec = epoch / self.num_epochs * 20.0
        max_kiloparsec = min_kiloparsec + 1.0
        distance_mask = (
            (self.supernovae.distances >= min_kiloparsec)
            & (self.supernovae.distances <= max_kiloparsec)
        )
        candidate_indices = np.where(distance_mask)[0]
        if candidate_indices.size == 0:
            raise ValueError(
                f"No supernovae found in [{min_kiloparsec:.3f}, {max_kiloparsec:.3f}] kpc range."
            )
        sampled_indices = np.random.choice(
            candidate_indices,
            size=n_samples,
            replace=candidate_indices.size < n_samples,
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
                noise=False
            )
            self.h_theta_multi_train_loader = DataLoader(
                self.h_theta_multi_train,
                shuffle=True,
                **loader_kwargs,
            )
            # self._save_epoch_data_plots(epoch)

            for signal, noisy_signal, params in self.h_theta_multi_train_loader:
                signal = signal.view(signal.size(0), -1).to(DEVICE, non_blocking=use_cuda)
                noisy_signal = noisy_signal.view(noisy_signal.size(0), -1).to(DEVICE, non_blocking=use_cuda)
                params = params.view(params.size(0), -1).to(DEVICE, non_blocking=use_cuda)
                if self.toy or self.estimate_intrinsic_params:
                    params_target = params
                else:
                    params_target = params[:, -self.sky_param_dim:]

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
                val_ra, val_dec, val_d = self._sample_sky_params_for_epoch(epoch, n_val_signals)
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
                    noise=False,
                )
                self.h_theta_multi_val_loader = DataLoader(
                    self.h_theta_multi_val,
                    shuffle=False,
                    **loader_kwargs,
                )


                for val_signal, val_noisy_signal, val_params in self.h_theta_multi_val_loader:
                    val_noisy_signal = val_noisy_signal.view(val_noisy_signal.size(0), -1).to(DEVICE, non_blocking=use_cuda)
                    val_params = val_params.view(val_params.size(0), -1).to(DEVICE, non_blocking=use_cuda)
                    if self.toy or self.estimate_intrinsic_params:
                        val_params_target = val_params
                    else:
                        val_params_target = val_params[:, -self.sky_param_dim:]

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

            # Use a single sampled validation case so corner and sky plots compare
            # against the exact same truth values (including beta when enabled).
            plot_case = self.h_theta_multi_val[0]  # First sample from validation set for consistent plotting
            print(f"Plotting corner and sky localisation for epoch {epoch + 1} using validation sample with parameters: {plot_case[2].cpu().numpy()}")
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


        samples_cpu = self.h_theta_multi_val.denormalize_parameters(samples_cpu)
        true_params = self.h_theta_multi_val.denormalize_parameters(true_params_norm.reshape(1, -1)).flatten()


        labels = ["beta", "ra", "dec", "d", "psi"] if self.include_beta else ["ra", "dec", "d", "psi"]

        # Calculate axis ranges from unnormalized dataset values
        mins = np.min(self.h_theta_multi_train.parameters, axis=0)
        maxs = np.max(self.h_theta_multi_train.parameters, axis=0)
        span = np.maximum(maxs - mins, 1e-8)
        pad = 0.03 * span
        ranges = [
            (float(mins[i] - pad[i]), float(maxs[i] + pad[i]))
            for i in range(samples_cpu.shape[1])
        ]

        # manually set limits on d
        ranges[-2] = (0.1, 20.0)

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

        if self.toy or self.estimate_intrinsic_params:
            # Full-vector denormalization; sky parameters remain the trailing coordinates.
            samples_cpu = self.h_theta_multi_val.denormalize_parameters(samples_cpu)
            true_params = self.h_theta_multi_val.denormalize_parameters(true_params_norm.reshape(1, -1)).flatten()
            ra_samples = samples_cpu[:, -4]
            dec_samples = samples_cpu[:, -3]
            true_ra = true_params[-4]
            true_dec = true_params[-3]
        else:
            sky_min = self.h_theta_multi_val.min_theta[-self.sky_param_dim:]
            sky_max = self.h_theta_multi_val.max_theta[-self.sky_param_dim:]
            sky_samples = self._denormalize_with_bounds(samples_cpu, sky_min, sky_max)
            true_sky = self._denormalize_with_bounds(
                true_params_norm[-self.sky_param_dim:].reshape(1, -1),
                sky_min,
                sky_max,
            ).flatten()
            ra_samples = sky_samples[:, 0]
            dec_samples = sky_samples[:, 1]
            true_ra = true_sky[0]
            true_dec = true_sky[1]

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
        plot_candidate_signal_method(
            val_loader=self.val_loader,
            snr=snr,
            background=background,
            index=index,
            fname=fname
        )

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