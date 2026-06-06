import os
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import trange

from ..nn.cvae import ConditionalVAE

from ..utils.defaults import TEN_KPC, Y_LENGTH, HIDDEN_DIM, Z_DIM, BATCH_SIZE, DEVICE

from . import create_train_val_split, plot_signal_grid

from ..plotting.signals import plot_reconstruction, plot_candidate_signal
from ..plotting.latent import plot_latent_space_2d_3d

def _set_seed(seed: int):
    """Set the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return seed

class ConditionalVAETrainer:
    """Trainer for Conditional VAE (CVAE) that conditions on physical parameters."""
    
    def __init__(
        self,
        y_length: int = Y_LENGTH,
        hidden_dim: int = HIDDEN_DIM,
        z_dim: int = Z_DIM,
        seed: int = 99,
        batch_size: int = BATCH_SIZE,
        num_epochs: int = 256,
        validation_split: float = 0.1,
        lr_flow: float = 5e-4,
        checkpoint_interval: int = 16,
        outdir: str = "outdir",
        detector_noise_on: bool = True,
        curriculum: bool = True,
        toy: bool = True,
        max_grad_norm: float = 1.0,
        varying_param_index: int = 0,
        theta_label: Optional[str] = None,
        theta_param_index: Optional[int] = None
    ):
        self.y_length = y_length
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.seed = seed
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.validation_split = validation_split
        self.lr_flow = lr_flow
        self.checkpoint_interval = checkpoint_interval
        self.outdir = outdir
        self.toy = toy
        self.detector_noise_on = detector_noise_on
        self.curriculum = curriculum
        self.max_grad_norm = max_grad_norm
        self.varying_param_index = varying_param_index
        self.theta_label = theta_label
        # Use provided theta_param_index, or default to varying_param_index for display
        self.theta_param_index = theta_param_index if theta_param_index is not None else varying_param_index
        self.device = DEVICE

        # Create train/val split using shared utility function
        self.training_dataset, self.validation_dataset, self.val_indices = create_train_val_split(
            toy=self.toy,
            y_length=self.y_length,
            detector_noise_on=self.detector_noise_on,
            validation_split=self.validation_split,
            seed=self.seed,
            num_epochs=self.num_epochs
        )

        # Get parameter dimension from dataset
        self.param_dim = self.training_dataset.parameters.shape[1]
        print(f"\nParameter dimension: {self.param_dim}")
        print(f"Parameter names: {self.training_dataset.parameter_names if hasattr(self.training_dataset, 'parameter_names') else 'N/A'}")

        # Create DataLoaders
        self.train_loader = DataLoader(
            self.training_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.validation_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )

        print(f"\n=== Dataset Sizes ===")
        print(f"Training samples: {len(self.training_dataset)}")
        print(f"Validation samples: {len(self.validation_dataset)}")
        print("=" * 50)

        os.makedirs(outdir, exist_ok=True)
        os.makedirs(os.path.join(outdir, "cvae"), exist_ok=True)
        _set_seed(self.seed)

        # Initialize Conditional VAE
        self.cvae = ConditionalVAE(
            y_length=self.y_length,
            hidden_dim=self.hidden_dim,
            z_dim=self.z_dim,
            param_dim=self.param_dim
        ).to(DEVICE)
        
        self.optimizerCVAE = optim.Adam(self.cvae.parameters(), lr=self.lr_flow, weight_decay=1e-5)

        # Fixed noise and parameters for visualization in grid format
        # Grid: 4 rows (different params) × 4 cols (different noise) = 16 samples
        num_rows = 4  # Different parameter sets
        num_cols = 4  # Different noise samples
        num_fixed_samples = num_rows * num_cols
        
        # Create noise samples for columns (repeat across rows)
        noise_samples = torch.randn(num_cols, self.z_dim)  # 4 unique noise samples
        self.fixed_noise = noise_samples.repeat(num_rows, 1).to(DEVICE)  # Shape: (16, z_dim)
        
        # Define parameter sets for rows (already in normalized space [-1, 1])
        # Varying parameter (index specified by self.varying_param_index) goes from -1 to 1, others are 0
        param_sets_norm = []
        for i in range(num_rows):
            varying_value = -1.0 + (2.0 * i / (num_rows - 1))  # Linspace: -1, -0.33, 0.33, 1
            params = [0.0] * self.param_dim
            params[self.varying_param_index] = varying_value
            param_sets_norm.append(np.array(params))

        print(param_sets_norm)
        
        # Create parameter tensor where each row has same params (repeated across columns)
        # Shape: (16, param_dim) where rows 0-3 have params[0], rows 4-7 have params[1], etc.
        params_list = []
        for param_norm in param_sets_norm:
            params_list.extend([param_norm] * num_cols)  # Repeat each param set num_cols times
        
        self.fixed_params = torch.tensor(
            np.array(params_list),
            dtype=torch.float32
        ).to(DEVICE)

    def loss_function_cvae(self, y, y_hat, mean, log_var):
        """Compute CVAE loss (same as VAE: reconstruction + KL divergence).
        
        Uses standard KL[q(z|x,y) || p(z)] with isotropic Gaussian prior.
        """
        # MSE reconstruction loss
        reproduction_loss = nn.functional.mse_loss(y_hat, y, reduction='sum')
        reproduction_loss *= 1 * y.shape[1]

        # KL Divergence loss with beta for β-VAE
        kld_beta = 1.0  # Standard VAE
        kld_loss = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        kld_loss = kld_loss * kld_beta

        # Total loss
        total_loss = reproduction_loss + kld_loss

        return total_loss, reproduction_loss, kld_loss

    def train(self):
        """Train the Conditional VAE."""
        # DIAGNOSTICS: Print dataset and parameter information
        print("\n" + "=" * 60)
        print("DIAGNOSTIC: Dataset Information")
        print("=" * 60)
        print(f"Max strain in dataset: {self.training_dataset.max_strain:.6e}")
        sample_signal = self.training_dataset.signals[0]
        print(f"Sample raw signal range: [{sample_signal.min():.6e}, {sample_signal.max():.6e}]")
        
        print(f"\nParameter ranges in dataset:")
        for i in range(self.param_dim):
            p_min = self.training_dataset.parameters[:, i].min()
            p_max = self.training_dataset.parameters[:, i].max()
            param_name = self.training_dataset.parameter_names[i] if hasattr(self.training_dataset, 'parameter_names') else f"Param {i}"
            print(f"  {param_name}: [{p_min:.4f}, {p_max:.4f}]")
            
            # Print unique values for Ye parameter if present
            if 'Ye_c' in param_name or 'ye' in param_name.lower():
                unique_vals = np.unique(np.round(self.training_dataset.parameters[:, i], 4))
                print(f"    Unique {param_name} values: {unique_vals}")
        
        # Test normalization
        print(f"\nTesting parameter normalization:")
        test_param_min = np.array([self.training_dataset.parameters[:, i].min() for i in range(self.param_dim)])
        test_param_max = np.array([self.training_dataset.parameters[:, i].max() for i in range(self.param_dim)])
        
        print(f"  Raw param (all min): {test_param_min}")
        print(f"  Normalized: {self.training_dataset.normalize_parameters(test_param_min)}")
        print(f"  Raw param (all max): {test_param_max}")
        print(f"  Normalized: {self.training_dataset.normalize_parameters(test_param_max)}")
        
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60 + "\n")

        t0 = time.time()

        self.avg_total_losses = []
        self.avg_reproduction_losses = []
        self.avg_kld_losses = []
        self.avg_total_losses_val = []
        self.avg_reproduction_losses_val = []
        self.avg_kld_losses_val = []

        for epoch in trange(self.num_epochs, desc="Epochs", position=0, leave=True):
            self.cvae.train()
            total_loss = 0
            reproduction_loss = 0
            kld_loss = 0
            total_samples = 0

            # self.val_loader.dataset.set_epoch(epoch)
            # self.train_loader.dataset.set_epoch(epoch)

            for signal, noisy_signal, params in self.train_loader:
                signal = signal.view(signal.size(0), -1).to(DEVICE)
                noisy_signal = noisy_signal.view(noisy_signal.size(0), -1).to(DEVICE)
                params = params.view(params.size(0), -1).to(DEVICE)

                self.optimizerCVAE.zero_grad()
                
                # Forward pass through CVAE with conditioning
                recon, mean, log_var = self.cvae(noisy_signal, params)
                loss, rec_loss, kld = self.loss_function_cvae(signal, recon, mean, log_var)
                
                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cvae.parameters(), max_norm=self.max_grad_norm)
                self.optimizerCVAE.step()

                total_loss += loss.item()
                reproduction_loss += rec_loss.item()
                kld_loss += kld.item()
                total_samples += signal.size(0)

            avg_total_loss = total_loss / total_samples
            avg_reproduction_loss = reproduction_loss / total_samples
            avg_kld_loss = kld_loss / total_samples

            self.avg_total_losses.append(avg_total_loss)
            self.avg_reproduction_losses.append(avg_reproduction_loss)
            self.avg_kld_losses.append(avg_kld_loss)

            # Validation
            self.cvae.eval()
            val_total_loss = 0
            val_reproduction_loss = 0
            val_kld_loss = 0
            val_samples = 0
            
            with torch.no_grad():
                for val_signal, val_noisy_signal, val_params in self.val_loader:
                    val_noisy_signal = val_noisy_signal.view(val_noisy_signal.size(0), -1).to(DEVICE)
                    val_signal = val_signal.view(val_signal.size(0), -1).to(DEVICE)
                    val_params = val_params.view(val_params.size(0), -1).to(DEVICE)
                    
                    recon, mean, log_var = self.cvae(val_noisy_signal, val_params)
                    v_loss, v_rec_loss, v_kld = self.loss_function_cvae(val_signal, recon, mean, log_var)
                    
                    val_total_loss += v_loss.item()
                    val_reproduction_loss += v_rec_loss.item()
                    val_kld_loss += v_kld.item()
                    val_samples += val_signal.size(0)
            
            avg_total_loss_val = val_total_loss / val_samples
            avg_reproduction_loss_val = val_reproduction_loss / val_samples
            avg_kld_loss_val = val_kld_loss / val_samples

            self.avg_total_losses_val.append(avg_total_loss_val)
            self.avg_reproduction_losses_val.append(avg_reproduction_loss_val)
            self.avg_kld_losses_val.append(avg_kld_loss_val)

            # Checkpoint: generate signals with fixed parameters
            if (epoch + 1) % self.checkpoint_interval == 0:
                with torch.no_grad():
                    self._plot_reconstruction(signal_idx=0)
                    self._plot_signal_grid(epoch)
                    # self._plot_latent_space(epoch)
                    
                    # Encode a sample of training data to visualize latent space
                    num_samples_to_encode = min(500, len(self.training_dataset))
                    sample_indices = np.random.choice(len(self.training_dataset), num_samples_to_encode, replace=False)
                    
                    latent_means = []
                    parameters_list = []
                    
                    for idx in sample_indices:
                        signal, noisy_signal, params = self.training_dataset[idx]
                        noisy_signal = noisy_signal.view(1, -1).to(DEVICE)
                        params = params.view(1, -1).to(DEVICE)
                        
                        mean, logvar = self.cvae.encoder(noisy_signal, params)
                        latent_means.append(mean.cpu().numpy().flatten())
                        parameters_list.append(params.cpu().numpy().flatten())
                    
                    latent_means = np.array(latent_means)
                    parameters_array = np.array(parameters_list)
                    
                    # Denormalize parameters for visualization
                    param_denorm = np.array([
                        self.training_dataset.denormalize_parameters(p) 
                        for p in parameters_array
                    ])
                    
                print(f"\nEpoch {epoch+1}/{self.num_epochs}")
                print(f"  Train Loss: {avg_total_loss:.4f} | Val Loss: {avg_total_loss_val:.4f}")
                
                # Extract Ye values for color-coding latent space
                ye_colors = None
                ye_param_label = None
                if self.theta_param_index is not None:
                    ye_values = param_denorm[:, self.theta_param_index]
                    
                    # Create color gradient from blue to yellow based on unique Ye values
                    unique_ye_values = np.unique(np.round(ye_values, 4))
                    ye_min = unique_ye_values.min()
                    ye_max = unique_ye_values.max()
                    
                    # Create colormap: blue to yellow
                    from matplotlib.colors import LinearSegmentedColormap
                    colors_list = ['blue', 'yellow']
                    n_bins = len(unique_ye_values)
                    cmap = LinearSegmentedColormap.from_list('blue_yellow', colors_list, N=n_bins)
                    
                    # Map each Ye value to a color
                    ye_colors = []
                    for val in ye_values:
                        normalized = (val - ye_min) / (ye_max - ye_min) if ye_max > ye_min else 0.5
                        ye_colors.append(cmap(normalized))
                    
                    ye_param_label = self.theta_label if self.theta_label else "Ye"
                
                # Plot latent space using dedicated function
                latent_plot_fname = os.path.join(self.outdir, "cvae", f'cvae_latent_space_epoch_{epoch+1}.svg')
                plot_latent_space_2d_3d(
                    latent_means=latent_means,
                    param_denorm=param_denorm,
                    epoch=epoch + 1,
                    fname=latent_plot_fname,
                    background="black",
                    param_label=ye_param_label,
                    ye_colors=ye_colors
                )
                print(f"  Saved latent space plot to {latent_plot_fname}")


        runtime = (time.time() - t0) / 60
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Training Time: {runtime:.2f} minutes")
        print(f"{'='*60}")
        
        self.save_data()

    def generate_signals_with_params(self, target_params: np.ndarray, num_samples: int = 100) -> np.ndarray:
        """Generate signals with specific parameter values.
        
        Args:
            target_params: Parameter values of shape (param_dim,) or (num_batches, param_dim)
            num_samples: Number of samples to generate per parameter set
            
        Returns:
            Generated signals of shape (num_samples * num_batches, signal_length)
        """
        self.cvae.eval()
        
        # Ensure params is 2D
        if target_params.ndim == 1:
            target_params = target_params.reshape(1, -1)
        
        num_param_sets = target_params.shape[0]
        
        # Repeat each parameter set num_samples times
        params_repeated = np.repeat(target_params, num_samples, axis=0)
        params_tensor = torch.tensor(params_repeated, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            # Sample from standard Gaussian prior
            z = torch.randn(num_samples * num_param_sets, self.z_dim).to(DEVICE)
        
        with torch.no_grad():
            generated = self.cvae.decoder(z, params_tensor).cpu().numpy()
        
        return generated
    

    def _plot_signal_grid(self, epoch):
        generated_signals = self.cvae.decoder(self.fixed_noise, self.fixed_params).cpu().detach().numpy()
        
        # Extract parameter values - use Ye if available, otherwise use theta_label
        param_values_to_display = None
        param_colors = None
        param_label_to_use = None
        
        if self.theta_param_index is not None:
            # Denormalize fixed parameters to physical units for display
            fixed_params_denorm = np.array([
                self.training_dataset.denormalize_parameters(p) 
                for p in self.fixed_params.cpu().numpy()
            ])
            param_values_to_display = fixed_params_denorm[:, self.theta_param_index]
            
            # Get parameter name
            if hasattr(self.training_dataset, 'parameter_names'):
                param_label_to_use = self.training_dataset.parameter_names[self.theta_param_index]
            else:
                param_label_to_use = self.theta_label if self.theta_label else f"Param {self.theta_param_index}"
            
            # Create color gradient from blue to yellow based on unique Ye values
            unique_ye_values = np.unique(np.round(param_values_to_display, 4))
            ye_min = unique_ye_values.min()
            ye_max = unique_ye_values.max()
            
            # Create colormap: blue to yellow
            from matplotlib.colors import LinearSegmentedColormap
            colors_list = ['blue', 'yellow']
            n_bins = len(unique_ye_values)
            cmap = LinearSegmentedColormap.from_list('blue_yellow', colors_list, N=n_bins)
            
            # Map each Ye value to a color
            param_colors = []
            for val in param_values_to_display:
                # Normalize value to [0, 1]
                normalized = (val - ye_min) / (ye_max - ye_min) if ye_max > ye_min else 0.5
                param_colors.append(cmap(normalized))
        
        # Plot signal grid with custom coloring
        self._plot_signal_grid_with_colors(
            signals=generated_signals / TEN_KPC,
            max_value=self.validation_dataset.max_strain,
            num_cols=4,
            num_rows=4,
            fname=os.path.join(self.outdir, "cvae", f"cvae_generated_signals_epoch_{epoch+1}.svg"),
            param_values=param_values_to_display,
            param_label=param_label_to_use,
            param_colors=param_colors
        )

    def _plot_signal_grid_with_colors(self, signals, max_value, num_cols, num_rows, fname, 
                                       param_values=None, param_label=None, param_colors=None):
        """Plot signal grid with color-coded frames based on parameter values."""
        from ..plotting import set_plot_style, get_time_axis
        from ..utils.plotting_defaults import GENERATED_SIGNAL_COLOUR, SIGNAL_LIM_UPPER, SIGNAL_LIM_LOWER
        
        set_plot_style("white", "Serif", "Times New Roman")
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
        axes = axes.flatten()
        
        d = get_time_axis()
        
        for i, ax in enumerate(axes):
            if i >= len(signals):
                ax.axis('off')
                continue
            
            y = signals[i].flatten()
            y = y * max_value
            ax.set_ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
            ax.set_xlim(min(d), max(d))
            ax.plot(d, y, color=GENERATED_SIGNAL_COLOUR)
            
            ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
            ax.grid(False)
            
            # Display parameter value above each subplot
            if param_values is not None and param_label is not None and i < len(param_values):
                param_text = f"{param_label} = {param_values[i]:.4f}"
                ax.set_title(param_text, fontsize=11, color="black", pad=8, fontweight='bold')
            
            if i % num_cols != 0:
                ax.yaxis.set_ticklabels([])
            if i < num_cols * (num_rows - 1):
                ax.xaxis.set_ticklabels([])
        
        fig.supxlabel('time (s)', fontsize=20)
        fig.supylabel('h', fontsize=20)
        
        plt.tight_layout()
        if fname:
            plt.savefig(fname, dpi=300, bbox_inches="tight")
        
        plt.close()

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
        
    def _plot_reconstruction(self, signal_idx=0):
        """Plot reconstruction of a single signal from the validation set."""
        self.cvae.eval()
        val_signal, noisy_signal, val_params = self.val_loader.dataset.__getitem__(signal_idx)
        val_signal = val_signal.view(1, -1).to(DEVICE)
        val_noisy_signal = noisy_signal.view(1, -1).to(DEVICE)
        val_params = val_params.view(1, -1).to(DEVICE)

        recon, _, _ = self.cvae(val_noisy_signal, val_params)

        plot_reconstruction(
            original=val_signal.cpu().numpy(),
            reconstructed=recon.cpu().numpy(),
            max_value=self.validation_dataset.max_strain / TEN_KPC,
            font_family="Serif",
            font_name="Times New Roman",
            fname=os.path.join(self.outdir, "cvae", "cvae_reconstruction.svg")
        )

    def display_results(self, background="black"):
        """Display training results."""
        from ..plotting import plot_loss
        
        # Plot total losses
        print("\nPlotting Total Losses...")
        plot_loss(
            train_losses=self.avg_total_losses,
            val_losses=self.avg_total_losses_val,
            background=background,
            fname=os.path.join(self.outdir, "cvae", "cvae_total_loss.svg")
        )
        
        # Plot reconstruction losses
        print("Plotting Reconstruction Losses...")
        plot_loss(
            train_losses=self.avg_reproduction_losses,
            val_losses=self.avg_reproduction_losses_val,
            background=background,
            fname=os.path.join(self.outdir, "cvae", "cvae_reconstruction_loss.svg")
        )
        
        # Plot KLD losses
        print("Plotting KL Divergence Losses...")
        plot_loss(
            train_losses=self.avg_kld_losses,
            val_losses=self.avg_kld_losses_val,
            background=background,
            fname=os.path.join(self.outdir, "cvae", "cvae_kld_loss.svg")
        )

    @property
    def save_fname(self):
        return f"{self.outdir}/cvae_weights.pt"

    def save_data(self):
        torch.save(self.cvae.state_dict(), self.save_fname)
        print(f"Saved CVAE model to {self.save_fname}")
        
        # Save validation signals and parameters (real data for final testing)
        # Format: signals shape (signal_length, num_samples), params shape (num_samples, param_dim)
        # This matches the format expected by CCSNData custom_data parameter
        val_signals_path = f"{self.outdir}/cvae_val_signals.npy"
        val_params_path = f"{self.outdir}/cvae_val_parameters.npy"
        
        # Extract validation signals and parameters from validation dataset
        # validation_dataset.signals has shape (signal_length, num_val_samples)
        # validation_dataset.parameters has shape (num_val_samples, param_dim)
        val_signals = self.validation_dataset.signals  # Shape: (signal_length, num_samples)
        val_params = self.validation_dataset.parameters  # Shape: (num_samples, param_dim)
        
        np.save(val_signals_path, val_signals)
        np.save(val_params_path, val_params)
        print(f"Saved validation signals to {val_signals_path}")
        print(f"  Shape: {val_signals.shape} (signal_length, num_samples)")
        print(f"Saved validation parameters to {val_params_path}")
        print(f"  Shape: {val_params.shape} (num_samples, param_dim)")
        print(f"  Validation set: {val_signals.shape[1]} real signals for final testing")
    
    @classmethod
    def load_model(
        cls,
        model_path: str,
        y_length: int = Y_LENGTH,
        hidden_dim: int = HIDDEN_DIM,
        z_dim: int = Z_DIM,
        param_dim: int = 4
    ) -> ConditionalVAE:
        """Load a trained CVAE model from disk.
        
        Args:
            model_path: Path to the saved model weights (.pt file)
            y_length: Signal length dimension
            hidden_dim: Hidden layer dimension
            z_dim: Latent dimension
            param_dim: Number of physical parameters
            
        Returns:
            Loaded ConditionalVAE model
        """
        # Reconstruct model architecture
        cvae = ConditionalVAE(
            y_length=y_length,
            hidden_dim=hidden_dim,
            z_dim=z_dim,
            param_dim=param_dim
        ).to(DEVICE)
        
        # Load saved weights
        cvae.load_state_dict(torch.load(model_path, map_location=DEVICE))
        cvae.eval()
        
        print(f"✓ Loaded CVAE model from {model_path}")
        print(f"  Architecture: y_length={y_length}, hidden_dim={hidden_dim}, z_dim={z_dim}, param_dim={param_dim}")
        
        return cvae
