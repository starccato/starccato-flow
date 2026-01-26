import os
import time
from typing import List, Optional

import matplotlib.pyplot as plt
import math
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import SubsetRandomSampler, DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm, trange

from ..nn.cvae import ConditionalVAE

from ..utils.defaults import TEN_KPC, Y_LENGTH, HIDDEN_DIM, Z_DIM, BATCH_SIZE, DEVICE

from . import create_train_val_split, plot_generated_signal_distribution, plot_candidate_signal_method, display_results_method, plot_signal_grid, plot_latent_space_3d

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
        noise: bool = True,
        curriculum: bool = True,
        toy: bool = True,
        max_grad_norm: float = 1.0,
        start_snr: int = 100,
        end_snr: int = 10,
        noise_realizations: int = 1
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
        self.noise = noise
        self.curriculum = curriculum
        self.max_grad_norm = max_grad_norm
        self.start_snr = start_snr
        self.end_snr = end_snr
        self.noise_realizations = noise_realizations
        self.device = DEVICE

        # Create train/val split using shared utility function
        self.training_dataset, self.validation_dataset = create_train_val_split(
            toy=self.toy,
            y_length=self.y_length,
            noise=self.noise,
            validation_split=self.validation_split,
            seed=self.seed,
            num_epochs=self.num_epochs,
            start_snr=start_snr,
            end_snr=end_snr,
            curriculum=self.curriculum,
            noise_realizations=self.noise_realizations
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
        
        # Define parameter sets for rows
        if self.param_dim == 1:
            # 4 different beta values
            param_sets = [
                np.array([0.02]),   # Row 1: Low beta
                np.array([0.08]),   # Row 2: Medium-low beta
                np.array([0.14]),   # Row 3: Medium-high beta
                np.array([0.18])    # Row 4: High beta
            ]
        elif self.param_dim == 4:
            # 4 different parameter combinations (beta, omega, A, Ye)
            param_sets = [
                np.array([0.02, 6.0, 3000.0, 0.10]),    # Row 1: Low values
                np.array([0.08, 8.5, 5000.0, 0.13]),    # Row 2: Medium-low
                np.array([0.14, 11.0, 7000.0, 0.17]),   # Row 3: Medium-high
                np.array([0.18, 14.0, 9000.0, 0.20])    # Row 4: High values
            ]
        else:
            # Use linspace for any other param_dim
            param_sets = [np.full(self.param_dim, i / (num_rows - 1)) for i in range(num_rows)]
        
        # Normalize parameter sets
        param_sets_norm = [self.training_dataset.normalize_parameters(p) for p in param_sets]
        
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

            self.val_loader.dataset.set_epoch(epoch)
            self.train_loader.dataset.set_epoch(epoch)

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
                    # Generate signals conditioned on fixed parameters
                    # Using fixed noise for visualization consistency across epochs
                    generated_signals = self.cvae.decoder(self.fixed_noise, self.fixed_params).cpu().detach().numpy()
                    
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
                print(f"  Generated signals shape: {generated_signals.shape}")
                
                # DIAGNOSTIC: Check if decoder is sensitive to parameter changes
                with torch.no_grad():
                    # Define 2 very different parameter sets
                    if self.param_dim == 1:
                        test_params = [
                            np.array([0.02]),   # Low beta
                            np.array([0.18])    # High beta
                        ]
                    elif self.param_dim == 4:
                        test_params = [
                            np.array([0.02, 6.0, 3000.0, 0.10]),   # Low values
                            np.array([0.18, 14.0, 9000.0, 0.20])   # High values
                        ]
                    else:
                        test_params = [np.zeros(self.param_dim), np.ones(self.param_dim)]
                    
                    test_params_norm = [self.training_dataset.normalize_parameters(p) for p in test_params]
                    
                    # Use SAME noise for both parameter sets
                    z_test = torch.randn(1, self.z_dim).to(DEVICE)
                    
                    test_signals = []
                    for params_norm in test_params_norm:
                        params_tensor = torch.tensor(params_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        signal = self.cvae.decoder(z_test, params_tensor).cpu().numpy()
                        test_signals.append(signal)
                    
                    # Check if signals are actually different
                    diff = np.abs(test_signals[0] - test_signals[1]).mean()
                    signal_std = np.mean([test_signals[0].std(), test_signals[1].std()])
                    relative_diff = diff / (signal_std + 1e-8)
                    
                    print(f"  Parameter Sensitivity Check:")
                    print(f"    Mean absolute difference: {diff:.6f}")
                    print(f"    Relative difference: {relative_diff:.4f}")
                    
                    if relative_diff < 0.1:
                        print(f"    ⚠️  WARNING: Model may be ignoring parameters (diff too small)")
                    else:
                        print(f"    ✓ Model is using parameters (signals differ)")
                
                # Plot generated signals
                plot_signal_grid(
                    signals=generated_signals / TEN_KPC,
                    noisy_signals=None,
                    max_value=self.training_dataset.max_strain,
                    num_cols=4,
                    num_rows=4,
                    fname=f"plots/cvae_generated_signals_epoch_{epoch+1}.svg",
                    background="white",
                    generated=True
                )
                
                # Plot latent space colored by first parameter (beta)
                fig = plt.figure(figsize=(15, 5))
                
                # 2D: dims 0-1
                ax1 = fig.add_subplot(131)
                scatter1 = ax1.scatter(latent_means[:, 0], latent_means[:, 1], 
                                     c=param_denorm[:, 0], cmap='viridis', 
                                     alpha=0.6, s=20)
                ax1.set_xlabel('Latent Dim 0', fontsize=11)
                ax1.set_ylabel('Latent Dim 1', fontsize=11)
                ax1.set_title('Latent Space (0-1)', fontsize=12)
                ax1.grid(True, alpha=0.3)
                cbar1 = plt.colorbar(scatter1, ax=ax1)
                cbar1.set_label('β', fontsize=11)
                
                # 2D: dims 1-2
                ax2 = fig.add_subplot(132)
                scatter2 = ax2.scatter(latent_means[:, 1], latent_means[:, 2], 
                                     c=param_denorm[:, 0], cmap='viridis', 
                                     alpha=0.6, s=20)
                ax2.set_xlabel('Latent Dim 1', fontsize=11)
                ax2.set_ylabel('Latent Dim 2', fontsize=11)
                ax2.set_title('Latent Space (1-2)', fontsize=12)
                ax2.grid(True, alpha=0.3)
                cbar2 = plt.colorbar(scatter2, ax=ax2)
                cbar2.set_label('β', fontsize=11)
                
                # 3D view
                ax3 = fig.add_subplot(133, projection='3d')
                scatter3 = ax3.scatter(latent_means[:, 0], latent_means[:, 1], latent_means[:, 2],
                                     c=param_denorm[:, 0], cmap='viridis', 
                                     alpha=0.6, s=20)
                ax3.set_xlabel('Latent Dim 0', fontsize=9)
                ax3.set_ylabel('Latent Dim 1', fontsize=9)
                ax3.set_zlabel('Latent Dim 2', fontsize=9)
                ax3.set_title('3D Latent Space', fontsize=12)
                cbar3 = plt.colorbar(scatter3, ax=ax3, pad=0.1, shrink=0.8)
                cbar3.set_label('β', fontsize=11)
                
                plt.suptitle(f'CVAE Latent Space (Epoch {epoch+1})', fontsize=14)
                plt.tight_layout()
                plt.savefig(f'plots/cvae_latent_space_epoch_{epoch+1}.svg', bbox_inches='tight', dpi=150)
                plt.close()
                
                print(f"  Saved latent space plot to plots/cvae_latent_space_epoch_{epoch+1}.svg")


        runtime = (time.time() - t0) / 60
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Training Time: {runtime:.2f} minutes")
        print(f"{'='*60}")
        
        self.save_models()

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
        from ..plotting.plotting import plot_loss
        
        # Plot total losses
        print("\nPlotting Total Losses...")
        plot_loss(
            train_losses=self.avg_total_losses,
            val_losses=self.avg_total_losses_val,
            background=background,
            fname="plots/cvae_total_loss.svg"
        )
        
        # Plot reconstruction losses
        print("Plotting Reconstruction Losses...")
        plot_loss(
            train_losses=self.avg_reproduction_losses,
            val_losses=self.avg_reproduction_losses_val,
            background=background,
            fname="plots/cvae_reconstruction_loss.svg"
        )
        
        # Plot KLD losses
        print("Plotting KL Divergence Losses...")
        plot_loss(
            train_losses=self.avg_kld_losses,
            val_losses=self.avg_kld_losses_val,
            background=background,
            fname="plots/cvae_kld_loss.svg"
        )

    @property
    def save_fname(self):
        return f"{self.outdir}/cvae_weights.pt"

    def save_models(self):
        torch.save(self.cvae.state_dict(), self.save_fname)
        print(f"Saved CVAE model to {self.save_fname}")
