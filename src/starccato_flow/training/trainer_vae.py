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

from ..plotting.plotting import plot_reconstruction_distribution, plot_signal_grid, plot_latent_space_3d, plot_loss, plot_individual_loss, plot_signal_distribution, plot_corner, plot_candidate_signal

from ..utils.defaults import Y_LENGTH, HIDDEN_DIM, Z_DIM, BATCH_SIZE, DEVICE, TEN_KPC
from ..nn.flow import Flow

from ..data.toy_data import ToyData
from ..data.ccsn_data import CCSNData
from . import create_train_val_split, plot_generated_signal_distribution, plot_candidate_signal_method, display_results_method

def _set_seed(seed: int):
    """Set the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return seed

class VAETrainer:
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
        max_grad_norm: float = 1.0,  # Maximum gradient norm for clipping
        start_snr: int = 100,
        end_snr: int = 10,
        noise_realizations: int = 1  # Number of noise realizations per signal
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

        self.checkpoint_interval = checkpoint_interval

        os.makedirs(outdir, exist_ok=True)
        _set_seed(self.seed)

        # setup Flow Matching model
        # Get parameter dimension (2 for toy data with two moons)
        self.flow = Flow(dim=self.training_dataset.parameters.shape[1]).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.lr_flow, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()

    def train(self):
        t0 = time.time()

        self.avg_mse_losses = []
        self.avg_mse_losses_val = []

        for epoch in trange(self.num_epochs, desc="Epochs", position=0, leave=True):
            self.flow.train()
            total_loss = 0
            total_samples = 0

            self.val_loader.dataset.set_epoch(epoch)
            self.train_loader.dataset.set_epoch(epoch)

            for signal, noisy_signal, params in self.train_loader:
                signal = signal.view(signal.size(0), -1).to(DEVICE)
                noisy_signal = noisy_signal.view(signal.size(0), -1).to(DEVICE)
                params = params.view(params.size(0), -1).to(DEVICE)

                x_0 = torch.randn_like(params)  # noise in parameter space
                t = torch.rand(len(params), 1, device=DEVICE)  # random time values on correct device
                x_t = (1 - t) * x_0 + t * params  # interpolated parameters
                dx_t = params - x_0  # true velocity direction in parameter space

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
                for val_signal, val_noisy_signal, val_params in self.val_loader:
                    # Match training: evaluate VAE on noisy inputs for a fair comparison
                    val_noisy_signal = val_noisy_signal.view(val_noisy_signal.size(0), -1).to(DEVICE)
                    val_signal = val_signal.view(val_signal.size(0), -1).to(DEVICE)
                    val_params = val_params.view(val_params.size(0), -1).to(DEVICE)

                    x_0 = torch.randn_like(val_params)  # noise in parameter space
                    t = torch.rand(len(val_params), 1, device=DEVICE)  # random time values on correct device
                    x_t = (1 - t) * x_0 + t * val_params  # interpolated parameters
                    dx_t = val_params - x_0  # true velocity direction in parameter space

                    loss = self.loss_fn(self.flow(x_t, t, val_noisy_signal), dx_t)
                    val_total_loss += loss.item()
                    val_samples += val_signal.size(0)
            
            avg_total_loss_val = val_total_loss / val_samples
            self.avg_mse_losses_val.append(avg_total_loss_val)

            print(f"Epoch {epoch+1}/{self.num_epochs} | Train MSE Loss: {avg_total_loss:.4f} | Val MSE Loss: {avg_total_loss_val:.4f}")

            # Optionally: add plotting or checkpointing here
            # if (epoch + 1) % self.checkpoint_interval == 0:
                # # gridded plots
                # with torch.no_grad():
                #     generated_signals = self.vae.decoder(self.fixed_noise).cpu().detach().numpy()
                # print(f"Generated signals shape: {generated_signals.shape}")
                # # plot_waveform_grid(signals=generated_signals, max_value=self.training_dataset.max_strain, generated=True)
                # if self.vae_parameter_test:
                #     print("Parameter values:", generated_signals)
                # else:
                #     plot_signal_grid(
                #         signals=generated_signals/TEN_KPC,
                #         noisy_signals=None,
                #         max_value=self.training_dataset.max_strain,
                #         num_cols=3,
                #         num_rows=1,
                #         fname="plots/ccsn_generated_signal_grid.svg",
                #         background="white",
                #         generated=True
                #     )
                # plot_latent_space_3d(
                #     model=self.vae,
                #     dataloader=self.train_loader
                # )
        

        runtime = (time.time() - t0) / 60
        print(f"Training Time: {runtime:.2f}min")
        # Optionally: plot final results or save model
        # self.save_models()


    def plot_candidate_signal(self, snr=100, background="white", index=0, fname="plots/candidate_signal.png"):
        """Plot a candidate signal with noise."""
        plot_candidate_signal_method(
            val_loader=self.val_loader,
            snr=snr,
            background=background,
            index=index,
            fname=fname
        )

    def plot_generated_signal_distribution(self, background="white", font_family="serif", font_name="Times New Roman", fname=None, number_of_signals=10000):
        """Plot distribution of VAE-generated signals."""
        plot_generated_signal_distribution(
            vae=self.vae,
            training_dataset=self.training_dataset,
            background=background,
            font_family=font_family,
            font_name=font_name,
            fname=fname,
            number_of_signals=number_of_signals
        )

    # def plot_reconstruction_distribution(self, index=100, num_samples=1000, background="white", font_family="sans-serif", font_name="Avenir", fname=None):
    #     signal, noisy_signal, params = self.val_loader.dataset[index]

    #     # Generate reconstructions
    #     self.vae.eval()
    #     noisy_signal = noisy_signal.unsqueeze(0)
    #     reconstructed_signals = []
        
    #     with torch.no_grad():
    #         for _ in range(num_samples):
    #             reconstruction, _, _ = self.vae(noisy_signal)
    #             reconstructed_signals.append(
    #                 reconstruction.squeeze().cpu().numpy() * self.val_loader.dataset.max_strain
    #             )

    #     plot_reconstruction_distribution(
    #         reconstructed_signals=reconstructed_signals/TEN_KPC,
    #         noisy_signal=noisy_signal/TEN_KPC,
    #         true_signal=signal/TEN_KPC,
    #         max_value=self.val_loader.dataset.max_strain,
    #         num_samples=num_samples,
    #         background=background,
    #         font_family=font_family,
    #         font_name=font_name,
    #         fname=fname
    #     )

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

# def _init_weights_vae(m: torch.nn.Module) -> None:
#     """Initialize weights with Xavier/Glorot initialization for better gradient flow."""
#     if isinstance(m, nn.Linear):
#         if hasattr(m, 'weight') and m.weight is not None:
#             # Use Xavier uniform initialization for better gradient flow
#             nn.init.xavier_uniform_(m.weight, gain=1.0)
#         if hasattr(m, 'bias') and m.bias is not None:
#             # Initialize biases to small positive values for better gradient flow
#             nn.init.constant_(m.bias, 0.01)
#     elif isinstance(m, nn.BatchNorm1d):
#         if hasattr(m, 'weight') and m.weight is not None:
#             nn.init.constant_(m.weight, 1.0)
#         if hasattr(m, 'bias') and m.bias is not None:
#             nn.init.constant_(m.bias, 0.0)
