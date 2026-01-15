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
from ..nn.vae import VAE

from ..data.toy_data import ToyData
from ..data.ccsn_data import CCSNData
from . import plot_generated_signal_distribution as plot_generated_signal_distribution_shared
from . import plot_candidate_signal_method, display_results_method

from nflows.distributions.normal import StandardNormal
from nflows.transforms import CompositeTransform, ReversePermutation, MaskedAffineAutoregressiveTransform
from nflows.flows import Flow

def _set_seed(seed: int):
    """Set the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return seed

class Trainer:
    def __init__(
        self,
        y_length: int = Y_LENGTH,
        hidden_dim: int = HIDDEN_DIM,
        z_dim: int = Z_DIM,
        seed: int = 99,
        batch_size: int = BATCH_SIZE,
        num_epochs: int = 256,
        validation_split: float = 0.1,
        lr_vae: float = 1e-3,
        lr_flow: float = 1e-4,
        checkpoint_interval: int = 16,
        outdir: str = "outdir",
        noise: bool = True,
        curriculum: bool = True,
        toy: bool = True,
        vae_parameter_test: bool = False,
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
        self.lr_vae = lr_vae
        self.lr_flow = lr_flow
        self.checkpoint_interval = checkpoint_interval
        self.outdir = outdir
        self.toy = toy
        self.noise = noise
        self.curriculum = curriculum
        self.vae_parameter_test = vae_parameter_test
        self.max_grad_norm = max_grad_norm
        self.start_snr = start_snr
        self.end_snr = end_snr
        self.noise_realizations = noise_realizations

        # selector between toy and real data
        if self.toy:
            self.training_dataset = ToyData(num_signals=1684, signal_length=self.y_length)
            self.validation_dataset = ToyData(num_signals=1684, signal_length=self.y_length)
        else:
            # Create a temporary dataset to get the number of base signals (before augmentation)
            temp_dataset = CCSNData(
                num_epochs=self.num_epochs,
                start_snr=start_snr,
                end_snr=end_snr,
                noise=self.noise,
                curriculum=False, 
                noise_realizations=1
            )
            num_base_signals = temp_dataset.signals.shape[1]
            
            # Split on BASE signal indices (before augmentation)
            base_indices = list(range(num_base_signals))
            split = int(np.floor(self.validation_split * num_base_signals))
            
            # Deterministic split with fixed seed
            rng = np.random.RandomState(self.seed)
            rng.shuffle(base_indices)
            train_base_indices = np.array(base_indices[split:])
            val_base_indices = np.array(base_indices[:split])
            
            print(f"\n=== Data Split (on base signals) ===")
            print(f"Total base signals: {num_base_signals}")
            print(f"Training base signals: {len(train_base_indices)}")
            print(f"Validation base signals: {len(val_base_indices)}")
            print(f"First 5 training indices: {train_base_indices[:5]}")
            print(f"First 5 validation indices: {val_base_indices[:5]}")
            
            # VERIFY: No overlap between train and validation base indices
            train_set = set(train_base_indices)
            val_set = set(val_base_indices)
            overlap = train_set.intersection(val_set)
            
            if len(overlap) > 0:
                raise ValueError(
                    f"❌ DATA LEAKAGE DETECTED! {len(overlap)} signals appear in both "
                    f"train and validation sets: {sorted(list(overlap))[:10]}"
                )
            else:
                print(f"✓ Verification PASSED: No overlap between train and validation sets")
                print(f"  Train signals: {len(train_set)} unique indices")
                print(f"  Val signals: {len(val_set)} unique indices")
                print(f"  Total coverage: {len(train_set) + len(val_set)} / {num_base_signals}")
            
            # Create SEPARATE dataset instances with disjoint base indices
            # Training: with curriculum and multiple noise realizations
            self.training_dataset = CCSNData(
                num_epochs=self.num_epochs,
                start_snr=start_snr,
                end_snr=end_snr,
                noise=self.noise,
                curriculum=self.curriculum,
                noise_realizations=self.noise_realizations,
                indices=train_base_indices
            )
            
            # Validation: FIXED SNR (no curriculum) with single noise realization
            self.validation_dataset = CCSNData(
                num_epochs=self.num_epochs,
                start_snr=end_snr,
                end_snr=end_snr,
                noise=self.noise,
                curriculum=self.curriculum, 
                noise_realizations=1,
                indices=val_base_indices
            )
    
        self.training_dataset.verify_alignment()
        self.validation_dataset.verify_alignment()

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

        print(f"\n=== Dataset Sizes (after augmentation) ===")
        print(f"Training samples: {len(self.training_dataset)} ({len(train_base_indices)} base × {self.noise_realizations} realizations)")
        print(f"Validation samples: {len(self.validation_dataset)} ({len(val_base_indices)} base × 1)")
        print("=" * 50)

        self.checkpoint_interval = checkpoint_interval

        os.makedirs(outdir, exist_ok=True)
        _set_seed(self.seed)

        # setup VAE
        self.vae = VAE(z_dim=self.z_dim, hidden_dim=self.hidden_dim, y_length=self.y_length).to(DEVICE)
        # self.vae = vae_test.VAETest(z_dim=self.z_dim, hidden_dim=self.hidden_dim, y_length=self.y_length).to(DEVICE)
        self.vae.apply(_init_weights_vae)

        # Setup optimizer and scheduler
        self.optimizerVAE = optim.Adam(self.vae.parameters(), lr=self.lr_vae)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizerVAE,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )

        self.fixed_noise = torch.randn(batch_size, z_dim, device=DEVICE)

    def loss_function_vae(self, y, y_hat, mean, log_var):
        # sse loss
        reproduction_loss = nn.functional.mse_loss(y_hat, y, reduction='sum')
        reproduction_loss *= 1 * y.shape[1]

        # KL Divergence loss
        kld_beta = 1
        kld_loss = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        kld_loss = kld_loss * kld_beta

        # total loss
        total_loss = reproduction_loss + kld_loss

        return total_loss, reproduction_loss, kld_loss
    
    @property
    def plt_kwgs(self):
        return dict(
            scaling_factor=self.dataset.scaling_factor,
            mean=self.dataset.mean,
            std=self.dataset.std,
            max_value=self.dataset.max_value,
            num_cols=4,
            num_rows=4,
        )
    
    def train(self):
        t0 = time.time()

        self.avg_total_losses = []
        self.avg_reproduction_losses = []
        self.avg_kld_losses = []
        self.avg_total_losses_val = []
        self.avg_reproduction_losses_val = []
        self.avg_kld_losses_val = []

        for epoch in trange(self.num_epochs, desc="Epochs", position=0, leave=True):
            self.vae.train()
            total_loss = 0
            reproduction_loss = 0
            kld_loss = 0
            total_samples = 0

            self.val_loader.dataset.set_epoch(epoch)
            self.train_loader.dataset.set_epoch(epoch)

            for signal, noisy_signal, params in self.train_loader:
                signal = signal.view(signal.size(0), -1).to(DEVICE)
                noisy_signal = noisy_signal.view(signal.size(0), -1).to(DEVICE)
                params = params.view(params.size(0), -1).to(DEVICE)

                self.optimizerVAE.zero_grad()
                recon, mean, log_var = self.vae(noisy_signal)
                loss, rec_loss, kld = self.loss_function_vae(signal, recon, mean, log_var)
                
                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=self.max_grad_norm)
                self.optimizerVAE.step()

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
            self.vae.eval()
            val_total_loss = 0
            val_reproduction_loss = 0
            val_kld_loss = 0
            val_samples = 0
            with torch.no_grad():
                for val_signal, val_noisy_signal, val_params in self.val_loader:
                    # Match training: evaluate VAE on noisy inputs for a fair comparison
                    val_noisy_signal = val_noisy_signal.view(val_noisy_signal.size(0), -1).to(DEVICE)
                    val_signal = val_signal.view(val_signal.size(0), -1).to(DEVICE)
                    val_params = val_params.view(val_params.size(0), -1).to(DEVICE)
                    recon, mean, log_var = self.vae(val_noisy_signal)
                    v_loss, v_rec_loss, v_kld = self.loss_function_vae(val_signal, recon, mean, log_var)
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
            
            # Step the learning rate scheduler
            self.scheduler.step(avg_total_loss_val)

            # print(f"Epoch {epoch+1}/{self.num_epochs} | Train Loss: {avg_total_loss:.4f} | Val Loss: {avg_total_loss_val:.4f}")

            # Optionally: add plotting or checkpointing here
            if (epoch + 1) % self.checkpoint_interval == 0:
                # gridded plots
                with torch.no_grad():
                    generated_signals = self.vae.decoder(self.fixed_noise).cpu().detach().numpy()
                print(f"Generated signals shape: {generated_signals.shape}")
                # plot_waveform_grid(signals=generated_signals, max_value=self.training_dataset.max_strain, generated=True)
                if self.vae_parameter_test:
                    print("Parameter values:", generated_signals)
                else:
                    plot_signal_grid(
                        signals=generated_signals/TEN_KPC,
                        noisy_signals=None,
                        max_value=self.training_dataset.max_strain,
                        num_cols=3,
                        num_rows=1,
                        fname="plots/ccsn_generated_signal_grid.svg",
                        background="white",
                        generated=True
                    )
                plot_latent_space_3d(
                    model=self.vae,
                    dataloader=self.train_loader
                )
                

        runtime = (time.time() - t0) / 60
        print(f"Training Time: {runtime:.2f}min")
        # Optionally: plot final results or save model
        self.save_models()

        # train flows with overfitting prevention
        print("\n" + "="*60)
        print("Starting Flow Training")
        print("="*60)
        # self.train_npe_with_vae_standard(num_epochs=500, lr=self.lr_flow)
        # self.train_npe_with_vae_improved(num_epochs=500)
    
    def train_npe_with_vae_standard(self, num_epochs=256, lr=1e-4):
        """
        Train a MaskedAutoregressiveFlow to estimate p(params | latent)
        """
        self.vae.eval()
        param_dim = 4

        num_layers = 10

        # model starts here
        base_dist = StandardNormal(shape=[param_dim])

        # composite transform
        transforms = []
        for i in range(num_layers):
            if i % 2 == 0:
                transforms.append(ReversePermutation(features=param_dim))
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=param_dim,
                    hidden_features=128,
                    context_features=Z_DIM,
                )
            )

        transform = CompositeTransform(transforms)

        # create flow on CPU first, in float32
        self.flow = Flow(transform, base_dist)

        # move to device explicitly, MPS requires float32
        self.flow = self.flow.to(DEVICE, dtype=torch.float32)

        # model ends here

        optimizer = optim.Adam(self.flow.parameters(), lr=lr)

        for epoch in range(num_epochs):
            self.flow.train()
            total_loss = 0.0

            self.train_loader.dataset.set_epoch(epoch)
            self.val_loader.dataset.set_epoch(epoch)

            for signal, noisy_signal, params in self.train_loader:
                signal = signal.to(DEVICE).float()
                noisy_signal = noisy_signal.to(DEVICE).float()
                params = params.to(DEVICE).float()
                # take only the first param
                params = torch.log(params + 1e-8)  # log-transform

                # Encode signal into latent space
                with torch.no_grad():
                    _, mean, log_var = self.vae(noisy_signal)
                    mean = mean.view(mean.size(0), -1)
                    log_var = log_var.view(log_var.size(0), -1)
                    # don't sample from latent, use mean only due to stable training
                    # z_latent = vae.reparameterization(mean, log_var)
                    # z_latent = z_latent.view(z_latent.size(0), -1)

                # p(params | z)
                params = params.view(params.size(0), -1) 

                optimizer.zero_grad(set_to_none=True)

                log_prob = self.flow.log_prob(params, context=mean)
                loss = -log_prob.mean()

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            
            # validation step
            self.flow.eval()
            val_total_NLL_loss = 0.0
            with torch.no_grad():
                for val_signal, val_noisy_signal, val_params in self.val_loader:
                    val_signal = val_signal.to(DEVICE).float()
                    val_noisy_signal = val_noisy_signal.to(DEVICE).float()
                    val_params = val_params.to(DEVICE).float()
                    # take only the first param
                    # params = params[:, :, 0:1]
                    val_params = torch.log(val_params + 1e-8)  # log-transform
                    # params = params[:, :, 0]

                    # Encode signal into latent space
                    with torch.no_grad():
                        _, mean, _ = self.vae(val_noisy_signal)
                        mean = mean.view(mean.size(0), -1)
                        # don't sample from latent, use mean only due to stable training

                    # p(params | z)
                    val_params = val_params.view(val_params.size(0), -1) 

                    optimizer.zero_grad(set_to_none=True)

                    log_prob = self.flow.log_prob(val_params, context=mean) # this conditions the flow on the latent variable z
                    val_NLL_loss = -log_prob.mean()
                    val_total_NLL_loss += val_NLL_loss.item()

            # self.avg_total_losses_val.append(avg_total_NLL_loss_val)

            # Do not advance validation epoch-driven curriculum; keep it stable across epochs
            # self.val_loader.dataset.set_epoch(epoch)
            
            # Step the learning rate scheduler
            # self.scheduler.step(avg_total_NLL_loss_val)

            print(f"Epoch [{epoch+1}/{num_epochs}] | Flow Training NLL: {total_loss / len(self.train_loader):.4f}, | Flow Validation NLL: {val_total_NLL_loss / len(self.val_loader):.4f}") 


    def train_npe_with_vae_improved(self, num_epochs=500, lr=5e-5, patience=20):
        """
        Train a MaskedAutoregressiveFlow to estimate p(params | latent)
        With early stopping and regularization to prevent overfitting.
        """
        self.vae.eval()
        param_dim = 4

        num_layers = 5  # Further reduced from 5 to prevent overfitting

        # model starts here
        base_dist = StandardNormal(shape=[param_dim])

        # composite transform
        transforms = []
        for i in range(num_layers):
            if i % 2 == 0:
                transforms.append(ReversePermutation(features=param_dim))
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=param_dim,
                    hidden_features=128,
                    context_features=Z_DIM,
                    dropout_probability=0.1  # Added dropout for regularization
                )
            )

        transform = CompositeTransform(transforms)

        # create flow on CPU first, in float32
        self.flow = Flow(transform, base_dist)

        # move to device explicitly, MPS requires float32
        self.flow = self.flow.to(DEVICE, dtype=torch.float32)

        # model ends here
        
        # Add stronger weight decay for regularization
        optimizer = optim.Adam(self.flow.parameters(), lr=lr, weight_decay=5e-4)  # Increased from 1e-4
        
        # Learning rate scheduler to reduce LR on plateau
        self.flow_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7
        )
        
        # Initialize loss tracking lists (like VAE training)
        self.flow_train_nll_losses = []
        self.flow_val_nll_losses = []
        
        # Initialize gradient tracking
        self.flow_gradient_norms = []
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(num_epochs):
            self.flow.train()
            total_loss = 0.0
            epoch_grad_norms = []

            # Update epoch for training dataset only (curriculum learning)
            self.train_loader.dataset.set_epoch(epoch)
            self.val_loader.dataset.set_epoch(epoch)

            train_samples = 0
            for signal, noisy_signal, params in self.train_loader:
                signal = signal.to(DEVICE).float()
                noisy_signal = noisy_signal.to(DEVICE).float()
                params = params.to(DEVICE).float()
                # take only the first param
                # params = params[:, :, 0:1]
                params = torch.log(params + 1e-8)  # log-transform
                # params = params[:, :, 0]

                # Encode signal into latent space
                with torch.no_grad():
                    _, mean, _ = self.vae(noisy_signal)
                    mean = mean.view(mean.size(0), -1)

                # p(params | z)
                params = params.view(params.size(0), -1) 

                optimizer.zero_grad(set_to_none=True)

                log_prob = self.flow.log_prob(params, context=mean)
                loss = -log_prob.mean()
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss at epoch {epoch+1}, skipping batch")
                    continue

                loss.backward()
                # Gradient clipping to prevent exploding gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=1.0)
                epoch_grad_norms.append(grad_norm.item())
                optimizer.step()

                total_loss += loss.item()
                train_samples += signal.size(0)

            
            # Track average gradient norm for this epoch
            if epoch_grad_norms:
                avg_grad_norm = np.mean(epoch_grad_norms)
                self.flow_gradient_norms.append(avg_grad_norm)
                        
            # validation step
            self.flow.eval()
            val_total_NLL_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for val_signal, val_noisy_signal, val_params in self.val_loader:
                    val_signal = val_signal.to(DEVICE).float()
                    val_noisy_signal = val_noisy_signal.to(DEVICE).float()
                    val_params = val_params.to(DEVICE).float()
                    val_params = torch.log(val_params + 1e-8)

                    # Encode signal into latent space - use mean for validation (consistent with inference)
                    _, mean, _ = self.vae(val_noisy_signal)
                    mean = mean.view(mean.size(0), -1)

                    # p(params | z)
                    val_params = val_params.view(val_params.size(0), -1)

                    log_prob = self.flow.log_prob(val_params, context=mean)
                    val_NLL_loss = -log_prob.mean()
                    val_total_NLL_loss += val_NLL_loss.item()
                    val_samples += val_signal.size(0)

            avg_total_NLL_loss_val = val_total_NLL_loss / val_samples
            avg_train_NLL = total_loss / train_samples
            
            # Track losses (like VAE training)
            self.flow_train_nll_losses.append(avg_train_NLL)
            self.flow_val_nll_losses.append(avg_total_NLL_loss_val)
                        
            # Step the learning rate scheduler
            self.flow_scheduler.step(avg_total_NLL_loss_val)
            
            # Early stopping check
            if avg_total_NLL_loss_val < best_val_loss:
                best_val_loss = avg_total_NLL_loss_val
                patience_counter = 0
                best_model_state = self.flow.state_dict()
                grad_info = f" | Grad Norm: {avg_grad_norm:.4f}" if epoch_grad_norms else ""
                print(f"Epoch [{epoch+1}/{num_epochs}] | Flow Train NLL: {avg_train_NLL:.4f} | Val NLL: {avg_total_NLL_loss_val:.4f}{grad_info} ✓ (Best)")
            else:
                patience_counter += 1
                grad_info = f" | Grad Norm: {avg_grad_norm:.4f}" if epoch_grad_norms else ""
                print(f"Epoch [{epoch+1}/{num_epochs}] | Flow Train NLL: {avg_train_NLL:.4f} | Val NLL: {avg_total_NLL_loss_val:.4f}{grad_info} (Patience: {patience_counter}/{patience})")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                print(f"Best validation NLL: {best_val_loss:.4f}")
                self.flow.load_state_dict(best_model_state)
                break
        
        if best_model_state is not None and patience_counter < patience:
            print(f"\n✓ Training completed. Loading best model (Val NLL: {best_val_loss:.4f})")
            self.flow.load_state_dict(best_model_state) 
    
    def plot_corner(self, signal, noisy_signal, params, fname):
        # Validation dataset already contains only validation samples
        # signal, noisy_signal, params = self.validation_dataset[index]
        plot_corner(vae=self.vae, flow=self.flow, signal=signal, noisy_signal=noisy_signal, params=params, fname=fname)

    def plot_candidate_signal(self, snr=100, background="white", index=0, fname="plots/candidate_signal.png"):
        self.val_loader.dataset.update_snr(snr)
        signal, noisy_signal, _ = self.val_loader.dataset.__getitem__(index)
        plot_candidate_signal(signal=signal/TEN_KPC, noisy_signal=noisy_signal/TEN_KPC, max_value=self.val_loader.dataset.max_strain, background=background, fname=fname)


    def plot_generated_signal_distribution(self, background="white", font_family="serif", font_name="Times New Roman", fname=None, number_of_signals=10000):
        """Plot distribution of VAE-generated signals."""
        plot_generated_signal_distribution_shared(
            vae=self.vae,
            training_dataset=self.training_dataset,
            background=background,
            font_family=font_family,
            font_name=font_name,
            fname=fname,
            number_of_signals=number_of_signals
        )

    def plot_reconstruction_distribution(self, index=100, num_samples=1000, background="white", font_family="sans-serif", font_name="Avenir", fname=None):
        signal, noisy_signal, params = self.val_loader.dataset[index]

        # Generate reconstructions
        self.vae.eval()
        noisy_signal = noisy_signal.unsqueeze(0)
        reconstructed_signals = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                reconstruction, _, _ = self.vae(noisy_signal)
                reconstructed_signals.append(
                    reconstruction.squeeze().cpu().numpy() * self.val_loader.dataset.max_strain
                )

        plot_reconstruction_distribution(
            reconstructed_signals=reconstructed_signals/TEN_KPC,
            noisy_signal=noisy_signal/TEN_KPC,
            true_signal=signal/TEN_KPC,
            max_value=self.val_loader.dataset.max_strain,
            num_samples=num_samples,
            background=background,
            font_family=font_family,
            font_name=font_name,
            fname=fname
        )

    def display_results(self):
        # Plot VAE training and validation losses
        plot_individual_loss(
            self.avg_total_losses, self.avg_reproduction_losses, self.avg_kld_losses
        )
        plot_individual_loss(
            self.avg_total_losses_val, self.avg_reproduction_losses_val, self.avg_kld_losses_val
        )
        plot_loss(self.avg_total_losses, self.avg_total_losses_val, background="black")
        
        # Plot VAE gradient norms if available
        if hasattr(self, 'vae_gradient_norms'):
            if len(self.vae_gradient_norms) > 0:
                print("\nPlotting VAE Gradient Norms...")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(self.vae_gradient_norms, label='VAE Gradient Norm', color='#3498db', linewidth=2)
                ax.set_xlabel('Epoch', size=16)
                ax.set_ylabel('Gradient Norm', size=16)
                ax.set_title('VAE Gradient Norms During Training', size=18)
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=self.max_grad_norm, color='red', linestyle='--', alpha=0.5, label=f'Clipping Threshold ({self.max_grad_norm})')
                ax.legend(fontsize=12)
                plt.tight_layout()
                plt.show()
        
        # Plot Flow NLL losses if available
        if hasattr(self, 'flow_train_nll_losses') and hasattr(self, 'flow_val_nll_losses'):
            if len(self.flow_train_nll_losses) > 0:
                print("\nPlotting Flow NLL Losses...")
                plot_loss(
                    train_losses=self.flow_train_nll_losses, 
                    val_losses=self.flow_val_nll_losses,
                    background="black",
                    fname="plots/flow_loss_curve.svg"
                )
        
        # Plot Flow gradient norms if available
        if hasattr(self, 'flow_gradient_norms'):
            if len(self.flow_gradient_norms) > 0:
                print("\nPlotting Flow Gradient Norms...")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(self.flow_gradient_norms, label='Flow Gradient Norm', color='#9b59b6', linewidth=2)
                ax.set_xlabel('Epoch', size=16)
                ax.set_ylabel('Gradient Norm', size=16)
                ax.set_title('Flow Gradient Norms During Training', size=18)
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Clipping Threshold')
                plt.tight_layout()
                plt.show()
        

    @property
    def save_fname(self):
        return f"{self.outdir}/generator_weights.pt"

    def save_models(self):
        torch.save(self.vae.state_dict(), self.save_fname)
        print(f"Saved VAE model to {self.save_fname}")

def _init_weights_vae(m: torch.nn.Module) -> None:
    """Initialize weights with Xavier/Glorot initialization for better gradient flow."""
    if isinstance(m, nn.Linear):
        if hasattr(m, 'weight') and m.weight is not None:
            # Use Xavier uniform initialization for better gradient flow
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        if hasattr(m, 'bias') and m.bias is not None:
            # Initialize biases to small positive values for better gradient flow
            nn.init.constant_(m.bias, 0.01)
    elif isinstance(m, nn.BatchNorm1d):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def train(**kwargs):
    trainer = Trainer(**kwargs)
    trainer.train()
    return trainer
