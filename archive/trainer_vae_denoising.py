"""VAE trainer for signal denoising with curriculum learning on SNR."""

import os
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import trange

from .vae_vanilla import VAE
from ..src.starccato_flow.utils.defaults_general import TEN_KPC, Y_LENGTH, HIDDEN_DIM, Z_DIM, BATCH_SIZE, DEVICE
from ..src.starccato_flow.plotting import plot_loss
from ..src.starccato_flow.plotting.signals import plot_reconstruction, plot_candidate_signal, plot_signal_distribution
from ..src.starccato_flow.utils.defaults_plotting import PARAMETER_LABELS


def _set_seed(seed: int):
    """Set the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return seed


class VAEDenoisingTrainer:
    """Trainer for VAE-based signal denoising with curriculum learning.
    
    Features:
    - Vanilla VAE architecture (no parameter conditioning)
    - Curriculum learning: SNR linearly decreases from 200 to 8 over epochs
    - Per-signal SNR calculation for dynamic curriculum
    - LIGO noise from detector sensitivity curve
    """
    
    def __init__(
        self,
        y_length: int = Y_LENGTH,
        hidden_dim: int = HIDDEN_DIM,
        z_dim: int = Z_DIM,
        seed: int = 99,
        batch_size: int = BATCH_SIZE,
        num_epochs: int = 256,
        validation_split: float = 0.1,
        learning_rate: float = 1e-3,
        checkpoint_interval: int = 16,
        outdir: Optional[str] = None,
        detector_noise_on: bool = True,
        max_grad_norm: float = 1.0,
        curriculum_snr_start: float = 200.0,
        curriculum_snr_end: float = 8.0,
        beta: float = 1.0,
    ):
        """Initialize VAE denoising trainer.
        
        Args:
            y_length: Length of signal
            hidden_dim: Hidden dimension for VAE
            z_dim: Latent dimension for VAE
            seed: Random seed
            batch_size: Batch size for training
            num_epochs: Number of epochs
            validation_split: Validation set fraction
            learning_rate: Learning rate for optimizer
            checkpoint_interval: Interval for saving checkpoints
            outdir: Output directory for models and plots
            detector_noise_on: Whether to add detector noise
            max_grad_norm: Maximum gradient norm for clipping
            curriculum_snr_start: Starting SNR for curriculum (high quality = high SNR)
            curriculum_snr_end: Ending SNR for curriculum (low quality = low SNR)
            beta: Weight for KL divergence in ELBO (0 = autoencoder, 1 = full VAE)
        """
        self.y_length = y_length
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.seed = seed
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.checkpoint_interval = checkpoint_interval
        self.detector_noise_on = detector_noise_on
        self.max_grad_norm = max_grad_norm
        self.beta = beta
        self.curriculum_snr_start = curriculum_snr_start
        self.curriculum_snr_end = curriculum_snr_end
        self.device = DEVICE
        
        # Construct absolute outdir path if not provided
        if outdir is None:
            _module_dir = os.path.dirname(os.path.abspath(__file__))
            _starccato_flow_root = os.path.dirname(os.path.dirname(os.path.dirname(_module_dir)))
            outdir = os.path.join(_starccato_flow_root, "outdir")
        
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"VAE Denoising Trainer")
        print(f"{'='*60}")
        print(f"Z dimension: {self.z_dim}")
        print(f"Hidden dimension: {self.hidden_dim}")
        print(f"Output directory: {self.outdir}")
        print(f"Curriculum SNR: {curriculum_snr_start} -> {curriculum_snr_end}")
        print(f"Beta (KL weight): {self.beta}")
        print(f"{'='*60}\n")
        
        # Set seed
        _set_seed(self.seed)
        
        # Import here to avoid circular import
        from ..src.starccato_flow.training import create_train_val_split
        
        # Create train/val split
        self.training_dataset, self.validation_dataset, self.val_indices = create_train_val_split(
            y_length=self.y_length,
            detector_noise_on=self.detector_noise_on,
            validation_split=self.validation_split,
            seed=self.seed,
            num_epochs=self.num_epochs
        )
        
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
        
        print(f"Training samples: {len(self.training_dataset)}")
        print(f"Validation samples: {len(self.validation_dataset)}")
        print(f"{'='*60}\n")
        
        # Initialize VAE model
        self.vae = VAE(z_dim=self.z_dim, hidden_dim=self.hidden_dim, y_length=self.y_length).to(self.device)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.vae.parameters(), lr=self.learning_rate)
        
        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        self.recon_losses = []
        self.kl_losses = []
        
        # Create subdirectory for checkpoints
        self.checkpoint_dir = os.path.join(self.outdir, "vae_denoising", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _get_curriculum_snr(self, epoch: int) -> float:
        """Get target SNR for current epoch using linear curriculum.
        
        SNR decreases linearly from curriculum_snr_start to curriculum_snr_end.
        Higher SNR = cleaner signals (easier). Lower SNR = noisier signals (harder).
        
        Args:
            epoch: Current epoch (0-indexed)
            
        Returns:
            Target SNR for this epoch
        """
        progress = epoch / max(self.num_epochs - 1, 1)
        snr = self.curriculum_snr_start + progress * (self.curriculum_snr_end - self.curriculum_snr_start)
        return snr
    
    def vae_loss(self, reconstruction, original, mean, log_var):
        """Compute ELBO loss: reconstruction + beta * KL divergence.
        
        Args:
            reconstruction: Reconstructed signal
            original: Original signal
            mean: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            Total loss
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = nn.MSELoss()(reconstruction, original)
        
        # KL divergence (analytic form for Gaussian)
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        kl_loss /= original.shape[0] * self.y_length  # Normalize by batch and signal length
        
        # Total loss with beta weighting
        total_loss = reconstruction_loss + self.beta * kl_loss
        
        return total_loss, reconstruction_loss, kl_loss
    
    def train(self):
        """Train the VAE with curriculum learning on SNR."""
        print("Starting training...")
        print(f"Device: {self.device}\n")
        
        t0 = time.time()
        
        for epoch in trange(self.num_epochs, desc="Epoch"):
            # Get curriculum SNR for this epoch
            target_snr = self._get_curriculum_snr(epoch)
            self.training_dataset.update_snr(target_snr)
            self.validation_dataset.update_snr(target_snr)
            
            # Training phase
            self.vae.train()
            train_loss = 0
            train_recon_loss = 0
            train_kl_loss = 0
            train_samples = 0
            
            for signals, noisy_signals, params in self.train_loader:
                # Use noisy signals as both input and target for denoising
                noisy_signals = noisy_signals.to(self.device)
                
                # Forward pass
                reconstruction, mean, log_var = self.vae(noisy_signals)
                
                # Loss computation
                loss, recon_loss, kl_loss = self.vae_loss(
                    reconstruction, noisy_signals, mean, log_var
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Accumulate losses
                train_loss += loss.item() * noisy_signals.shape[0]
                train_recon_loss += recon_loss.item() * noisy_signals.shape[0]
                train_kl_loss += kl_loss.item() * noisy_signals.shape[0]
                train_samples += noisy_signals.shape[0]
            
            avg_train_loss = train_loss / train_samples
            avg_train_recon = train_recon_loss / train_samples
            avg_train_kl = train_kl_loss / train_samples
            
            # Validation phase
            self.vae.eval()
            val_loss = 0
            val_recon_loss = 0
            val_kl_loss = 0
            val_samples = 0
            
            with torch.no_grad():
                for signals, noisy_signals, params in self.val_loader:
                    noisy_signals = noisy_signals.to(self.device)
                    
                    reconstruction, mean, log_var = self.vae(noisy_signals)
                    loss, recon_loss, kl_loss = self.vae_loss(
                        reconstruction, noisy_signals, mean, log_var
                    )
                    
                    val_loss += loss.item() * noisy_signals.shape[0]
                    val_recon_loss += recon_loss.item() * noisy_signals.shape[0]
                    val_kl_loss += kl_loss.item() * noisy_signals.shape[0]
                    val_samples += noisy_signals.shape[0]
            
            avg_val_loss = val_loss / val_samples
            avg_val_recon = val_recon_loss / val_samples
            avg_val_kl = val_kl_loss / val_samples
            
            # Store losses
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.recon_losses.append((avg_train_recon, avg_val_recon))
            self.kl_losses.append((avg_train_kl, avg_val_kl))
            
            # Logging
            if (epoch + 1) % max(1, self.num_epochs // 10) == 0:
                print(
                    f"Epoch {epoch+1}/{self.num_epochs} | "
                    f"SNR: {target_snr:.1f} | "
                    f"Train Loss: {avg_train_loss:.4f} (R: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f}) | "
                    f"Val Loss: {avg_val_loss:.4f} (R: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f})"
                )
            
            # Checkpointing
            if (epoch + 1) % self.checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"vae_denoising_epoch_{epoch+1:04d}.pt"
                )
                torch.save(self.vae.state_dict(), checkpoint_path)
        
        runtime = (time.time() - t0) / 60
        print(f"\nTraining completed in {runtime:.2f} minutes")
        
        # Save final model and losses
        self.save_models()
        self.save_losses()
        
        # Plot training curves
        self.plot_losses()
    
    def save_models(self):
        """Save the trained VAE model."""
        model_path = os.path.join(self.outdir, "vae_denoising", "vae_weights_final.pt")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.vae.state_dict(), model_path)
        print(f"✓ Saved model to {model_path}")
    
    def save_losses(self):
        """Save training and validation losses."""
        loss_path = os.path.join(self.outdir, "vae_denoising", "losses.npz")
        os.makedirs(os.path.dirname(loss_path), exist_ok=True)
        
        recon_train = np.array([r[0] for r in self.recon_losses])
        recon_val = np.array([r[1] for r in self.recon_losses])
        kl_train = np.array([k[0] for k in self.kl_losses])
        kl_val = np.array([k[1] for k in self.kl_losses])
        
        np.savez(
            loss_path,
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            recon_train=recon_train,
            recon_val=recon_val,
            kl_train=kl_train,
            kl_val=kl_val,
        )
        print(f"✓ Saved losses to {loss_path}")
    
    def plot_losses(self, fname: Optional[str] = None):
        """Plot training and validation losses."""
        if fname is None:
            fname = os.path.join(self.outdir, "vae_denoising", "training_losses.png")
        
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total loss
        axes[0, 0].plot(self.train_losses, label="Train", linewidth=2)
        axes[0, 0].plot(self.val_losses, label="Validation", linewidth=2)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Total Loss")
        axes[0, 0].set_title("ELBO Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        recon_train = np.array([r[0] for r in self.recon_losses])
        recon_val = np.array([r[1] for r in self.recon_losses])
        axes[0, 1].plot(recon_train, label="Train", linewidth=2)
        axes[0, 1].plot(recon_val, label="Validation", linewidth=2)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Reconstruction Loss (MSE)")
        axes[0, 1].set_title("Reconstruction Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # KL loss
        kl_train = np.array([k[0] for k in self.kl_losses])
        kl_val = np.array([k[1] for k in self.kl_losses])
        axes[1, 0].plot(kl_train, label="Train", linewidth=2)
        axes[1, 0].plot(kl_val, label="Validation", linewidth=2)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("KL Divergence")
        axes[1, 0].set_title("KL Loss (β={:.1f})".format(self.beta))
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Curriculum SNR over epochs
        snrs = [self._get_curriculum_snr(e) for e in range(self.num_epochs)]
        axes[1, 1].plot(snrs, linewidth=2, color="purple")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Target SNR")
        axes[1, 1].set_title("Curriculum Learning: SNR Schedule")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved loss plot to {fname}")
    
    def reconstruct_signals(self, num_samples: int = 100, snr: float = None) -> dict:
        """Generate reconstructions on validation set.
        
        Args:
            num_samples: Number of signals to reconstruct
            snr: Target SNR (if None, uses final curriculum SNR)
            
        Returns:
            Dictionary with clean, noisy, and reconstructed signals
        """
        if snr is None:
            snr = self.curriculum_snr_end
        
        self.validation_dataset.update_snr(snr)
        
        self.vae.eval()
        
        clean_signals = []
        noisy_signals = []
        reconstructed_signals = []
        
        with torch.no_grad():
            for i, (signals, noisy, params) in enumerate(self.val_loader):
                if i * self.batch_size >= num_samples:
                    break
                
                signals = signals.to(self.device)
                noisy = noisy.to(self.device)
                
                reconstruction, _, _ = self.vae(noisy)
                
                clean_signals.append(signals.cpu().numpy())
                noisy_signals.append(noisy.cpu().numpy())
                reconstructed_signals.append(reconstruction.cpu().numpy())
        
        return {
            "clean": np.vstack(clean_signals)[:num_samples],
            "noisy": np.vstack(noisy_signals)[:num_samples],
            "reconstructed": np.vstack(reconstructed_signals)[:num_samples],
        }


if __name__ == "__main__":
    # Example usage
    trainer = VAEDenoisingTrainer(
        num_epochs=256,
        curriculum_snr_start=200.0,
        curriculum_snr_end=8.0,
        beta=1.0,
    )
    trainer.train()
