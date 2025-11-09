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


from ..plotting.plotting import plot_reconstruction_distribution, plot_waveform_grid, plot_signal_grid, plot_latent_space_3d, plot_loss, plot_individual_loss, plot_signal_distribution, plot_corner

from ..utils.defaults import Y_LENGTH, HIDDEN_DIM, Z_DIM, BATCH_SIZE, DEVICE
from ..nn.vae import VAE
from ..nn.vae_parameter import VAE_PARAMETER
# from ..nn.flow import FLOW

from ..data.toy_data import ToyData
from ..data.ccsn_data import CCSNData
from ..data.ccsn_snr_data import CCSNSNRData

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
        lr_flow: float = 1e-3,
        checkpoint_interval: int = 16,
        outdir: str = "outdir",
        noise: bool = True,
        curriculum: bool = True,
        toy: bool = True,
        vae_parameter_test: bool = False,
        max_grad_norm: float = 1.0,  # Maximum gradient norm for clipping
        snr: bool = False
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
        self.snr = snr

        # selector between toy and real data
        if self.toy:
            self.training_dataset = ToyData(num_signals=1684, signal_length=self.y_length)
            # self.validation_dataset = ToyData(num_signals=1684, signal_length=self.y_length)
        else:
            # placeholder
            # if self.snr:
                # print("Using SNR-based dataset")
                # self.training_dataset = CCSNDataSNR(num_epochs=self.num_epochs, noise=self.no
            self.training_dataset = CCSNData(num_epochs=self.num_epochs, noise=self.noise, curriculum=self.curriculum)
            # self.validation_dataset = CCSNData(num_epochs=self.num_epochs, noise=self.noise, curriculum=self.curriculum)

        # Use the same underlying dataset object with disjoint samplers
        # (alternatively, you could create a separate instance with identical preprocessing)
        self.validation_dataset = self.training_dataset
    
        self.training_dataset.verify_alignment()
        self.validation_dataset.verify_alignment()

        dataset_size = self.training_dataset.__len__()
        indices = list(range(dataset_size))
        split = int(np.floor(self.validation_split * dataset_size))
        # Deterministic split
        rng = np.random.RandomState(self.seed)
        rng.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        self.training_sampler = SubsetRandomSampler(train_indices)
        self.validation_sampler = SubsetRandomSampler(val_indices)

        self.train_loader = DataLoader(self.training_dataset, batch_size=self.batch_size, sampler=self.training_sampler)
        self.val_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size, sampler=self.validation_sampler)

        print(f"Training samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")

        self.checkpoint_interval = checkpoint_interval # what is this?

        os.makedirs(outdir, exist_ok=True)
        _set_seed(self.seed)

        # setup networks

        if self.vae_parameter_test:
            self.vae = VAE_PARAMETER(z_dim=self.z_dim, hidden_dim=self.hidden_dim, y_length=self.y_length, param_dim=self.training_dataset.parameters.shape[1]).to(DEVICE)
            self.vae.apply(_init_weights_vae)
        else:
            self.vae = VAE(z_dim=self.z_dim, hidden_dim=self.hidden_dim, y_length=self.y_length).to(DEVICE)
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

            # Update epoch for curriculum learning
            self.train_loader.dataset.set_epoch(epoch)

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

            # set validation 
            self.val_loader.dataset.set_epoch(epoch)
            
            # Step the learning rate scheduler
            self.scheduler.step(avg_total_loss_val)

            print(f"Epoch {epoch+1}/{self.num_epochs} | Train Loss: {avg_total_loss:.4f} | Val Loss: {avg_total_loss_val:.4f}")

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
                        signals=generated_signals,
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

        # train flows
        self.train_npe_with_vae(num_epochs=500, batch_size=32, lr=5e-4)


    def train_npe_with_vae(self, num_epochs=500, batch_size=BATCH_SIZE, lr=1e-4, flow=None):
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
        self.flow = flow.to(DEVICE, dtype=torch.float32)

        # model ends here

        optimizer = optim.Adam(self.flow.parameters(), lr=lr)

        for epoch in range(num_epochs):
            self.flow.train()
            total_loss = 0.0

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
                    _, mean, log_var = self.vae(noisy_signal)
                    mean = mean.view(mean.size(0), -1)
                    log_var = log_var.view(log_var.size(0), -1)
                    # don't sample from latent, use mean only due to stable training
                    # z_latent = vae.reparameterization(mean, log_var)
                    # z_latent = z_latent.view(z_latent.size(0), -1)

                # p(params | z)
                params = params.view(params.size(0), -1) 

                optimizer.zero_grad(set_to_none=True)

                log_prob = self.flow.log_prob(params, context=mean) # this conditions the flow on the latent variable z
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
    
    def plot_corner(self, index=0):
        val_idx = self.validation_sampler.indices[index]
        signal, noisy_signal, params = self.val_loader.dataset.__getitem__(val_idx)
        plot_corner(self.vae, self.flow, signal, noisy_signal, params)


    def plot_generated_signal_distribution(self, background="white", font_family="serif", font_name="Times New Roman", fname=None):
        number_of_signals = 10000
        noise = torch.randn(number_of_signals, Z_DIM).to(DEVICE)

        start_time = time.time()
        with torch.no_grad():
            generated_signals = self.vae.decoder(noise).cpu().detach().numpy()
        end_time = time.time()

        execution_time = end_time - start_time
        print("Execution Time:", execution_time, "seconds")    

        generated_signals_transpose = np.empty((Y_LENGTH, 0))

        for i in range(number_of_signals):
            y = generated_signals[i, :].flatten()
            y = y * self.training_dataset.max_strain
            y = y.reshape(-1, 1)
            
            generated_signals_transpose = np.concatenate((generated_signals_transpose, y), axis=1)

        plot_signal_distribution(signals=generated_signals_transpose, generated=True, background=background, font_family=font_family, font_name=font_name, fname=fname)

    def plot_reconstruction_distribution(self, index, num_samples, background="white", font_family="sans-serif", font_name="Avenir", fname=None):
        val_idx = self.validation_sampler.indices[index]
        signal, noisy_signal, params = self.val_loader.dataset.__getitem__(val_idx)
        plot_reconstruction_distribution(
            vae=self.vae,
            signal=noisy_signal,
            max_value=self.val_loader.dataset.max_strain,
            num_samples=num_samples,
            background=background,
            font_family=font_family,
            font_name=font_name,
            fname=fname
        )

    def display_results(self):
        # Plot training and validation losses
        plot_individual_loss(
            self.avg_total_losses, self.avg_reproduction_losses, self.avg_kld_losses
        )
        plot_individual_loss(
            self.avg_total_losses_val, self.avg_reproduction_losses_val, self.avg_kld_losses_val
        )
        plot_loss(self.avg_total_losses, self.avg_total_losses_val)
        

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
