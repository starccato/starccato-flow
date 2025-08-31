import os
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm.auto import tqdm, trange

from ..plotting.plotting import plot_waveform_grid, plot_latent_space_3d, plot_loss, plot_individual_loss, plot_signal_distribution

from ..utils.defaults import Y_LENGTH, HIDDEN_DIM, Z_DIM, BATCH_SIZE, DEVICE
from ..nn.vae import VAE
from ..nn.flow import FLOW

from ..data.toy_data import ToyData
from ..data.ccsn_data import CCSNData

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
        num_epochs=256,
        lr_vae=1e-3,
        lr_flow=1e-3,
        checkpoint_interval=16,
        outdir: str = "outdir",
        toy: bool = True
    ):
        self.y_length = y_length
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.seed = seed
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr_vae = lr_vae
        self.lr_flow = lr_flow
        self.checkpoint_interval = checkpoint_interval
        self.outdir = outdir
        self.toy = toy

        # selector between toy and real data
        if self.toy:
            self.training_dataset = ToyData(num_signals=1684, signal_length=self.y_length)
            self.validation_dataset = ToyData(num_signals=1684, signal_length=self.y_length)
        else:
            # placeholder
            self.training_dataset = CCSNData()
            self.validation_dataset = CCSNData()

        self.checkpoint_interval = checkpoint_interval # what is this?

        os.makedirs(outdir, exist_ok=True)
        _set_seed(self.seed)

        # setup networks
        self.vae = VAE(z_dim=self.z_dim, hidden_dim=self.hidden_dim, y_length=self.y_length).to(DEVICE)
        self.vae.apply(_init_weights_vae)
        self.flow = FLOW(z_dim=self.z_dim, hidden_dim=self.hidden_dim, y_length=self.y_length).to(DEVICE)
        self.flow.apply(_init_weights_flow)

        # setup optimisers
        self.optimizerVAE = optim.Adam(
            self.vae.parameters(), lr=self.lr_vae
        )
        # self.optimizerFlow = optim.Adam(
        #     self.flow.parameters(), lr=self.lr_flow
        # )

        self.fixed_noise = torch.randn(batch_size, z_dim, device=DEVICE)

        # self.train_metadata: TrainMetadata = TrainMetadata() # what is this?

    def loss_function(x, x_hat, mean, log_var):
        # sse loss
        reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
        reproduction_loss *= 1 * x.shape[1]
        
        # KL Divergence loss
        kld_loss = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

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

    # def _prog_dict(self, loss_g, loss_d, lr_g, lr_d):
    #     return {
    #         "Loss(d,g)": f"[{loss_d:.2E}, {loss_g:.2E}]",
    #         "LR(d,g)": f"[{lr_d:.2E}, {lr_g:.2E}]",
    #     }

    def train(self):
        t0 = time.time()

        train_loader = self.training_dataset.get_loader(self.batch_size)
        val_loader = self.validation_dataset.get_loader(self.batch_size)

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

            for batch_idx, (signal, params) in enumerate(train_loader):
                signal = signal.view(signal.size(0), -1).to(DEVICE)
                params = params.view(params.size(0), -1).to(DEVICE)

                self.optimizerVAE.zero_grad()
                recon, mean, log_var = self.vae(signal)
                loss, rec_loss, kld = Trainer.loss_function(signal, recon, mean, log_var)
                loss.backward()
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
                for val_signal, val_params in val_loader:
                    val_signal = val_signal.view(val_signal.size(0), -1).to(DEVICE)
                    val_params = val_params.view(val_params.size(0), -1).to(DEVICE)
                    recon, mean, log_var = self.vae(val_signal)
                    v_loss, v_rec_loss, v_kld = Trainer.loss_function(val_signal, recon, mean, log_var)
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

            print(f"Epoch {epoch+1}/{self.num_epochs} | Train Loss: {avg_total_loss:.4f} | Val Loss: {avg_total_loss_val:.4f}")

            # Optionally: add plotting or checkpointing here
            if (epoch + 1) % self.checkpoint_interval == 0:
                # gridded plots
                with torch.no_grad():
                    generated_signals = self.vae.decoder(self.fixed_noise).cpu().detach().numpy()
                
                plot_waveform_grid(signals=generated_signals, max_value=self.training_dataset.max_strain)
                plot_latent_space_3d(
                    model=self.vae,
                    dataloader=train_loader
                )
                

        runtime = (time.time() - t0) / 60
        print(f"Training Time: {runtime:.2f}min")
        # Optionally: plot final results or save model
        self.save_models()

    def plot_generated_signal_distribution(self, background, font_family, font_name, fname):
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

    def display_results(self):
        # Plot training and validation losses
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        plot_individual_loss(
            self.avg_total_losses, self.avg_reproduction_losses, self.avg_kld_losses
        )
        plot_individual_loss(
            self.avg_total_losses_val, self.avg_reproduction_losses_val, self.avg_kld_losses_val
        )
        plot_loss(self.avg_total_losses, self.avg_total_losses_val)
        plt.tight_layout()
        plt.show()
        

    @property
    def save_fname(self):
        return f"{self.outdir}/generator_weights.pt"

    def save_models(self):
        torch.save(self.vae.state_dict(), self.save_fname)
        print(f"Saved VAE model to {self.save_fname}")


# class TrainMetadata:
#     def __init__(self):
#         self.iter: List[int] = []
#         self.g_loss: List[float] = []
#         self.d_loss: List[float] = []
#         self.g_gradient: List[float] = []
#         self.d_gradient: List[float] = []

#     def append(self, iter, g_loss, d_loss, g_gradient, d_gradient):
#         self.iter.append(iter)
#         self.g_loss.append(g_loss)
#         self.d_loss.append(d_loss)
#         self.g_gradient.append(g_gradient)
#         self.d_gradient.append(d_gradient)

#     def plot(self, fname="training_metrics.png"):
#         fig, axes = plt.subplots(3, 1, figsize=(10, 6))
#         plot_gradients(
#             self.d_gradient, "tab:red", "Discriminator", axes=axes[0]
#         )
#         plot_gradients(self.g_gradient, "tab:blue", "Generator", axes=axes[1])
#         plot_loss(self.g_loss, self.d_loss, axes=axes[2])
#         plt.tight_layout()
#         plt.savefig(fname)

def _init_weights_vae(m: torch.nn.Module) -> None:
    """This function initialises the weights of the model."""
    if type(m) == torch.nn.Conv1d or type(m) == torch.nn.ConvTranspose1d:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if type(m) == torch.nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

### TODO: alter this for flows
def _init_weights_flow(m: torch.nn.Module) -> None:
    """This function initialises the weights of the model."""
    if type(m) == torch.nn.Conv1d or type(m) == torch.nn.ConvTranspose1d:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if type(m) == torch.nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def train(**kwargs):
    trainer = Trainer(**kwargs)
    trainer.train()
    return trainer
