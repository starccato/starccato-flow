"""Training utilities and shared functions."""

import time
import numpy as np
import torch

from ..data.toy_data import ToyData
from ..data.ccsn_data import CCSNData
from ..plotting import plot_signal_distribution, plot_candidate_signal, plot_loss, plot_signal_grid, plot_latent_space_3d
from ..utils.defaults import Y_LENGTH, Z_DIM, DEVICE, TEN_KPC


def create_train_val_split(
    toy: bool,
    y_length: int,
    noise: bool,
    validation_split: float,
    seed: int,
    num_epochs: int,
    start_snr: int,
    end_snr: int,
    curriculum: bool,
    noise_realizations: int
):
    """Create training and validation datasets with proper splitting.
    
    Args:
        toy: Whether to use toy dataset or CCSN data
        y_length: Length of signal
        noise: Whether to add noise
        validation_split: Fraction of data for validation
        seed: Random seed for reproducible split
        num_epochs: Number of training epochs
        start_snr: Starting SNR for curriculum
        end_snr: Ending SNR for curriculum
        curriculum: Whether to use curriculum learning
        noise_realizations: Number of noise realizations per signal
        
    Returns:
        tuple: (training_dataset, validation_dataset)
    """
    if toy:
        # Create full toy dataset
        full_toy_dataset = ToyData(
            num_signals=1684, 
            signal_length=y_length, 
            noise=noise
        )
        
        # Split toy data using same logic as real data
        num_signals = full_toy_dataset.num_signals
        base_indices = list(range(num_signals))
        split = int(np.floor(validation_split * num_signals))
        
        # Deterministic split with fixed seed
        rng = np.random.RandomState(seed)
        rng.shuffle(base_indices)
        train_indices = base_indices[split:]
        val_indices = base_indices[:split]
        
        print(f"\n=== Toy Data Split ===")
        print(f"Total signals: {num_signals}")
        print(f"Training signals: {len(train_indices)}")
        print(f"Validation signals: {len(val_indices)}")
        
        # Create subsets using shared parameter ranges from full dataset
        training_dataset = ToyData(
            num_signals=len(train_indices),
            signal_length=y_length,
            noise=noise,
            shared_params=full_toy_dataset.parameters[train_indices],
            shared_min=full_toy_dataset.min_parameter,
            shared_max=full_toy_dataset.max_parameter,
            shared_max_strain=full_toy_dataset.max_strain
        )
        validation_dataset = ToyData(
            num_signals=len(val_indices),
            signal_length=y_length,
            noise=noise,
            shared_params=full_toy_dataset.parameters[val_indices],
            shared_min=full_toy_dataset.min_parameter,
            shared_max=full_toy_dataset.max_parameter,
            shared_max_strain=full_toy_dataset.max_strain
        )
    else:
        # Create a temporary dataset to get the number of base signals (before augmentation)
        temp_dataset = CCSNData(
            num_epochs=num_epochs,
            start_snr=start_snr,
            end_snr=end_snr,
            noise=noise,
            curriculum=False, 
            noise_realizations=1
        )
        num_base_signals = temp_dataset.signals.shape[1]
        
        # Split on BASE signal indices (before augmentation)
        base_indices = list(range(num_base_signals))
        split = int(np.floor(validation_split * num_base_signals))
        
        # Deterministic split with fixed seed
        rng = np.random.RandomState(seed)
        rng.shuffle(base_indices)
        train_base_indices = np.array(base_indices[split:])
        val_base_indices = np.array(base_indices[:split])
        
        print(f"\n=== Data Split (on base signals) ===")
        print(f"Total base signals: {num_base_signals}")
        print(f"Training base signals: {len(train_base_indices)}")
        print(f"Validation base signals: {len(val_base_indices)}")
        print(f"First 5 training indices: {train_base_indices[:5]}")
        print(f"First 5 validation indices: {val_base_indices[:5]}")
        
        # Create SEPARATE dataset instances with disjoint base indices
        # Training: with curriculum and multiple noise realizations
        training_dataset = CCSNData(
            num_epochs=num_epochs,
            start_snr=start_snr,
            end_snr=end_snr,
            noise=noise,
            curriculum=curriculum,
            noise_realizations=noise_realizations,
            indices=train_base_indices,
            shared_min=temp_dataset.min_parameter,
            shared_max=temp_dataset.max_parameter,
            shared_max_strain=temp_dataset.max_strain
        )
        
        # Validation: FIXED SNR (no curriculum) with single noise realization
        validation_dataset = CCSNData(
            num_epochs=num_epochs,
            start_snr=end_snr,
            end_snr=end_snr,
            noise=noise,
            curriculum=curriculum, 
            noise_realizations=1,
            indices=val_base_indices,
            shared_min=temp_dataset.min_parameter,
            shared_max=temp_dataset.max_parameter,
            shared_max_strain=temp_dataset.max_strain
        )
    
    # Verify alignment
    training_dataset.verify_alignment()
    validation_dataset.verify_alignment()
    
    # Return datasets along with validation indices for later use
    if toy:
        return training_dataset, validation_dataset, val_indices
    else:
        return training_dataset, validation_dataset, val_base_indices


def plot_generated_signal_distribution(
    vae,
    training_dataset,
    background="white",
    font_family="serif",
    font_name="Times New Roman",
    fname=None,
    number_of_signals=10000
):
    """Plot distribution of VAE-generated signals.
    
    Args:
        vae: VAE model with decoder
        training_dataset: Dataset to get denormalization parameters
        background: Plot background color
        font_family: Font family for plot
        font_name: Font name for plot
        fname: Filename to save plot
        number_of_signals: Number of signals to generate
    """
    noise = torch.randn(number_of_signals, Z_DIM).to(DEVICE)

    start_time = time.time()
    with torch.no_grad():
        generated_signals = vae.decoder(noise).cpu().detach().numpy()
    end_time = time.time()

    execution_time = end_time - start_time
    print("Execution Time:", execution_time, "seconds")    

    generated_signals_transpose = np.empty((Y_LENGTH, 0))

    for i in range(number_of_signals):
        y = generated_signals[i, :].flatten()
        y = training_dataset.denormalise_signals(y)
        y = y.reshape(-1, 1)
        
        generated_signals_transpose = np.concatenate((generated_signals_transpose, y), axis=1)

    plot_signal_distribution(
        signals=generated_signals_transpose,
        generated=True,
        background=background,
        font_family=font_family,
        font_name=font_name,
        fname=fname
    )


def plot_candidate_signal_method(
    val_loader,
    snr=100,
    background="white",
    index=0,
    fname="plots/candidate_signal.png"
):
    """Plot a candidate signal with noise.
    
    Args:
        val_loader: Validation data loader
        snr: Signal-to-noise ratio
        background: Plot background color
        index: Index of signal to plot
        fname: Filename to save plot
    """
    val_loader.dataset.update_snr(snr)
    signal, noisy_signal, _ = val_loader.dataset.__getitem__(index)
    signal_denorm = val_loader.dataset.denormalise_signals(signal) / TEN_KPC
    noisy_signal_denorm = val_loader.dataset.denormalise_signals(noisy_signal) / TEN_KPC
    plot_candidate_signal(
        signal=signal_denorm,
        noisy_signal=noisy_signal_denorm,
        max_value=val_loader.dataset.max_strain,
        background=background,
        fname=fname
    )


def display_results_method(avg_mse_losses, avg_mse_losses_val, background="black"):
    """Display training results with loss plots.
    
    Args:
        avg_mse_losses: List of average training MSE losses per epoch
        avg_mse_losses_val: List of average validation MSE losses per epoch
        background: Plot background color
    """
    plot_loss(avg_mse_losses, avg_mse_losses_val, background=background)
