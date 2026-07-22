"""Training utilities and shared functions."""

import time
import numpy as np
import torch

from ..data.s_theta import sTheta
from ..plotting import plot_signal_distribution
from ..utils.defaults_general import Y_LENGTH, Z_DIM, DEVICE

def create_train_val_split(
    y_length: int,
    detector_noise_on: bool,
    validation_split: float,
    seed: int,
    num_epochs: int,
    parameters: list = None,
):
    """Create training and validation datasets with proper splitting.
    
    Args:
        y_length: Length of signal
        detector_noise_on: Whether to add detector noise
        validation_split: Fraction of data for validation
        seed: Random seed for reproducible split
        num_epochs: Number of training epochs
        parameters: List of parameter names to estimate. If None, defaults to
                    ["beta_ic_b", "ra", "dec", "d", "psi"]
        
    Returns:
        tuple: (training_dataset, validation_dataset, val_indices)
    """
    if parameters is None:
        parameters = ["beta_ic_b", "ra", "dec", "d", "psi"]
    
    # Filter parameters to only include those available in sTheta (intrinsic CCSN parameters)
    # Sky parameters (ra, dec, d, psi) are added later by hThetaMulti
    intrinsic_params = {"beta1_IC_b", "omega_0(rad|s)", "A(km)", "Ye_c_b"}
    stheta_parameters = [p for p in parameters if p in intrinsic_params]
    # If user requests ONLY sky parameters, stheta_parameters will be empty
    # hThetaMulti will then only output sky parameters
    

    full_dataset = sTheta(
        num_epochs=num_epochs,
        detector_noise_on=False,
        parameters=stheta_parameters,
    )
    num_signals = full_dataset.signals.shape[1]
    
    # Split signal indices
    indices = list(range(num_signals))
    split = int(np.floor(validation_split * num_signals))
    if num_signals > 1:
        split = max(1, min(split, num_signals - 1))
    
    # Deterministic split with fixed seed
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    train_indices = np.array(indices[split:])
    val_indices = np.array(indices[:split])
    
    print(f"\n=== Data Split ===")
    print(f"Total signals: {num_signals}")
    print(f"Training signals: {len(train_indices)}")
    print(f"Validation signals: {len(val_indices)}")
    print(f"First 5 training indices: {train_indices[:5]}")
    print(f"First 5 validation indices: {val_indices[:5]}")
    
    # Create SEPARATE dataset instances with disjoint indices
    # Note: sTheta should always have detector_noise_on=False (noise added only in hThetaMulti)
    training_dataset = sTheta(
        num_epochs=num_epochs,
        detector_noise_on=False,
        parameters=stheta_parameters,
        indices=train_indices,
        shared_min=full_dataset.shared_min_theta,
        shared_max=full_dataset.shared_max_theta,
        shared_max_strain=full_dataset.shared_max_strain
    )
    
    validation_dataset = sTheta(
        num_epochs=num_epochs,
        detector_noise_on=False,
        parameters=stheta_parameters,
        indices=val_indices,
        shared_min=full_dataset.shared_min_theta,
        shared_max=full_dataset.shared_max_theta,
        shared_max_strain=full_dataset.shared_max_strain
    )
    
    # Verify alignment
    training_dataset.verify_alignment()
    validation_dataset.verify_alignment()
    
    return full_dataset, training_dataset, validation_dataset, val_indices

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