"""Training utilities and shared functions."""

import numpy as np
from ..data.toy_data import ToyData
from ..data.ccsn_data import CCSNData


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
    
    return training_dataset, validation_dataset
