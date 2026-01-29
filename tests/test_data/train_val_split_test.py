"""Tests for train/validation split to ensure no data leakage."""

import numpy as np
from starccato_flow.training import create_train_val_split
from starccato_flow.utils.defaults import Y_LENGTH


class TestTrainValSplit:
    """Test suite for train/validation split functionality."""
    
    def test_no_overlap_ccsn_data(self):
        """Verify no overlap between train and validation sets for CCSN data."""
        # Create train/val split
        training_dataset, validation_dataset = create_train_val_split(
            toy=False,
            y_length=Y_LENGTH,
            noise=True,
            validation_split=0.1,
            seed=42,
            num_epochs=256,
            start_snr=200,
            end_snr=10,
            curriculum=True,
            noise_realizations=1
        )
        
        # Get the base signal indices used by each dataset
        train_indices = set(training_dataset.indices)
        val_indices = set(validation_dataset.indices)
        
        # Check for overlap
        overlap = train_indices.intersection(val_indices)
        
        # Assert no overlap
        assert len(overlap) == 0, (
            f"Data leakage detected! {len(overlap)} signals appear in both "
            f"train and validation sets: {sorted(list(overlap))[:10]}"
        )
        
        # Verify coverage
        total_signals = len(train_indices) + len(val_indices)
        print(f"\n✓ No overlap verified:")
        print(f"  Training signals: {len(train_indices)}")
        print(f"  Validation signals: {len(val_indices)}")
        print(f"  Total coverage: {total_signals}")
    
    def test_no_overlap_toy_data(self):
        """Verify no overlap between train and validation sets for toy data."""
        # Create train/val split
        training_dataset, validation_dataset = create_train_val_split(
            toy=True,
            y_length=Y_LENGTH,
            noise=True,
            validation_split=0.1,
            seed=42,
            num_epochs=256,
            start_snr=200,
            end_snr=10,
            curriculum=False,
            noise_realizations=1
        )
        
        # For toy data, we need to check if parameters overlap
        # Convert to tuples for set comparison
        train_params = set(map(tuple, training_dataset.parameters))
        val_params = set(map(tuple, validation_dataset.parameters))
        
        # Check for overlap
        overlap = train_params.intersection(val_params)
        
        # Assert no overlap
        assert len(overlap) == 0, (
            f"Data leakage detected! {len(overlap)} parameter sets appear in both "
            f"train and validation sets"
        )
        
        # Verify coverage
        total_signals = len(train_params) + len(val_params)
        print(f"\n✓ No overlap verified:")
        print(f"  Training signals: {len(train_params)}")
        print(f"  Validation signals: {len(val_params)}")
        print(f"  Total coverage: {total_signals}")
    
    def test_split_ratio_ccsn(self):
        """Verify the split ratio is approximately correct for CCSN data."""
        validation_split = 0.15
        
        training_dataset, validation_dataset = create_train_val_split(
            toy=False,
            y_length=Y_LENGTH,
            noise=True,
            validation_split=validation_split,
            seed=42,
            num_epochs=256,
            start_snr=200,
            end_snr=10,
            curriculum=True,
            noise_realizations=1
        )
        
        train_size = len(training_dataset.indices)
        val_size = len(validation_dataset.indices)
        total_size = train_size + val_size
        
        actual_val_ratio = val_size / total_size
        
        # Allow 1% tolerance
        assert abs(actual_val_ratio - validation_split) < 0.01, (
            f"Split ratio incorrect: expected {validation_split}, got {actual_val_ratio}"
        )
        
        print(f"\n✓ Split ratio verified:")
        print(f"  Expected: {validation_split}")
        print(f"  Actual: {actual_val_ratio:.4f}")
    
    def test_split_ratio_toy(self):
        """Verify the split ratio is approximately correct for toy data."""
        validation_split = 0.15
        
        training_dataset, validation_dataset = create_train_val_split(
            toy=True,
            y_length=Y_LENGTH,
            noise=True,
            validation_split=validation_split,
            seed=42,
            num_epochs=256,
            start_snr=200,
            end_snr=10,
            curriculum=False,
            noise_realizations=1
        )
        
        train_size = training_dataset.num_signals
        val_size = validation_dataset.num_signals
        total_size = train_size + val_size
        
        actual_val_ratio = val_size / total_size
        
        # Allow 1% tolerance
        assert abs(actual_val_ratio - validation_split) < 0.01, (
            f"Split ratio incorrect: expected {validation_split}, got {actual_val_ratio}"
        )
        
        print(f"\n✓ Split ratio verified:")
        print(f"  Expected: {validation_split}")
        print(f"  Actual: {actual_val_ratio:.4f}")
    
    def test_deterministic_split(self):
        """Verify that using the same seed produces the same split."""
        seed = 123
        
        # Create first split
        train1, val1 = create_train_val_split(
            toy=False,
            y_length=Y_LENGTH,
            noise=True,
            validation_split=0.1,
            seed=seed,
            num_epochs=256,
            start_snr=200,
            end_snr=10,
            curriculum=True,
            noise_realizations=1
        )
        
        # Create second split with same seed
        train2, val2 = create_train_val_split(
            toy=False,
            y_length=Y_LENGTH,
            noise=True,
            validation_split=0.1,
            seed=seed,
            num_epochs=256,
            start_snr=200,
            end_snr=10,
            curriculum=True,
            noise_realizations=1
        )
        
        # Verify indices are identical
        assert np.array_equal(train1.indices, train2.indices), (
            "Training indices differ with same seed"
        )
        assert np.array_equal(val1.indices, val2.indices), (
            "Validation indices differ with same seed"
        )
        
        print("\n✓ Deterministic split verified: same seed produces same indices")
    
    def test_different_seeds_produce_different_splits(self):
        """Verify that different seeds produce different splits."""
        # Create split with seed 1
        train1, val1 = create_train_val_split(
            toy=False,
            y_length=Y_LENGTH,
            noise=True,
            validation_split=0.1,
            seed=1,
            num_epochs=256,
            start_snr=200,
            end_snr=10,
            curriculum=True,
            noise_realizations=1
        )
        
        # Create split with seed 2
        train2, val2 = create_train_val_split(
            toy=False,
            y_length=Y_LENGTH,
            noise=True,
            validation_split=0.1,
            seed=2,
            num_epochs=256,
            start_snr=200,
            end_snr=10,
            curriculum=True,
            noise_realizations=1
        )
        
        # Verify indices are different
        assert not np.array_equal(train1.indices, train2.indices), (
            "Training indices should differ with different seeds"
        )
        assert not np.array_equal(val1.indices, val2.indices), (
            "Validation indices should differ with different seeds"
        )
        
        print("\n✓ Different seeds produce different splits")
