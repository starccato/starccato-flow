import unittest
import numpy as np
import torch

from starccato_flow.data.ccsn_data import CCSNData


class TestNoiseRealizations(unittest.TestCase):
    """Test suite for noise realization feature in CCSNData."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.noise_realizations_1 = 1
        self.noise_realizations_3 = 3
        self.noise_realizations_5 = 5
    
    def test_dataset_size_multiplier(self):
        """Test that dataset size is correctly multiplied by noise_realizations."""
        # Create datasets with different noise realizations
        dataset_1 = CCSNData(noise=True, curriculum=True, noise_realizations=1)
        dataset_3 = CCSNData(noise=True, curriculum=True, noise_realizations=3)
        dataset_5 = CCSNData(noise=True, curriculum=True, noise_realizations=5)
        
        base_size = len(dataset_1)
        
        # Verify size multiplication
        self.assertEqual(len(dataset_3), base_size * 3, 
                        f"Dataset with 3 realizations should be 3x base size")
        self.assertEqual(len(dataset_5), base_size * 5,
                        f"Dataset with 5 realizations should be 5x base size")
        
        print(f"✓ Dataset size test passed:")
        print(f"  - 1 realization: {len(dataset_1)} samples")
        print(f"  - 3 realizations: {len(dataset_3)} samples")
        print(f"  - 5 realizations: {len(dataset_5)} samples")
    
    def test_different_noise_per_realization(self):
        """Test that different realizations produce different noise."""
        dataset = CCSNData(noise=True, curriculum=True, noise_realizations=3)
        base_size = len(dataset) // 3
        
        # Get the same signal with different noise realizations
        signal_0_real_0 = dataset[0][1]  # First signal, first realization
        signal_0_real_1 = dataset[base_size][1]  # First signal, second realization
        signal_0_real_2 = dataset[2 * base_size][1]  # First signal, third realization
        
        # Calculate differences
        diff_0_1 = torch.abs(signal_0_real_0 - signal_0_real_1).mean().item()
        diff_1_2 = torch.abs(signal_0_real_1 - signal_0_real_2).mean().item()
        diff_0_2 = torch.abs(signal_0_real_0 - signal_0_real_2).mean().item()
        
        # Noise should be different (mean absolute difference > 0)
        self.assertGreater(diff_0_1, 0.0, "Realization 0 and 1 should have different noise")
        self.assertGreater(diff_1_2, 0.0, "Realization 1 and 2 should have different noise")
        self.assertGreater(diff_0_2, 0.0, "Realization 0 and 2 should have different noise")
        
        # Differences should be significant (normalized signals, so expect > 0.001)
        self.assertGreater(diff_0_1, 0.001, "Noise difference should be significant")
        self.assertGreater(diff_1_2, 0.001, "Noise difference should be significant")
        self.assertGreater(diff_0_2, 0.001, "Noise difference should be significant")
        
        print(f"✓ Different noise per realization test passed:")
        print(f"  - Real 0 vs Real 1 diff: {diff_0_1:.6f}")
        print(f"  - Real 1 vs Real 2 diff: {diff_1_2:.6f}")
        print(f"  - Real 0 vs Real 2 diff: {diff_0_2:.6f}")
    
    def test_same_clean_signal_across_realizations(self):
        """Test that the underlying clean signal is the same across realizations."""
        dataset = CCSNData(noise=True, curriculum=True, noise_realizations=3)
        base_size = len(dataset) // 3
        
        # Get the clean signal (first element of tuple) for different realizations
        clean_signal_real_0 = dataset[0][0]  # First signal, first realization (clean)
        clean_signal_real_1 = dataset[base_size][0]  # First signal, second realization (clean)
        clean_signal_real_2 = dataset[2 * base_size][0]  # First signal, third realization (clean)
        
        # Clean signals should be identical
        self.assertTrue(torch.allclose(clean_signal_real_0, clean_signal_real_1, atol=1e-6),
                       "Clean signals should be identical across realizations")
        self.assertTrue(torch.allclose(clean_signal_real_1, clean_signal_real_2, atol=1e-6),
                       "Clean signals should be identical across realizations")
        
        print(f"✓ Same clean signal test passed - all realizations use same underlying signal")
    
    def test_same_parameters_across_realizations(self):
        """Test that parameters are the same across realizations of the same signal."""
        dataset = CCSNData(noise=True, curriculum=True, noise_realizations=3)
        base_size = len(dataset) // 3
        
        # Get parameters for different realizations of the same signal
        params_real_0 = dataset[0][2]  # First signal, first realization
        params_real_1 = dataset[base_size][2]  # First signal, second realization
        params_real_2 = dataset[2 * base_size][2]  # First signal, third realization
        
        # Parameters should be identical
        self.assertTrue(torch.allclose(params_real_0, params_real_1, atol=1e-6),
                       "Parameters should be identical across realizations")
        self.assertTrue(torch.allclose(params_real_1, params_real_2, atol=1e-6),
                       "Parameters should be identical across realizations")
        
        print(f"✓ Same parameters test passed")
        print(f"  Parameters: {params_real_0.cpu().numpy()}")
    
    def test_index_mapping(self):
        """Test that index mapping correctly cycles through signals."""
        dataset = CCSNData(noise=True, curriculum=True, noise_realizations=3)
        base_size = len(dataset) // 3
        
        # Test that indices map correctly
        # Indices 0, base_size, 2*base_size should all map to original signal 0
        # Indices 1, base_size+1, 2*base_size+1 should all map to original signal 1
        
        signal_0_params = [dataset[0][2], dataset[base_size][2], dataset[2*base_size][2]]
        signal_1_params = [dataset[1][2], dataset[base_size+1][2], dataset[2*base_size+1][2]]
        
        # All realizations of signal 0 should have same parameters
        for i in range(1, len(signal_0_params)):
            self.assertTrue(torch.allclose(signal_0_params[0], signal_0_params[i]),
                           f"Signal 0 realization {i} has different parameters")
        
        # All realizations of signal 1 should have same parameters
        for i in range(1, len(signal_1_params)):
            self.assertTrue(torch.allclose(signal_1_params[0], signal_1_params[i]),
                           f"Signal 1 realization {i} has different parameters")
        
        # Signal 0 and signal 1 should have different parameters
        self.assertFalse(torch.allclose(signal_0_params[0], signal_1_params[0]),
                        "Signal 0 and signal 1 should have different parameters")
        
        print(f"✓ Index mapping test passed - indices correctly map to original signals")
    
    def test_reproducibility(self):
        """Test that noise is reproducible with the same realization index."""
        dataset1 = CCSNData(noise=True, curriculum=True, noise_realizations=3)
        dataset2 = CCSNData(noise=True, curriculum=True, noise_realizations=3)
        
        # Get the same signal from both datasets
        signal1_real0 = dataset1[0][1]
        signal2_real0 = dataset2[0][1]
        
        signal1_real1 = dataset1[len(dataset1)//3][1]
        signal2_real1 = dataset2[len(dataset2)//3][1]
        
        # Same realization should produce the same noise
        self.assertTrue(torch.allclose(signal1_real0, signal2_real0, atol=1e-6),
                       "Same realization should be reproducible across dataset instances")
        self.assertTrue(torch.allclose(signal1_real1, signal2_real1, atol=1e-6),
                       "Same realization should be reproducible across dataset instances")
        
        print(f"✓ Reproducibility test passed - noise is deterministic for each realization")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
