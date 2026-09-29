"""Tests for precision-escalation quantization module."""

import numpy as np
import pytest
import sys

# Add parent directory to path
sys.path.insert(0, str(__file__).replace('test_quant.py', ''))

from pe.quant import quantize_and_store, reconstruct_from_storage


def test_reconstruction_error_decreases_with_bits():
    """Test that reconstruction error decreases as bit precision increases."""
    # Generate random weight matrix
    np.random.seed(42)
    weights = np.random.randn(1024, 256).astype(np.float32)
    
    # Quantize
    data = quantize_and_store(weights)
    
    # Compute relative error for each precision
    errors = {}
    for bits in [2, 3, 4, 8]:
        reconstructed = reconstruct_from_storage(data, bits)
        error = np.abs(weights - reconstructed).mean() / (np.abs(weights).mean() + 1e-8)
        errors[bits] = error
    
    # Errors should decrease as bits increase
    assert errors[2] > errors[3], f"2-bit error ({errors[2]:.4f}) should be > 3-bit error ({errors[3]:.4f})"
    assert errors[3] > errors[4], f"3-bit error ({errors[3]:.4f}) should be > 4-bit error ({errors[4]:.4f})"
    assert errors[4] > errors[8], f"4-bit error ({errors[4]:.4f}) should be > 8-bit error ({errors[8]:.4f})"


def test_8bit_error_under_1_percent():
    """Test that 8-bit reconstruction error is under 1% relative."""
    np.random.seed(42)
    weights = np.random.randn(1024, 256).astype(np.float32)
    
    data = quantize_and_store(weights)
    reconstructed = reconstruct_from_storage(data, 8)
    
    # Relative error
    error = np.abs(weights - reconstructed).mean() / (np.abs(weights).mean() + 1e-8)
    
    assert error < 0.01, f"8-bit relative error {error:.4f} should be under 1%"


def test_shape_preservation():
    """Test that reconstructed weights have same shape as original."""
    np.random.seed(42)
    weights = np.random.randn(512, 128).astype(np.float32)
    
    data = quantize_and_store(weights)
    
    for bits in [2, 3, 4, 8]:
        reconstructed = reconstruct_from_storage(data, bits)
        assert reconstructed.shape == weights.shape, f"Shape mismatch for {bits}-bit: {reconstructed.shape} vs {weights.shape}"


def test_2bit_base_only():
    """Test that 2-bit reconstruction returns base values."""
    np.random.seed(42)
    weights = np.random.randn(256, 64).astype(np.float32)
    
    data = quantize_and_store(weights)
    reconstructed = reconstruct_from_storage(data, 2)
    
    # Should have much coarser quantization
    unique_values = np.unique(reconstructed)
    # 2-bit should give us roughly 4 levels per group
    assert len(unique_values) < weights.size * 0.1, "2-bit reconstruction should have coarse quantization"


def test_group_wise_scaling():
    """Test group-wise scaling preserves statistics."""
    np.random.seed(42)
    weights = np.random.randn(512, 128).astype(np.float32)
    
    data = quantize_and_store(weights)
    
    # Check that scales and zeros have expected shape
    expected_groups = (512 * 128 + 63) // 64  # Group size = 64
    assert data['scales'].shape[0] == expected_groups, "Expected {expected_groups} scale groups"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
