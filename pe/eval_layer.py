"""
Evaluate layer quantization error for a Qwen3-8B-like weight matrix.

Since no real Qwen3-8B weights are available locally, we create a synthetic
weight matrix with similar dimensions (e.g., a QKV projection layer).
"""
import os
import sys
import json
import numpy as np

# Add pe directory to path
sys.path.insert(0, os.path.dirname(__file__))
from quant import quantize_and_store, reconstruct_from_storage


def create_synthetic_qwen_layer():
    """
    Create a synthetic weight matrix approximating a Qwen3-8B layer.
    Qwen3-8B has hidden_size=2048, typical projection layers are 2048x2048
    or similar. We use a 2048x2048 matrix.
    """
    # Typical transformer layer dimension
    hidden_size = 2048
    
    # Create a random weight matrix similar to actual transformer weights
    # Uses small random values like initialized weights (often ~0.02 scale)
    rng = np.random.RandomState(42)
    weights = rng.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
    
    return weights


def compute_relative_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute relative reconstruction error (Frobenius norm)."""
    diff = original - reconstructed
    return float(np.linalg.norm(diff) / np.linalg.norm(original))


def main():
    # Create synthetic Qwen3-8B-like layer
    weights = create_synthetic_qwen_layer()
    print(f"Created synthetic layer with shape: {weights.shape}")
    
    # Quantize at full precision (8-bit equivalent)
    quant_data = quantize_and_store(weights, group_size=64, max_bits=8)
    print(f"Quantization complete: {quant_data['total_elements']} elements")
    
    # Test reconstruction at different bit widths
    results = {}
    for bits in [2, 4, 6, 8]:
        reconstructed = reconstruct_from_storage(quant_data, target_bits=bits)
        error = compute_relative_error(weights, reconstructed)
        results[str(bits)] = error
        print(f"  {bits}-bit: relative error = {error:.6f}")
    
    # Write results
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "layer_error.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results written to: {output_path}")
    return results


if __name__ == "__main__":
    main()
