"""
Quantization module for precision-escalation.

Splits weight matrices into a 2-bit base plus residual bit-planes,
allowing reconstruction at 2, 3, 4, and 8 bits effective precision.
"""

import numpy as np


def quantize_and_store(weights: np.ndarray, 
                       group_size: int = 64,
                       max_bits: int = 8) -> dict:
    """
    Full quantization pipeline: quantize weights into base + residuals.
    
    Quantizes weights to 2^max_bits levels per group, then encodes
    as a 2-bit base plus (max_bits - 2) residual bit-planes.
    
    Args:
        weights: Original weight matrix
        group_size: Group size for scaling
        max_bits: Maximum target precision (default 8)
        
    Returns:
        Dict with base_2bit, step, min_val, bit_planes, and metadata
    """
    original_shape = weights.shape
    total_elements = original_shape[0] * original_shape[1]
    
    # Flatten and pad to group_size multiple
    flat = weights.ravel().astype(np.float32)
    n_groups = (total_elements + group_size - 1) // group_size
    padded_size = n_groups * group_size
    
    if flat.shape[0] < padded_size:
        flat = np.pad(flat, (0, padded_size - flat.shape[0]), mode='edge')
    
    groups = flat.reshape(n_groups, group_size)
    
    # Per-group min/max for quantization (2^max_bits levels per group)
    group_min = groups.min(axis=1, keepdims=True)
    group_max = groups.max(axis=1, keepdims=True)
    group_range = group_max - group_min
    group_range[group_range == 0] = 1.0  # Avoid division by zero
    
    # Quantization step for full precision
    n_levels = 1 << max_bits  # 2^max_bits
    step = group_range / (n_levels - 1)
    
    # Scales for 2-bit base (same as group_range / 3)
    scales = group_range / 3.0
    
    # Quantize to max_bits levels (0 to n_levels-1)
    quantized = np.clip(np.round((groups - group_min) / step), 0, n_levels - 1).astype(np.uint16)
    
    # Extract 2-bit base (most significant 2 bits)
    base_2bit = (quantized >> (max_bits - 2)).astype(np.uint8)
    
    # Extract residual bits (remaining max_bits - 2 bits)
    residual_bits = max_bits - 2
    residual_mask = (1 << residual_bits) - 1
    residuals = quantized & residual_mask
    
    # Encode residuals as bit-planes
    bit_planes = []
    for i in range(residual_bits):
        bit_plane = ((residuals >> i) & 1).astype(np.uint8)
        bit_planes.append(bit_plane)
    
    return {
        'base_2bit': base_2bit,
        'step': step,
        'scales': scales,
        'min_val': group_min,
        'bit_planes': bit_planes,
        'group_size': group_size,
        'n_groups': n_groups,
        'original_shape': original_shape,
        'total_elements': total_elements,
        'max_bits': max_bits,
        'residual_bits': residual_bits,
        'n_levels': n_levels
    }


def reconstruct_from_storage(data: dict, target_bits: int) -> np.ndarray:
    """
    Reconstruct weights from quantized storage at target precision.
    
    Args:
        data: Output from quantize_and_store
        target_bits: Target precision (2, 3, 4, or 8)
        
    Returns:
        Reconstructed weight matrix
    """
    if target_bits < 2:
        raise ValueError("target_bits must be >= 2")
    
    base_2bit = data['base_2bit']
    step = data['step']
    min_val = data['min_val']
    bit_planes = data['bit_planes']
    original_shape = data['original_shape']
    total_elements = data['total_elements']
    residual_bits = data['residual_bits']
    
    # Progressive precision: base (top 2 bits) plus the most significant residual bits first
    k = min(max(target_bits - 2, 0), residual_bits)
    quantized = base_2bit.astype(np.uint16) << residual_bits
    for j in range(k):
        i = residual_bits - 1 - j  # plane i holds bit i; take MSB first
        quantized |= bit_planes[i].astype(np.uint16) << i
    if k < residual_bits:
        # centre the missing low bits to halve the expected error
        quantized |= np.uint16(1 << (residual_bits - k - 1))

    # Dequantize to float
    reconstructed = min_val + quantized.astype(np.float32) * step
    
    return reconstructed.ravel()[:total_elements].reshape(original_shape)
