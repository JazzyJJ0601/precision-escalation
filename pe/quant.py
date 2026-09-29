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
    
    Args:
        weights: Original weight matrix
        group_size: Group size for scaling
        max_bits: Maximum target precision (default 8)
        
    Returns:
        Dict with base_2bit, scales, zeros, bit_planes, and metadata
    """
    original_shape = weights.shape
    total_elements = original_shape[0] * original_shape[1]
    
    # Flatten and pad
    flat = weights.ravel().astype(np.float32)
    n_groups = (total_elements + group_size - 1) // group_size
    padded_size = n_groups * group_size
    
    if flat.shape[0] < padded_size:
        flat = np.pad(flat, (0, padded_size - flat.shape[0]), mode='edge')
    
    groups = flat.reshape(n_groups, group_size)
    
    # 2-bit base quantization per group (4 levels: 0, 1, 2, 3)
    group_min = groups.min(axis=1, keepdims=True)
    group_max = groups.max(axis=1, keepdims=True)
    group_range = group_max - group_min
    group_range[group_range == 0] = 1.0
    
    scales = group_range / 3.0
    zeros = group_min
    
    scaled = (groups - zeros) / scales
    base_2bit_groups = np.clip(np.round(scaled), 0, 3).astype(np.uint8)
    
    # Compute residuals (original - base)
    residuals = groups - (base_2bit_groups.astype(np.float32) * scales + zeros)
    
    # Encode residuals into bit-planes (signed, two's complement)
    residual_bits = max_bits - 2  # 6 bits for 8-bit total
    
    # Scale residuals to fit in signed residual_bits range
    max_abs_residual = np.abs(residuals).max()
    if max_abs_residual > 0:
        # Map to signed range [-2^(residual_bits-1), 2^(residual_bits-1)-1]
        max_signed = (2 ** (residual_bits - 1)) - 1
        min_signed = -(2 ** (residual_bits - 1))
        # Scale linearly
        residuals_scaled = residuals / max_abs_residual * ((max_signed - min_signed) / 2)
    else:
        residuals_scaled = residuals.copy()
    
    # Round and clip to signed range
    residuals_int = np.round(residuals_scaled).astype(np.int16)
    max_signed = (2 ** (residual_bits - 1)) - 1
    min_signed = -(2 ** (residual_bits - 1))
    residuals_int = np.clip(residuals_int, min_signed, max_signed)
    
    # Encode each bit plane (using two's complement representation)
    bit_planes = []
    for i in range(residual_bits):
        # For two's complement, we extract bits directly
        if i == residual_bits - 1:  # Sign bit
            # Sign bit: 1 if negative, 0 if positive
            bit_plane = ((residuals_int < 0).astype(np.int16))
        else:
            # Value bits
            bit_plane = ((np.abs(residuals_int) >> i) & 1)
        bit_planes.append(bit_plane)
    
    return {
        'base_2bit': base_2bit_groups,
        'scales': scales,
        'zeros': zeros,
        'bit_planes': bit_planes,
        'group_size': group_size,
        'n_groups': n_groups,
        'original_shape': original_shape,
        'total_elements': total_elements,
        'max_abs_residual': max_abs_residual,
        'max_bits': max_bits,
        'residual_bits': residual_bits
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
    scales = data['scales']
    zeros = data['zeros']
    bit_planes = data['bit_planes']
    original_shape = data['original_shape']
    max_abs_residual = data['max_abs_residual']
    residual_bits = data['residual_bits']
    total_elements = data['total_elements']
    
    # Base reconstruction
    reconstructed = base_2bit.astype(np.float32) * scales + zeros
    
    if target_bits == 2:
        return reconstructed.ravel()[:total_elements].reshape(original_shape)
    
    # Add residual bit-planes
    bits_to_add = min(target_bits - 2, residual_bits)
    
    if bits_to_add <= 0:
        return reconstructed.ravel()[:total_elements].reshape(original_shape)
    
    # Compute residual value from bit planes
    residual_value = np.zeros_like(reconstructed, dtype=np.int16)
    
    max_signed = (2 ** (residual_bits - 1)) - 1
    min_signed = -(2 ** (residual_bits - 1))
    
    for i in range(bits_to_add):
        bit_val = bit_planes[i].astype(np.int16)
        if i == residual_bits - 1:  # Sign bit
            # Apply two's complement: subtract 2^i from sum of other bits
            # Actually, let's compute directly
            residual_value = -residual_value  # Sign bit means negative
        else:
            residual_value += bit_val * (2 ** i)
    
    # Scale back to actual residuals
    if max_abs_residual > 0:
        scale_factor = max_abs_residual / ((max_signed - min_signed) / 2)
        residual_value = residual_value.astype(np.float32) * scale_factor
    else:
        residual_value = residual_value.astype(np.float32)
    
    reconstructed += residual_value
    
    return reconstructed.ravel()[:total_elements].reshape(original_shape)
