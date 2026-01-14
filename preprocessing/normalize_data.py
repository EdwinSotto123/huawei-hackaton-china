"""
Preprocessing Module
====================
Data normalization and preprocessing using MindSpore APIs.
Normalizes landmarks and computes velocity features.
"""

import mindspore
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import Tensor
from typing import List


# ============================================================================
# Landmark Definitions
# ============================================================================
LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
       291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
       78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
       95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

NOSE = [1, 2, 98, 327]

REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133,
        246, 161, 160, 159, 158, 157, 173]

LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362,
        466, 388, 387, 386, 385, 384, 398]

LHAND = list(range(468, 489))  # 21 landmarks
RHAND = list(range(522, 543))  # 21 landmarks

# 118 selected landmarks
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE
NUM_LANDMARKS = len(POINT_LANDMARKS)  # 118

# Nose tip index for centering
NOSE_TIP_INDEX = 17


# ============================================================================
# MindSpore NumPy Functions
# ============================================================================
def ms_nan_mean(x: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    """
    Compute mean ignoring NaN values using mindspore.numpy.
    
    Equivalent to numpy.nanmean but runs on GPU/Ascend.
    
    Args:
        x: Input tensor
        axis: Axes to reduce over
        keepdims: Keep reduced dimensions
        
    Returns:
        Mean tensor with NaN values ignored
    """
    # Create mask for valid (non-NaN) values
    mask = ~mnp.isnan(x)
    
    # Replace NaN with 0 for sum
    x_filled = mnp.where(mask, x, mnp.zeros_like(x))
    
    # Count valid values
    count = mnp.sum(mask.astype(mnp.float32), axis=axis, keepdims=keepdims)
    
    # Sum valid values
    total = mnp.sum(x_filled, axis=axis, keepdims=keepdims)
    
    # Avoid division by zero
    count = mnp.maximum(count, mnp.array(1.0))
    
    return total / count


def ms_nan_std(x: Tensor, center: Tensor = None, 
               axis=None, keepdims: bool = False) -> Tensor:
    """
    Compute std ignoring NaN values using mindspore.numpy.
    
    Args:
        x: Input tensor
        center: Center for std calculation (default: nanmean)
        axis: Axes to reduce over
        keepdims: Keep reduced dimensions
        
    Returns:
        Std tensor with NaN values ignored
    """
    if center is None:
        center = ms_nan_mean(x, axis=axis, keepdims=True)
    
    d = x - center
    variance = ms_nan_mean(d * d, axis=axis, keepdims=keepdims)
    
    return mnp.sqrt(variance)


# ============================================================================
# Normalization Functions using MindSpore NumPy
# ============================================================================
def normalize_landmarks_mindspore(landmarks: Tensor,
                                   max_len: int = 384,
                                   point_landmarks: List[int] = None) -> Tensor:
    """
    Normalize landmark sequence using mindspore.numpy.
    
    This is the MindSpore-native version of the preprocessing.
    All operations use mindspore.numpy instead of numpy.
    
    Args:
        landmarks: (seq_len, 543, 3) raw landmarks as MindSpore Tensor
        max_len: Maximum sequence length
        point_landmarks: Indices of landmarks to select
        
    Returns:
        normalized: (seq_len, 708) normalized features
    """
    if point_landmarks is None:
        point_landmarks = POINT_LANDMARKS
    
    num_landmarks = len(point_landmarks)
    
    # Add batch dimension if needed: (T, 543, 3) -> (1, T, 543, 3)
    if landmarks.ndim == 3:
        x = mnp.expand_dims(landmarks, axis=0)
    else:
        x = landmarks
    
    B, T, N, C = x.shape
    
    # Get nose tip for centering (index 17 in face mesh)
    # Using mnp.take along axis
    nose_idx = mnp.array([NOSE_TIP_INDEX])
    nose = x[:, :, NOSE_TIP_INDEX:NOSE_TIP_INDEX+1, :]  # (B, T, 1, 3)
    
    # Calculate mean of nose positions
    mean = ms_nan_mean(nose, axis=(1, 2), keepdims=True)  # (B, 1, 1, 3)
    
    # Replace NaN mean with 0.5 (center of normalized coordinates)
    mean = mnp.where(mnp.isnan(mean), mnp.full_like(mean, 0.5), mean)
    
    # Select point landmarks: (B, T, 543, 3) -> (B, T, 118, 3)
    landmark_indices = mnp.array(point_landmarks)
    x = x[:, :, point_landmarks, :]
    
    # Calculate std for normalization
    std = ms_nan_std(x, center=mean, axis=(1, 2), keepdims=True)
    
    # Prevent division by zero
    std = mnp.maximum(std, mnp.array(1e-6))
    
    # Normalize
    x = (x - mean) / std
    
    # Truncate to max_len
    if max_len is not None and T > max_len:
        x = x[:, :max_len]
        T = max_len
    
    # Use only x, y coordinates (z is often noisy)
    x = x[..., :2]  # (B, T, 118, 2)
    
    # Calculate velocities using mindspore.numpy
    # dx = x[t] - x[t-1]
    dx = mnp.zeros_like(x)
    if T > 1:
        dx_values = x[:, 1:] - x[:, :-1]
        # Pad with zeros at the end
        dx = mnp.concatenate([dx_values, mnp.zeros((B, 1, num_landmarks, 2))], axis=1)
    
    # dx2 = x[t] - x[t-2]
    dx2 = mnp.zeros_like(x)
    if T > 2:
        dx2_values = x[:, 2:] - x[:, :-2]
        # Pad with zeros at the end
        dx2 = mnp.concatenate([dx2_values, mnp.zeros((B, 2, num_landmarks, 2))], axis=1)
    
    # Reshape: (B, T, 118, 2) -> (B, T, 236)
    x_flat = mnp.reshape(x, (B, T, 2 * num_landmarks))
    dx_flat = mnp.reshape(dx, (B, T, 2 * num_landmarks))
    dx2_flat = mnp.reshape(dx2, (B, T, 2 * num_landmarks))
    
    # Concatenate all features: (B, T, 708)
    features = mnp.concatenate([x_flat, dx_flat, dx2_flat], axis=-1)
    
    # Replace NaN with 0
    features = mnp.where(mnp.isnan(features), mnp.zeros_like(features), features)
    
    # Remove batch dimension if added
    if landmarks.ndim == 3:
        features = features[0]
    
    return features


def pad_sequence_mindspore(features: Tensor, max_len: int) -> Tensor:
    """
    Pad sequence to max_len using mindspore.numpy.
    
    Args:
        features: (T, C) features
        max_len: Target length
        
    Returns:
        padded: (max_len, C) padded features
    """
    T, C = features.shape
    
    if T >= max_len:
        return features[:max_len]
    
    # Create padding
    pad = mnp.zeros((max_len - T, C), dtype=features.dtype)
    
    # Concatenate
    return mnp.concatenate([features, pad], axis=0)


def batch_normalize_mindspore(batch_landmarks: Tensor, max_len: int = 384) -> Tensor:
    """
    Normalize a batch of landmark sequences.
    
    Args:
        batch_landmarks: (batch_size, T, 543, 3)
        max_len: Maximum sequence length
        
    Returns:
        normalized: (batch_size, max_len, 708)
    """
    B = batch_landmarks.shape[0]
    
    # Process each sample
    results = []
    for i in range(B):
        features = normalize_landmarks_mindspore(batch_landmarks[i], max_len)
        features = pad_sequence_mindspore(features, max_len)
        results.append(features)
    
    return mnp.stack(results, axis=0)


# ============================================================================
# Data Augmentation using MindSpore NumPy
# ============================================================================
def augment_spatial_mindspore(features: Tensor, noise_scale: float = 0.02) -> Tensor:
    """
    Add random spatial noise using mindspore.numpy.
    
    Args:
        features: (T, C) features
        noise_scale: Standard deviation of noise
        
    Returns:
        Augmented features
    """
    noise = ops.StandardNormal()(features.shape).astype(features.dtype) * noise_scale
    return features + noise


def augment_flip_mindspore(features: Tensor, num_landmarks: int = 118) -> Tensor:
    """
    Horizontal flip augmentation (swap left/right landmarks).
    
    Args:
        features: (T, 708) features
        num_landmarks: Number of landmarks (118)
        
    Returns:
        Flipped features
    """
    T, C = features.shape
    
    # Reshape to (T, 118, 6) for easier manipulation
    # 6 = 2 (x,y) + 2 (dx) + 2 (dx2)
    x = mnp.reshape(features, (T, num_landmarks, 6))
    
    # Flip x coordinates (negate x values at indices 0, 2, 4)
    flip_mask = mnp.array([[-1, 1, -1, 1, -1, 1]])
    x = x * flip_mask
    
    return mnp.reshape(x, (T, C))


# ============================================================================
# Test
# ============================================================================
if __name__ == "__main__":
    print("Testing MindSpore-Native Preprocessing")
    print("=" * 60)
    
    # Create test data using mindspore.numpy
    print("\n1. Creating test data with mnp.random...")
    # Note: Using ops for random since mnp doesn't have randn
    landmarks = Tensor(ops.StandardNormal()((100, 543, 3)).asnumpy())
    print(f"   Input shape: {landmarks.shape}")
    
    # Test normalization
    print("\n2. Testing normalize_landmarks_mindspore...")
    features = normalize_landmarks_mindspore(landmarks)
    print(f"   Output shape: {features.shape}")
    
    expected_shape = (100, 708)
    assert features.shape == expected_shape, f"Expected {expected_shape}, got {features.shape}"
    print("   ✓ Normalization OK")
    
    # Test padding
    print("\n3. Testing pad_sequence_mindspore...")
    padded = pad_sequence_mindspore(features, max_len=384)
    print(f"   Padded shape: {padded.shape}")
    assert padded.shape == (384, 708)
    print("   ✓ Padding OK")
    
    # Test NaN handling
    print("\n4. Testing NaN handling...")
    has_nan = mnp.any(mnp.isnan(features))
    print(f"   Has NaN: {has_nan}")
    assert not has_nan
    print("   ✓ No NaN in output")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
