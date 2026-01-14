"""
MindSpore Preprocessing Layer
=============================
Preprocessing layer using mindspore.ops for GPU-accelerated data processing.
Implements landmark normalization and velocity computation for sign language recognition.
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Tensor, Parameter
from mindspore.common.initializer import Zero
from typing import List


# ============================================================================
# Landmark Definitions  
# ============================================================================
NOSE = [1, 2, 98, 327]
LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
       291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
       78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
       95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133,
        246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362,
        466, 388, 387, 386, 385, 384, 398]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))

POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE
MAX_LEN = 384


class PreprocessLayerMindSpore(nn.Cell):
    """
    MindSpore preprocessing layer using mindspore.ops.
    
    Implements landmark selection, normalization, and velocity computation.
    
    Operations using mindspore.ops:
    - ops.GatherD for landmark selection
    - ops.IsNan for NaN detection
    - ops.ReduceMean for averaging
    - ops.Concat for concatenation
    - ops.Pad for velocity calculation
    
    Args:
        max_len: Maximum sequence length
        point_landmarks: Indices of landmarks to select (default: 118)
    """
    
    def __init__(self, max_len: int = MAX_LEN, 
                 point_landmarks: List[int] = None):
        super().__init__()
        
        self.max_len = max_len
        self.point_landmarks = point_landmarks or POINT_LANDMARKS
        self.num_landmarks = len(self.point_landmarks)
        
        # Store landmark indices as parameter for graph mode
        self.landmark_indices = Tensor(self.point_landmarks, mindspore.int32)
        self.nose_idx = Tensor([17], mindspore.int32)
        
        # MindSpore ops for preprocessing
        self.isnan = ops.IsNan()
        self.concat = ops.Concat(axis=-1)
        self.reduce_mean = ops.ReduceMean(keep_dims=True)
        self.reduce_sum = ops.ReduceSum(keep_dims=True)
        self.zeros_like = ops.ZerosLike()
        self.ones_like = ops.OnesLike()
        self.sqrt = ops.Sqrt()
        self.gather = ops.Gather()
        self.reshape = ops.Reshape()
        self.select = ops.Select()
        
    def nan_mean(self, x: Tensor, axis: tuple, keepdims: bool = True) -> Tensor:
        """Compute mean ignoring NaN values using mindspore.ops."""
        mask = ~self.isnan(x)
        x_filled = self.select(mask, x, self.zeros_like(x))
        count = self.reduce_sum(mask.astype(mindspore.float32), axis)
        total = self.reduce_sum(x_filled, axis)
        return total / ops.maximum(count, Tensor(1.0, mindspore.float32))
    
    def nan_std(self, x: Tensor, center: Tensor, axis: tuple) -> Tensor:
        """Compute std ignoring NaN values using mindspore.ops."""
        d = x - center
        return self.sqrt(self.nan_mean(d * d, axis, keepdims=True))
    
    def construct(self, x: Tensor) -> Tensor:
        """
        Preprocess raw landmarks to normalized features with velocities.
        
        Args:
            x: (batch, T, 543, 3) raw landmarks or (batch, T, 543, 2) for x,y only
            
        Returns:
            features: (batch, T, 708) normalized features with velocities
        """
        # Handle 3D input (add batch dimension)
        if x.ndim == 3:
            x = x.expand_dims(0)
        
        B, T, N, C = x.shape
        
        # Get nose landmark for centering (index 17)
        # Using ops.Gather for landmark selection
        nose = self.gather(x, self.nose_idx, 2)  # (B, T, 1, C)
        
        # Compute mean of nose positions
        mean = self.nan_mean(nose, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)
        # Replace NaN mean with 0.5
        mean = self.select(self.isnan(mean), 
                          Tensor(0.5, mean.dtype) * self.ones_like(mean), 
                          mean)
        
        # Select point landmarks using ops.Gather
        x = self.gather(x, self.landmark_indices, 2)  # (B, T, 118, C)
        
        # Compute std for normalization
        std = self.nan_std(x, center=mean, axis=(1, 2))  # (B, 1, 1, C)
        # Prevent division by zero
        std = ops.maximum(std, Tensor(1e-6, std.dtype))
        
        # Normalize
        x = (x - mean) / std
        
        # Truncate to max_len
        if self.max_len is not None and T > self.max_len:
            x = x[:, :self.max_len]
            T = self.max_len
        
        # Use only x, y coordinates
        x = x[..., :2]  # (B, T, 118, 2)
        
        # Calculate velocities using ops.Pad
        # dx = x[t] - x[t-1]
        dx = ops.Pad(((0, 0), (0, 1), (0, 0), (0, 0)))(
            x[:, 1:] - x[:, :-1]
        ) if T > 1 else self.zeros_like(x)
        
        # dx2 = x[t] - x[t-2]
        dx2 = ops.Pad(((0, 0), (0, 2), (0, 0), (0, 0)))(
            x[:, 2:] - x[:, :-2]
        ) if T > 2 else self.zeros_like(x)
        
        # Reshape and concatenate
        # (B, T, 118, 2) -> (B, T, 236)
        x_flat = self.reshape(x, (B, T, 2 * self.num_landmarks))
        dx_flat = self.reshape(dx, (B, T, 2 * self.num_landmarks))
        dx2_flat = self.reshape(dx2, (B, T, 2 * self.num_landmarks))
        
        # Concatenate all features
        features = self.concat((x_flat, dx_flat, dx2_flat))  # (B, T, 708)
        
        # Replace NaN with 0
        features = self.select(self.isnan(features),
                              self.zeros_like(features),
                              features)
        
        return features


class TemporalInterpolation(nn.Cell):
    """
    Temporal interpolation layer using mindspore.ops.
    Resizes sequence to target length.
    """
    
    def __init__(self, method: str = 'bilinear'):
        super().__init__()
        self.method = method
        # Use ops.ResizeBilinear or similar
        self.resize = ops.ResizeBilinear((1, 1))  # Placeholder
    
    def construct(self, x: Tensor, target_len: int) -> Tensor:
        """Resize sequence to target length."""
        # x: (B, T, C)
        B, T, C = x.shape
        
        # Reshape for image-like resize: (B, C, T, 1) -> (B, C, target_len, 1)
        x = x.transpose(0, 2, 1).expand_dims(-1)
        
        # Resize
        x = ops.ResizeBilinear((target_len, 1))(x)
        
        # Reshape back: (B, C, target_len, 1) -> (B, target_len, C)
        x = x.squeeze(-1).transpose(0, 2, 1)
        
        return x


class DataAugmentationMindSpore(nn.Cell):
    """
    Data augmentation layer using mindspore.ops.
    """
    
    def __init__(self, 
                 noise_scale: float = 0.02,
                 time_mask_ratio: float = 0.1):
        super().__init__()
        self.noise_scale = noise_scale
        self.time_mask_ratio = time_mask_ratio
        
    def construct(self, x: Tensor, training: bool = True) -> Tensor:
        """Apply augmentation during training."""
        if not training:
            return x
        
        # Spatial noise using ops
        noise = ops.StandardNormal()(x.shape) * self.noise_scale
        x = x + noise
        
        return x


if __name__ == "__main__":
    print("Testing Preprocessing Layer")
    print("=" * 60)
    
    # Test PreprocessLayer
    print("\n1. Testing PreprocessLayerMindSpore...")
    layer = PreprocessLayerMindSpore(max_len=384)
    
    # Create test input using MindSpore ops
    batch_size, T, N, C = 2, 100, 543, 3
    x = Tensor(ops.StandardNormal()((batch_size, T, N, C)).asnumpy().astype('float32'))
    
    print(f"   Input shape: {x.shape}")
    
    # Forward pass
    output = layer(x)
    print(f"   Output shape: {output.shape}")
    
    expected_shape = (batch_size, T, 708)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print("   ✓ PreprocessLayer OK")
    
    # Verify no NaN in output
    has_nan = ops.IsNan()(output).any()
    print(f"   Has NaN in output: {has_nan}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
