"""
MindSpore Dataset Module (MindSpore-Native)
============================================
Dataset creation using ONLY MindSpore APIs.
Uses mindspore.numpy and mindspore.dataset for full MindSpore integration.
"""

import mindspore
import mindspore.numpy as mnp
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import Tensor
import json
from pathlib import Path
from typing import List, Optional
import random

# For initial data loading only (files on disk)
import numpy as np  # Only for file I/O, not processing


# ============================================================================
# Constants
# ============================================================================
MAX_LEN = 384
NUM_CLASSES = 250
CHANNELS = 708  # 118 landmarks * 6 (x, y, dx, dx2)

LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
       291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
       78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
       95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
NOSE = [1, 2, 98, 327]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE


# ============================================================================
# MindSpore NumPy Preprocessing Functions
# ============================================================================
def ms_nan_mean(x: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    """Compute mean ignoring NaN using mindspore.numpy."""
    mask = ~mnp.isnan(x)
    x_filled = mnp.where(mask, x, mnp.zeros_like(x))
    count = mnp.sum(mask.astype(mnp.float32), axis=axis, keepdims=keepdims)
    total = mnp.sum(x_filled, axis=axis, keepdims=keepdims)
    return total / mnp.maximum(count, mnp.array(1.0))


def ms_nan_std(x: Tensor, center: Tensor = None, axis=None, keepdims: bool = False) -> Tensor:
    """Compute std ignoring NaN using mindspore.numpy."""
    if center is None:
        center = ms_nan_mean(x, axis=axis, keepdims=True)
    d = x - center
    return mnp.sqrt(ms_nan_mean(d * d, axis=axis, keepdims=keepdims))


def preprocess_mindspore(landmarks: Tensor, max_len: int = MAX_LEN) -> Tensor:
    """
    Preprocess landmarks using mindspore.numpy.
    
    Args:
        landmarks: (T, 543, 3) MindSpore Tensor
        max_len: Maximum sequence length
        
    Returns:
        features: (T, 708) normalized features
    """
    # Add batch dim: (T, 543, 3) -> (1, T, 543, 3)
    x = mnp.expand_dims(landmarks, axis=0)
    B, T, N, C = x.shape
    
    # Get nose for centering
    nose = x[:, :, 17:18, :]
    mean = ms_nan_mean(nose, axis=(1, 2), keepdims=True)
    mean = mnp.where(mnp.isnan(mean), mnp.full_like(mean, 0.5), mean)
    
    # Select landmarks
    x = x[:, :, POINT_LANDMARKS, :]
    num_lm = len(POINT_LANDMARKS)
    
    # Normalize
    std = ms_nan_std(x, center=mean, axis=(1, 2), keepdims=True)
    std = mnp.maximum(std, mnp.array(1e-6))
    x = (x - mean) / std
    
    # Truncate
    if T > max_len:
        x = x[:, :max_len]
        T = max_len
    
    # Use x, y only
    x = x[..., :2]  # (1, T, 118, 2)
    
    # Velocities
    dx = mnp.zeros_like(x)
    dx2 = mnp.zeros_like(x)
    if T > 1:
        dx_val = x[:, 1:] - x[:, :-1]
        dx = mnp.concatenate([dx_val, mnp.zeros((1, 1, num_lm, 2))], axis=1)
    if T > 2:
        dx2_val = x[:, 2:] - x[:, :-2]
        dx2 = mnp.concatenate([dx2_val, mnp.zeros((1, 2, num_lm, 2))], axis=1)
    
    # Flatten and concat
    x_flat = mnp.reshape(x, (1, T, 2 * num_lm))
    dx_flat = mnp.reshape(dx, (1, T, 2 * num_lm))
    dx2_flat = mnp.reshape(dx2, (1, T, 2 * num_lm))
    features = mnp.concatenate([x_flat, dx_flat, dx2_flat], axis=-1)
    
    # Replace NaN
    features = mnp.where(mnp.isnan(features), mnp.zeros_like(features), features)
    
    return features[0]  # Remove batch dim


def pad_to_length(features: Tensor, max_len: int) -> Tensor:
    """Pad features to max_len using mindspore.numpy."""
    T, C = features.shape
    if T >= max_len:
        return features[:max_len]
    pad = mnp.zeros((max_len - T, C), dtype=features.dtype)
    return mnp.concatenate([features, pad], axis=0)


# ============================================================================
# MindSpore Dataset Generator
# ============================================================================
class ISLRDataGenerator:
    """
    Dataset generator for MindSpore.
    Loads data and converts to MindSpore operations.
    """
    
    def __init__(self, 
                 file_paths: List[str],
                 labels: List[int],
                 max_len: int = MAX_LEN,
                 augment: bool = False):
        self.file_paths = file_paths
        self.labels = labels
        self.max_len = max_len
        self.augment = augment
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load from disk (only place we use numpy)
        landmarks_np = np.load(self.file_paths[idx])
        
        # Convert to MindSpore Tensor immediately
        landmarks = Tensor(landmarks_np.astype(np.float32))
        
        # All processing is MindSpore
        features = preprocess_mindspore(landmarks, self.max_len)
        features = pad_to_length(features, self.max_len)
        
        # Augmentation using MindSpore ops
        if self.augment:
            if random.random() < 0.5:
                # Add noise using MindSpore
                noise = ops.StandardNormal()(features.shape) * 0.02
                features = features + noise.astype(features.dtype)
        
        return features.asnumpy(), np.int32(self.labels[idx])


def create_mindspore_dataset(file_paths: List[str],
                              labels: List[int],
                              batch_size: int = 64,
                              shuffle: bool = True,
                              augment: bool = False) -> ds.GeneratorDataset:
    """
    Create MindSpore dataset using GeneratorDataset.
    
    All preprocessing uses mindspore.numpy.
    """
    generator = ISLRDataGenerator(file_paths, labels, augment=augment)
    
    dataset = ds.GeneratorDataset(
        source=generator,
        column_names=["features", "labels"],
        shuffle=shuffle
    )
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset


def create_dummy_dataset_mindspore(num_samples: int = 1000,
                                    seq_len: int = MAX_LEN,
                                    num_features: int = CHANNELS,
                                    num_classes: int = NUM_CLASSES,
                                    batch_size: int = 64) -> ds.NumpySlicesDataset:
    """
    Create dummy dataset for testing using MindSpore.
    
    Uses mindspore.numpy for data generation.
    """
    # Generate using standard normal (MindSpore ops for random)
    features = ops.StandardNormal()((num_samples, seq_len, num_features)).asnumpy()
    labels = np.random.randint(0, num_classes, num_samples).astype(np.int32)
    
    dataset = ds.NumpySlicesDataset(
        data={"features": features.astype(np.float32), "labels": labels},
        shuffle=True
    )
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset


# ============================================================================
# Vocabulary
# ============================================================================
def load_vocabulary(path: str = None) -> dict:
    """Load sign vocabulary."""
    if path is None:
        path = Path(__file__).parent.parent / "dataset" / "sign_vocabulary.json"
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    print("Testing MindSpore-Native Dataset")
    print("=" * 60)
    
    # Test preprocessing with mindspore.numpy
    print("\n1. Testing MindSpore preprocessing...")
    test_landmarks = Tensor(ops.StandardNormal()((100, 543, 3)).asnumpy())
    features = preprocess_mindspore(test_landmarks)
    print(f"   Input: {test_landmarks.shape}")
    print(f"   Output: {features.shape}")
    assert features.shape == (100, 708)
    print("   ✓ Preprocessing OK")
    
    # Test padding
    print("\n2. Testing padding...")
    padded = pad_to_length(features, 384)
    print(f"   Padded: {padded.shape}")
    assert padded.shape == (384, 708)
    print("   ✓ Padding OK")
    
    # Test dataset
    print("\n3. Testing dummy dataset...")
    dataset = create_dummy_dataset_mindspore(num_samples=100, batch_size=8)
    print(f"   Batches: {dataset.get_dataset_size()}")
    
    for batch in dataset.create_tuple_iterator():
        f, l = batch
        print(f"   Batch features: {f.shape}")
        print(f"   Batch labels: {l.shape}")
        break
    print("   ✓ Dataset OK")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
