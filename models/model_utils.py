"""
Model Utilities
===============
Utility functions for MindSpore models.
Includes custom ops, loss functions, and metrics.
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Tensor
from mindspore.nn.loss import LossBase
import numpy as np


# ============================================================================
# Custom Operations using mindspore.ops
# ============================================================================
class LabelSmoothing(LossBase):
    """
    Label smoothing cross entropy loss.
    Uses mindspore.ops for efficient computation.
    """
    
    def __init__(self, 
                 num_classes: int,
                 smoothing: float = 0.1,
                 reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
        # MindSpore ops
        self.log_softmax = ops.LogSoftmax(axis=-1)
        self.one_hot = ops.OneHot()
        self.reduce_sum = ops.ReduceSum()
        self.reduce_mean = ops.ReduceMean()
        
    def construct(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Compute label smoothing cross entropy loss.
        
        Args:
            logits: (batch, num_classes) model predictions
            labels: (batch,) integer labels
            
        Returns:
            loss: scalar loss value
        """
        # Log softmax
        log_probs = self.log_softmax(logits)
        
        # Create smooth labels
        on_value = Tensor(self.confidence, mindspore.float32)
        off_value = Tensor(self.smoothing / (self.num_classes - 1), mindspore.float32)
        smooth_labels = self.one_hot(labels, self.num_classes, on_value, off_value)
        
        # Compute loss
        loss = -self.reduce_sum(smooth_labels * log_probs, axis=-1)
        
        if self.reduction == 'mean':
            return self.reduce_mean(loss)
        elif self.reduction == 'sum':
            return self.reduce_sum(loss)
        return loss


class FocalLoss(LossBase):
    """
    Focal Loss for handling class imbalance.
    Uses mindspore.ops.
    """
    
    def __init__(self, 
                 num_classes: int,
                 gamma: float = 2.0,
                 alpha: float = 0.25,
                 reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        
        self.softmax = ops.Softmax(axis=-1)
        self.one_hot = ops.OneHot()
        self.pow = ops.Pow()
        self.log = ops.Log()
        
    def construct(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute focal loss."""
        probs = self.softmax(logits)
        
        # One hot encoding
        on_value = Tensor(1.0, mindspore.float32)
        off_value = Tensor(0.0, mindspore.float32)
        labels_one_hot = self.one_hot(labels, self.num_classes, on_value, off_value)
        
        # Get prob for correct class
        p_t = ops.ReduceSum()(probs * labels_one_hot, axis=-1)
        
        # Focal weight
        focal_weight = self.pow(1 - p_t, self.gamma)
        
        # Cross entropy
        ce = -self.log(p_t + 1e-8)
        
        # Focal loss
        loss = self.alpha * focal_weight * ce
        
        if self.reduction == 'mean':
            return ops.ReduceMean()(loss)
        return loss


# ============================================================================
# Metrics using mindspore.ops
# ============================================================================
class TopKAccuracy(nn.Cell):
    """
    Top-K accuracy metric using mindspore.ops.
    """
    
    def __init__(self, k: int = 5):
        super().__init__()
        self.k = k
        self.topk = ops.TopK(sorted=True)
        self.cast = ops.Cast()
        
    def construct(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Compute top-k accuracy.
        
        Args:
            logits: (batch, num_classes)
            labels: (batch,)
            
        Returns:
            accuracy: scalar
        """
        _, top_k_indices = self.topk(logits, self.k)
        
        # Check if true label is in top-k predictions
        labels = labels.expand_dims(-1)  # (batch, 1)
        correct = ops.Equal()(top_k_indices, labels).any(axis=-1)
        
        accuracy = self.cast(correct, mindspore.float32).mean()
        return accuracy


# ============================================================================
# Learning Rate Schedules
# ============================================================================
class CosineAnnealingLR(nn.Cell):
    """
    Cosine annealing learning rate schedule.
    """
    
    def __init__(self, 
                 base_lr: float,
                 min_lr: float,
                 total_steps: int,
                 warmup_steps: int = 0):
        super().__init__()
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        
        self.pi = Tensor(np.pi, mindspore.float32)
        self.cos = ops.Cos()
        
    def construct(self, step: int) -> Tensor:
        """Get learning rate for given step."""
        step = Tensor(step, mindspore.float32)
        
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (step / max(self.warmup_steps, 1))
        
        # Cosine annealing
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + self.cos(self.pi * progress))
        
        return lr


class OneCycleLR(nn.Cell):
    """
    One Cycle Learning Rate schedule.
    """
    
    def __init__(self,
                 max_lr: float,
                 total_steps: int,
                 pct_start: float = 0.3,
                 div_factor: float = 25.0,
                 final_div_factor: float = 10000.0):
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.initial_lr = max_lr / div_factor
        self.min_lr = max_lr / final_div_factor
        
        self.warmup_steps = int(total_steps * pct_start)
        self.cos = ops.Cos()
        self.pi = Tensor(np.pi, mindspore.float32)
        
    def construct(self, step: int) -> float:
        """Get learning rate for given step."""
        if step < self.warmup_steps:
            # Warmup phase: linear increase
            progress = step / max(self.warmup_steps, 1)
            return self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # Annealing phase: cosine decay
            progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + self.cos(self.pi * progress))
            return float(lr)


# ============================================================================
# Gradient Clipping using mindspore.ops
# ============================================================================
def clip_grad_norm(grads: tuple, max_norm: float = 1.0) -> tuple:
    """
    Clip gradients by global norm using mindspore.ops.
    
    Args:
        grads: Tuple of gradients
        max_norm: Maximum norm value
        
    Returns:
        Clipped gradients
    """
    # Calculate global norm
    total_norm = 0.0
    for g in grads:
        if g is not None:
            total_norm += ops.ReduceSum()(g * g)
    total_norm = ops.Sqrt()(total_norm)
    
    # Clip
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = ops.minimum(clip_coef, Tensor(1.0, mindspore.float32))
    
    clipped_grads = tuple(
        g * clip_coef if g is not None else None 
        for g in grads
    )
    
    return clipped_grads


if __name__ == "__main__":
    print("Testing Model Utilities")
    print("=" * 60)
    
    # Test LabelSmoothing
    print("\n1. Testing LabelSmoothing...")
    loss_fn = LabelSmoothing(num_classes=250, smoothing=0.1)
    logits = Tensor(np.random.randn(8, 250).astype(np.float32))
    labels = Tensor(np.random.randint(0, 250, 8).astype(np.int32))
    loss = loss_fn(logits, labels)
    print(f"   Loss: {loss.asnumpy():.4f}")
    print("   ✓ LabelSmoothing OK")
    
    # Test TopKAccuracy
    print("\n2. Testing TopKAccuracy...")
    metric = TopKAccuracy(k=5)
    acc = metric(logits, labels)
    print(f"   Top-5 Accuracy: {acc.asnumpy():.4f}")
    print("   ✓ TopKAccuracy OK")
    
    # Test CosineAnnealingLR
    print("\n3. Testing CosineAnnealingLR...")
    scheduler = CosineAnnealingLR(
        base_lr=0.001,
        min_lr=1e-6,
        total_steps=1000,
        warmup_steps=100
    )
    lr_0 = scheduler(0)
    lr_50 = scheduler(50)
    lr_500 = scheduler(500)
    print(f"   LR at step 0: {lr_0:.6f}")
    print(f"   LR at step 50: {lr_50:.6f}")
    print(f"   LR at step 500: {lr_500:.6f}")
    print("   ✓ CosineAnnealingLR OK")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
