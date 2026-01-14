"""
ISLR Model - Squeezeformer Architecture
========================================
Sign language recognition with:
- Squeezeformer blocks (Macaron FFN structure)
- Rotary Position Embeddings (RoPE)
- Cross-Region Attention (Hands ↔ Face)
- SwiGLU activation
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Tensor
from typing import Optional

from .squeezeformer_block import SqueezeformerBlock, SqueezeformerEncoder, TemporalDownsample
from .cross_attention import RegionalProcessor, CrossRegionAttention
from .swiglu import SwiGLUFFN


class ISLRModelV2(nn.Cell):
    """
    Improved Sign Language Recognition Model.
    
    Architecture:
        Input (B, T, 708)
        → RegionalProcessor (hands↔face cross-attention)
        → Squeezeformer Encoder (6 blocks with RoPE + SwiGLU)
        → Temporal pooling
        → Classifier
    
    Key improvements over V1:
    - Macaron FFN structure (better gradients)
    - RoPE (better position encoding)
    - Cross-region attention (captures hand-face correlations)
    - SwiGLU activation (better than GELU)
    
    Args:
        input_dim: Input feature dimension (default: 708)
        num_classes: Number of sign classes (default: 250)
        dim: Model dimension (default: 256)
        num_blocks: Number of Squeezeformer blocks (default: 6)
        num_heads: Number of attention heads (default: 8)
        kernel_size: Convolution kernel size (default: 31)
        ffn_dim: FFN hidden dimension (default: 1024)
        dropout: Dropout rate (default: 0.1)
        use_regional: Use regional cross-attention (default: True)
    """
    
    def __init__(self,
                 input_dim: int = 708,
                 num_classes: int = 250,
                 dim: int = 256,
                 num_blocks: int = 6,
                 num_heads: int = 8,
                 kernel_size: int = 31,
                 ffn_dim: int = 1024,
                 dropout: float = 0.1,
                 use_regional: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dim = dim
        self.use_regional = use_regional
        
        # ============================================================
        # Input Processing
        # ============================================================
        if use_regional:
            # Regional processor with cross-attention
            self.input_processor = RegionalProcessor(
                input_dim=input_dim,
                hidden_dim=dim,
                num_heads=num_heads // 2
            )
        else:
            # Simple linear embedding
            self.input_processor = nn.SequentialCell([
                nn.Dense(input_dim, dim, has_bias=False),
                nn.LayerNorm([dim]),
                nn.Dropout(p=dropout)
            ])
        
        # ============================================================
        # Squeezeformer Encoder
        # ============================================================
        self.encoder = SqueezeformerEncoder(
            dim=dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            kernel_size=kernel_size,
            ffn_dim=ffn_dim,
            dropout=dropout,
            downsample_at=(2, 4)  # Downsample after blocks 2 and 4
        )
        
        # ============================================================
        # Classification Head
        # ============================================================
        self.pool = ops.ReduceMean(keep_dims=False)
        self.head_norm = nn.LayerNorm([dim])
        self.head_dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Dense(dim, num_classes)
    
    def construct(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, 708) input features
            
        Returns:
            logits: (batch, num_classes)
        """
        # Handle NaN inputs
        x = mnp.where(mnp.isnan(x), mnp.zeros_like(x), x)
        
        # Input processing (with regional cross-attention if enabled)
        x = self.input_processor(x)  # (B, T, dim)
        
        # Squeezeformer encoding
        x = self.encoder(x)  # (B, T', dim) with T' < T due to downsampling
        
        # Classification head
        x = self.head_norm(x)
        x = self.pool(x, 1)  # Global average pooling: (B, dim)
        x = self.head_dropout(x)
        logits = self.classifier(x)  # (B, num_classes)
        
        return logits
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.size for p in self.trainable_params())


class ISLRModelV2Lite(nn.Cell):
    """
    Lightweight version of ISLR V2 for faster inference.
    
    Reduces parameters while maintaining most accuracy gains.
    """
    
    def __init__(self,
                 input_dim: int = 708,
                 num_classes: int = 250,
                 dim: int = 192,          # Smaller
                 num_blocks: int = 4,     # Fewer blocks
                 num_heads: int = 4,      # Fewer heads
                 kernel_size: int = 17,   # Smaller kernel
                 ffn_dim: int = 512,      # Smaller FFN
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dim = dim
        
        # Simple embedding (no regional for speed)
        self.embed = nn.Dense(input_dim, dim, has_bias=False)
        self.embed_norm = nn.LayerNorm([dim])
        
        # Squeezeformer encoder (simplified)
        self.encoder = SqueezeformerEncoder(
            dim=dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            kernel_size=kernel_size,
            ffn_dim=ffn_dim,
            dropout=dropout,
            downsample_at=(1, 2)  # Early downsampling for speed
        )
        
        # Head
        self.pool = ops.ReduceMean(keep_dims=False)
        self.classifier = nn.Dense(dim, num_classes)
    
    def construct(self, x: Tensor) -> Tensor:
        x = mnp.where(mnp.isnan(x), mnp.zeros_like(x), x)
        x = self.embed(x)
        x = self.embed_norm(x)
        x = self.encoder(x)
        x = self.pool(x, 1)
        return self.classifier(x)
    
    def get_num_params(self) -> int:
        return sum(p.size for p in self.trainable_params())


def create_islr_model_v2(num_classes: int = 250, 
                         variant: str = 'base',
                         **kwargs) -> nn.Cell:
    """
    Factory function for ISLR V2 models.
    
    Args:
        num_classes: Number of sign classes
        variant: 'base', 'large', or 'lite'
        **kwargs: Additional parameters
        
    Returns:
        Model instance
    """
    if variant == 'lite':
        return ISLRModelV2Lite(num_classes=num_classes, **kwargs)
    
    elif variant == 'large':
        return ISLRModelV2(
            num_classes=num_classes,
            dim=384,
            num_blocks=8,
            num_heads=12,
            ffn_dim=1536,
            **kwargs
        )
    
    else:  # 'base'
        return ISLRModelV2(
            num_classes=num_classes,
            **kwargs
        )


if __name__ == "__main__":
    print("Testing ISLR Model V2 (Squeezeformer)")
    print("=" * 60)
    
    # Test V2 Base
    print("\n1. Testing ISLRModelV2 (Base)...")
    model = ISLRModelV2(
        input_dim=708,
        num_classes=250,
        dim=256,
        num_blocks=4,  # Reduced for testing
        use_regional=True
    )
    
    num_params = model.get_num_params()
    print(f"   Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    x = Tensor(ops.StandardNormal()((2, 100, 708)).asnumpy().astype('float32'))
    print(f"   Input: {x.shape}")
    
    logits = model(x)
    print(f"   Output: {logits.shape}")
    assert logits.shape == (2, 250)
    print("   ✓ ISLRModelV2 Base OK")
    
    # Test V2 Lite
    print("\n2. Testing ISLRModelV2Lite...")
    model_lite = ISLRModelV2Lite(num_classes=250)
    
    num_params_lite = model_lite.get_num_params()
    print(f"   Parameters: {num_params_lite:,} ({num_params_lite/1e6:.2f}M)")
    
    logits_lite = model_lite(x)
    print(f"   Output: {logits_lite.shape}")
    assert logits_lite.shape == (2, 250)
    print("   ✓ ISLRModelV2 Lite OK")
    
    # Test factory function
    print("\n3. Testing factory function...")
    model_factory = create_islr_model_v2(num_classes=250, variant='base')
    print(f"   Created: {type(model_factory).__name__}")
    print("   ✓ Factory OK")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print(f"\nModel comparison:")
    print(f"  V2 Base: {num_params:,} params")
    print(f"  V2 Lite: {num_params_lite:,} params")
