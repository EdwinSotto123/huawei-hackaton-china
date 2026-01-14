"""
Squeezeformer Block
===================
Squeezeformer architecture with Macaron-style FFN sandwiching.
Better than vanilla Transformer for temporal sequences.

Structure:
x -> 1/2 FFN -> MHSA -> Conv -> 1/2 FFN -> output
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Tensor
from typing import Optional

from .rotary_embedding import RotaryMultiHeadAttention
from .swiglu import SwiGLUFFN


class ConvolutionModule(nn.Cell):
    """
    Convolution module for Squeezeformer.
    
    Depthwise separable convolution with GLU gating.
    
    Args:
        dim: Model dimension
        kernel_size: Convolution kernel size
        dropout: Dropout rate
    """
    
    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        
        # Pointwise expansion
        self.pw_conv1 = nn.Conv1d(dim, dim * 2, kernel_size=1, has_bias=False)
        self.glu = nn.GLU(axis=1)  # Gated Linear Unit
        
        # Depthwise conv
        self.dw_conv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            pad_mode='same',
            group=dim,
            has_bias=False
        )
        self.bn = nn.BatchNorm1d(dim)
        self.silu = nn.SiLU()
        
        # Pointwise projection
        self.pw_conv2 = nn.Conv1d(dim, dim, kernel_size=1, has_bias=False)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def construct(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            output: (batch, seq_len, dim)
        """
        # Transpose for conv: (B, T, C) -> (B, C, T)
        x = x.transpose(0, 2, 1)
        
        # Pointwise + GLU
        x = self.pw_conv1(x)
        x = self.glu(x)
        
        # Depthwise conv
        x = self.dw_conv(x)
        x = self.bn(x)
        x = self.silu(x)
        
        # Pointwise projection
        x = self.pw_conv2(x)
        x = self.dropout(x)
        
        # Transpose back: (B, C, T) -> (B, T, C)
        x = x.transpose(0, 2, 1)
        
        return x


class SqueezeformerBlock(nn.Cell):
    """
    Squeezeformer Block with Macaron-style structure.
    
    Structure:
    x -> LayerNorm -> 1/2 FFN -> +x
      -> LayerNorm -> MHSA -> +x  
      -> LayerNorm -> Conv -> +x
      -> LayerNorm -> 1/2 FFN -> +x
    
    The "Macaron" structure sandwiches attention and conv
    between two half-weight FFN layers.
    
    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        kernel_size: Convolution kernel size
        ffn_dim: FFN hidden dimension
        dropout: Dropout rate
        max_seq_len: Maximum sequence length for RoPE
    """
    
    def __init__(self,
                 dim: int = 256,
                 num_heads: int = 8,
                 kernel_size: int = 31,
                 ffn_dim: int = 1024,
                 dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        
        self.dim = dim
        
        # First half FFN (scaled by 0.5)
        self.ffn_norm1 = nn.LayerNorm([dim])
        self.ffn1 = SwiGLUFFN(dim, ffn_dim, dropout)
        
        # Multi-head attention with RoPE
        self.attn_norm = nn.LayerNorm([dim])
        self.attn = RotaryMultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        # Convolution module
        self.conv_norm = nn.LayerNorm([dim])
        self.conv = ConvolutionModule(dim, kernel_size, dropout)
        
        # Second half FFN (scaled by 0.5)
        self.ffn_norm2 = nn.LayerNorm([dim])
        self.ffn2 = SwiGLUFFN(dim, ffn_dim, dropout)
        
        # Final layer norm
        self.final_norm = nn.LayerNorm([dim])
        
        # Scale factors for Macaron FFN
        self.ffn_scale = Tensor(0.5, mindspore.float32)
    
    def construct(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            output: (batch, seq_len, dim)
        """
        # First half FFN
        residual = x
        x = self.ffn_norm1(x)
        x = residual + self.ffn_scale * self.ffn1(x)
        
        # Multi-head attention
        residual = x
        x = self.attn_norm(x)
        x = residual + self.attn(x)
        
        # Convolution module
        residual = x
        x = self.conv_norm(x)
        x = residual + self.conv(x)
        
        # Second half FFN
        residual = x
        x = self.ffn_norm2(x)
        x = residual + self.ffn_scale * self.ffn2(x)
        
        # Final norm
        x = self.final_norm(x)
        
        return x


class TemporalDownsample(nn.Cell):
    """
    Temporal downsampling layer using strided convolution.
    
    Args:
        dim: Model dimension
        stride: Downsampling factor
    """
    
    def __init__(self, dim: int, stride: int = 2):
        super().__init__()
        
        self.conv = nn.Conv1d(
            dim, dim,
            kernel_size=stride * 2 - 1,
            stride=stride,
            pad_mode='same',
            has_bias=False
        )
        self.norm = nn.LayerNorm([dim])
    
    def construct(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            output: (batch, seq_len // stride, dim)
        """
        # Transpose for conv
        x = x.transpose(0, 2, 1)  # (B, C, T)
        x = self.conv(x)
        x = x.transpose(0, 2, 1)  # (B, T', C)
        x = self.norm(x)
        return x


class SqueezeformerEncoder(nn.Cell):
    """
    Full Squeezeformer encoder with multiple blocks.
    
    Args:
        dim: Model dimension
        num_blocks: Number of Squeezeformer blocks
        num_heads: Number of attention heads
        kernel_size: Convolution kernel size
        ffn_dim: FFN hidden dimension
        dropout: Dropout rate
        downsample_at: Block indices to apply downsampling
    """
    
    def __init__(self,
                 dim: int = 256,
                 num_blocks: int = 6,
                 num_heads: int = 8,
                 kernel_size: int = 31,
                 ffn_dim: int = 1024,
                 dropout: float = 0.1,
                 downsample_at: tuple = (2, 4)):
        super().__init__()
        
        self.blocks = nn.CellList()
        self.downsamples = nn.CellDict()
        
        for i in range(num_blocks):
            # Add Squeezeformer block
            self.blocks.append(SqueezeformerBlock(
                dim=dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                ffn_dim=ffn_dim,
                dropout=dropout
            ))
            
            # Add downsampling if needed
            if i in downsample_at:
                self.downsamples[str(i)] = TemporalDownsample(dim, stride=2)
        
        self.downsample_at = downsample_at
    
    def construct(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            output: (batch, reduced_seq_len, dim)
        """
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            # Apply downsampling if at specified position
            if i in self.downsample_at:
                x = self.downsamples[str(i)](x)
        
        return x


if __name__ == "__main__":
    print("Testing Squeezeformer Blocks")
    print("=" * 60)
    
    # Test ConvolutionModule
    print("\n1. Testing ConvolutionModule...")
    conv_mod = ConvolutionModule(dim=256, kernel_size=31)
    x = Tensor(ops.StandardNormal()((2, 100, 256)).asnumpy().astype('float32'))
    out = conv_mod(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == x.shape
    print("   ✓ ConvolutionModule OK")
    
    # Test SqueezeformerBlock
    print("\n2. Testing SqueezeformerBlock...")
    block = SqueezeformerBlock(dim=256, num_heads=8)
    out = block(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == x.shape
    print("   ✓ SqueezeformerBlock OK")
    
    # Test TemporalDownsample
    print("\n3. Testing TemporalDownsample...")
    downsample = TemporalDownsample(dim=256, stride=2)
    out = downsample(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == (2, 50, 256)
    print("   ✓ TemporalDownsample OK")
    
    # Test SqueezeformerEncoder
    print("\n4. Testing SqueezeformerEncoder...")
    encoder = SqueezeformerEncoder(
        dim=256, num_blocks=4, num_heads=8,
        downsample_at=(1, 2)
    )
    out = encoder(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    print("   ✓ SqueezeformerEncoder OK")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
