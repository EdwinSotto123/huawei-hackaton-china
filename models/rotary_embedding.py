"""
Rotary Position Embeddings (RoPE)
=================================
Rotary position encoding for transformers.
Encodes relative position through rotation.
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Tensor
import math


class RotaryEmbedding(nn.Cell):
    """
    Rotary Position Embedding (RoPE).
    
    Encodes position through rotation in the complex plane.
    Better for variable-length sequences than absolute position embeddings.
    
    Args:
        dim: Dimension of the embedding (must be even)
        max_seq_len: Maximum sequence length
        base: Base for frequency computation (default: 10000)
    """
    
    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        
        assert dim % 2 == 0, "dim must be even for RoPE"
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (mnp.arange(0, dim, 2).astype(mnp.float32) / dim))
        self.inv_freq = Tensor(inv_freq)
        
        # Precompute sin/cos for efficiency
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Precompute sin and cos caches."""
        positions = mnp.arange(seq_len).astype(mnp.float32)
        
        # Outer product: (seq_len,) × (dim/2,) -> (seq_len, dim/2)
        freqs = ops.Einsum('i,j->ij')((positions, self.inv_freq))
        
        # Duplicate for pairs: (seq_len, dim)
        freqs = mnp.concatenate([freqs, freqs], axis=-1)
        
        # Cache sin and cos
        self.cos_cache = Tensor(mnp.cos(freqs))
        self.sin_cache = Tensor(mnp.sin(freqs))
    
    def _rotate_half(self, x: Tensor) -> Tensor:
        """Rotate half the hidden dims."""
        # x: (..., dim) -> split into two halves and rotate
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return mnp.concatenate([-x2, x1], axis=-1)
    
    def construct(self, q: Tensor, k: Tensor, seq_len: int = None) -> tuple:
        """
        Apply rotary embeddings to query and key.
        
        Args:
            q: Query tensor (batch, heads, seq_len, head_dim)
            k: Key tensor (batch, heads, seq_len, head_dim)
            seq_len: Actual sequence length (for caching)
            
        Returns:
            q_rotated, k_rotated: Rotated query and key
        """
        if seq_len is None:
            seq_len = q.shape[2]
        
        # Get cached sin/cos for this sequence length
        cos = self.cos_cache[:seq_len]  # (seq_len, dim)
        sin = self.sin_cache[:seq_len]
        
        # Reshape for broadcasting: (1, 1, seq_len, dim)
        cos = cos.reshape(1, 1, seq_len, -1)
        sin = sin.reshape(1, 1, seq_len, -1)
        
        # Apply rotation: q' = q * cos + rotate_half(q) * sin
        q_rotated = q * cos + self._rotate_half(q) * sin
        k_rotated = k * cos + self._rotate_half(k) * sin
        
        return q_rotated, k_rotated


class RotaryMultiHeadAttention(nn.Cell):
    """
    Multi-Head Attention with Rotary Position Embeddings.
    
    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        max_seq_len: Maximum sequence length for RoPE
    """
    
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Dense(dim, dim, has_bias=False)
        self.k_proj = nn.Dense(dim, dim, has_bias=False)
        self.v_proj = nn.Dense(dim, dim, has_bias=False)
        self.out_proj = nn.Dense(dim, dim)
        
        # Rotary embedding
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)
        
        # Dropout
        self.attn_dropout = nn.Dropout(p=dropout)
        
        # Ops
        self.softmax = nn.Softmax(axis=-1)
        self.batch_matmul = ops.BatchMatMul()
    
    def construct(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            output: (batch, seq_len, dim)
        """
        B, T, C = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (B, T, dim)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to heads: (B, T, dim) -> (B, H, T, D)
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Apply rotary embeddings
        q, k = self.rotary_emb(q, k, seq_len=T)
        
        # Attention: (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        attn = self.batch_matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)
        
        # Apply to values: (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
        out = self.batch_matmul(attn, v)
        
        # Reshape back: (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


if __name__ == "__main__":
    print("Testing Rotary Position Embeddings")
    print("=" * 60)
    
    # Test RoPE
    print("\n1. Testing RotaryEmbedding...")
    rope = RotaryEmbedding(dim=64, max_seq_len=512)
    
    batch, heads, seq_len, head_dim = 2, 8, 100, 64
    q = Tensor(ops.StandardNormal()((batch, heads, seq_len, head_dim)).asnumpy().astype('float32'))
    k = Tensor(ops.StandardNormal()((batch, heads, seq_len, head_dim)).asnumpy().astype('float32'))
    
    q_rot, k_rot = rope(q, k)
    print(f"   Q: {q.shape} -> {q_rot.shape}")
    print(f"   K: {k.shape} -> {k_rot.shape}")
    print("   ✓ RoPE OK")
    
    # Test RotaryMultiHeadAttention
    print("\n2. Testing RotaryMultiHeadAttention...")
    attn = RotaryMultiHeadAttention(dim=256, num_heads=8)
    
    x = Tensor(ops.StandardNormal()((2, 100, 256)).asnumpy().astype('float32'))
    out = attn(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape}")
    assert out.shape == x.shape
    print("   ✓ Attention OK")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
