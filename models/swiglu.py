"""
SwiGLU Activation
=================
Swish-Gated Linear Unit for feed-forward networks.
Better gradient flow than GELU.
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Tensor


class SwiGLU(nn.Cell):
    """
    Swish-Gated Linear Unit.
    
    SwiGLU(x) = Swish(xW) * (xV)
    
    Better than GELU for deep networks.
    Reference: GLU Variants Improve Transformer (Shazeer, 2020)
    """
    
    def __init__(self):
        super().__init__()
        self.silu = nn.SiLU()
    
    def construct(self, x: Tensor, gate: Tensor) -> Tensor:
        """
        Args:
            x: Main input (batch, seq_len, dim)
            gate: Gate input (batch, seq_len, dim)
        Returns:
            output: (batch, seq_len, dim)
        """
        return self.silu(gate) * x


class SwiGLUFFN(nn.Cell):
    """
    Feed-Forward Network with SwiGLU activation.
    
    Structure:
    x -> [Dense -> split] -> SwiGLU -> Dense -> output
    
    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension (will be split for gating)
        dropout: Dropout rate
    """
    
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        
        hidden_dim = hidden_dim or dim * 4
        
        # First projection creates 2x hidden for gating
        self.w1 = nn.Dense(dim, hidden_dim, has_bias=False)
        self.w2 = nn.Dense(dim, hidden_dim, has_bias=False)  # Gate
        
        # Output projection
        self.w3 = nn.Dense(hidden_dim, dim, has_bias=False)
        
        # Activation
        self.swiglu = SwiGLU()
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
    
    def construct(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            output: (batch, seq_len, dim)
        """
        # SwiGLU: silu(xW2) * (xW1)
        hidden = self.swiglu(self.w1(x), self.w2(x))
        hidden = self.dropout(hidden)
        
        # Output projection
        out = self.w3(hidden)
        out = self.dropout(out)
        
        return out


class GeGLU(nn.Cell):
    """
    GELU-Gated Linear Unit (alternative to SwiGLU).
    
    GeGLU(x) = GELU(xW) * (xV)
    """
    
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()
    
    def construct(self, x: Tensor, gate: Tensor) -> Tensor:
        return self.gelu(gate) * x


class GeGLUFFN(nn.Cell):
    """Feed-Forward Network with GeGLU activation."""
    
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        
        hidden_dim = hidden_dim or dim * 4
        
        self.w1 = nn.Dense(dim, hidden_dim, has_bias=False)
        self.w2 = nn.Dense(dim, hidden_dim, has_bias=False)
        self.w3 = nn.Dense(hidden_dim, dim, has_bias=False)
        
        self.geglu = GeGLU()
        self.dropout = nn.Dropout(p=dropout)
    
    def construct(self, x: Tensor) -> Tensor:
        hidden = self.geglu(self.w1(x), self.w2(x))
        hidden = self.dropout(hidden)
        out = self.w3(hidden)
        out = self.dropout(out)
        return out


if __name__ == "__main__":
    print("Testing SwiGLU Activation")
    print("=" * 60)
    
    # Test SwiGLU
    print("\n1. Testing SwiGLU...")
    swiglu = SwiGLU()
    
    x = Tensor(ops.StandardNormal()((2, 100, 256)).asnumpy().astype('float32'))
    gate = Tensor(ops.StandardNormal()((2, 100, 256)).asnumpy().astype('float32'))
    
    out = swiglu(x, gate)
    print(f"   Input: {x.shape}, Gate: {gate.shape}")
    print(f"   Output: {out.shape}")
    assert out.shape == x.shape
    print("   ✓ SwiGLU OK")
    
    # Test SwiGLUFFN
    print("\n2. Testing SwiGLUFFN...")
    ffn = SwiGLUFFN(dim=256, hidden_dim=512)
    
    out = ffn(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape}")
    assert out.shape == x.shape
    print("   ✓ SwiGLUFFN OK")
    
    # Test GeGLUFFN
    print("\n3. Testing GeGLUFFN...")
    geglu_ffn = GeGLUFFN(dim=256, hidden_dim=512)
    
    out = geglu_ffn(x)
    print(f"   Output: {out.shape}")
    assert out.shape == x.shape
    print("   ✓ GeGLUFFN OK")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
