"""
Cross-Region Attention
======================
Cross-attention between anatomical regions (hands, face).
Captures correlations between hand gestures and facial expressions.
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Tensor
from typing import Tuple


# Region definitions (indices into 118 landmarks × 2 for x,y)
# After flattening: (T, 118, 2) -> (T, 236) for positions only
LIP_INDICES = list(range(0, 40))           # 0-39: lips
LHAND_INDICES = list(range(40, 61))        # 40-60: left hand  
RHAND_INDICES = list(range(61, 82))        # 61-81: right hand
NOSE_INDICES = list(range(82, 86))         # 82-85: nose
REYE_INDICES = list(range(86, 102))        # 86-101: right eye
LEYE_INDICES = list(range(102, 118))       # 102-117: left eye

# Grouped regions
HAND_INDICES = LHAND_INDICES + RHAND_INDICES  # 42 landmarks
FACE_INDICES = LIP_INDICES + NOSE_INDICES + REYE_INDICES + LEYE_INDICES  # 76 landmarks


class CrossRegionAttention(nn.Cell):
    """
    Cross-attention between anatomical regions.
    
    Allows hands to attend to face features and vice versa.
    This captures important correlations in sign language
    (e.g., mouth shape + hand gesture = meaning).
    
    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections for hands
        self.hand_q = nn.Dense(dim, dim, has_bias=False)
        self.hand_k = nn.Dense(dim, dim, has_bias=False)
        self.hand_v = nn.Dense(dim, dim, has_bias=False)
        
        # Projections for face
        self.face_q = nn.Dense(dim, dim, has_bias=False)
        self.face_k = nn.Dense(dim, dim, has_bias=False)
        self.face_v = nn.Dense(dim, dim, has_bias=False)
        
        # Output projections
        self.hand_out = nn.Dense(dim, dim)
        self.face_out = nn.Dense(dim, dim)
        
        # Layer norms
        self.hand_norm = nn.LayerNorm([dim])
        self.face_norm = nn.LayerNorm([dim])
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(axis=-1)
        self.batch_matmul = ops.BatchMatMul()
    
    def _cross_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Compute cross-attention."""
        B, T, C = query.shape
        T_kv = key.shape[1]
        
        # Reshape to heads: (B, T, C) -> (B, H, T, D)
        q = query.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = key.reshape(B, T_kv, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = value.reshape(B, T_kv, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Attention: (B, H, T, D) @ (B, H, D, T_kv) -> (B, H, T, T_kv)
        attn = self.batch_matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        
        # Apply to values: (B, H, T, T_kv) @ (B, H, T_kv, D) -> (B, H, T, D)
        out = self.batch_matmul(attn, v)
        
        # Reshape back: (B, H, T, D) -> (B, T, C)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        return out
    
    def construct(self, hand_feats: Tensor, face_feats: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Cross-attention between hand and face features.
        
        Args:
            hand_feats: (batch, seq_len, dim) hand features
            face_feats: (batch, seq_len, dim) face features
            
        Returns:
            enhanced_hands: (batch, seq_len, dim)
            enhanced_face: (batch, seq_len, dim)
        """
        # Normalize
        hand_normed = self.hand_norm(hand_feats)
        face_normed = self.face_norm(face_feats)
        
        # Hands attend to face
        hand_q = self.hand_q(hand_normed)
        face_k = self.face_k(face_normed)
        face_v = self.face_v(face_normed)
        hand_cross = self._cross_attention(hand_q, face_k, face_v)
        hand_cross = self.hand_out(hand_cross)
        enhanced_hands = hand_feats + self.dropout(hand_cross)
        
        # Face attends to hands
        face_q = self.face_q(face_normed)
        hand_k = self.hand_k(hand_normed)
        hand_v = self.hand_v(hand_normed)
        face_cross = self._cross_attention(face_q, hand_k, hand_v)
        face_cross = self.face_out(face_cross)
        enhanced_face = face_feats + self.dropout(face_cross)
        
        return enhanced_hands, enhanced_face


class RegionalProcessor(nn.Cell):
    """
    Process features by anatomical region with cross-attention.
    
    Splits input by region, processes each, applies cross-attention,
    then recombines.
    
    Args:
        input_dim: Input feature dimension (708)
        hidden_dim: Hidden dimension for processing
        num_heads: Number of attention heads
    """
    
    def __init__(self, 
                 input_dim: int = 708,
                 hidden_dim: int = 256,
                 num_heads: int = 4):
        super().__init__()
        
        # Input: (T, 708) = (T, 118*6) where 6 = x,y,dx,dy,dx2,dy2
        # Each landmark has 6 features
        self.landmarks_per_region = {
            'hands': 42,   # 21 left + 21 right
            'face': 76     # 40 lip + 4 nose + 16 reye + 16 leye
        }
        
        # Calculate feature dimensions
        # Hands: 42 landmarks * 6 features = 252
        # Face: 76 landmarks * 6 features = 456
        hand_dim = self.landmarks_per_region['hands'] * 6  # 252
        face_dim = self.landmarks_per_region['face'] * 6   # 456
        
        # Embeddings for each region
        self.hand_embed = nn.Dense(hand_dim, hidden_dim, has_bias=False)
        self.face_embed = nn.Dense(face_dim, hidden_dim, has_bias=False)
        
        # Cross-region attention
        self.cross_attn = CrossRegionAttention(hidden_dim, num_heads)
        
        # Output projection
        self.output_proj = nn.Dense(hidden_dim * 2, hidden_dim)
        self.output_norm = nn.LayerNorm([hidden_dim])
    
    def _split_regions(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Split features into hand and face regions.
        
        Input x: (B, T, 708) where 708 = 118 landmarks * 6 features
        Features are ordered: [x, y] for 118 landmarks, then [dx, dy], then [dx2, dy2]
        So: [0:236] = positions, [236:472] = velocities, [472:708] = accelerations
        
        For each group of 118 landmarks:
        - 0-39: lips
        - 40-60: left hand
        - 61-81: right hand
        - 82-85: nose
        - 86-101: right eye
        - 102-117: left eye
        """
        B, T, C = x.shape
        
        # Reshape to (B, T, 3, 118, 2) = batch, time, (pos/vel/acc), landmarks, (x/y)
        x_reshaped = x.reshape(B, T, 3, 118, 2)
        
        # Hand indices: 40-60 (left) + 61-81 (right) = total 42
        hand_idx = list(range(40, 82))  # Left + Right hand
        face_idx = list(range(0, 40)) + list(range(82, 118))  # Lips + Nose + Eyes
        
        # Extract regions
        hands = x_reshaped[:, :, :, hand_idx, :]  # (B, T, 3, 42, 2)
        face = x_reshaped[:, :, :, face_idx, :]   # (B, T, 3, 76, 2)
        
        # Flatten: (B, T, 3, N, 2) -> (B, T, 3*N*2)
        hands = hands.reshape(B, T, -1)  # (B, T, 252)
        face = face.reshape(B, T, -1)    # (B, T, 456)
        
        return hands, face
    
    def construct(self, x: Tensor) -> Tensor:
        """
        Process input with regional cross-attention.
        
        Args:
            x: (batch, seq_len, 708) raw features
            
        Returns:
            output: (batch, seq_len, hidden_dim) enhanced features
        """
        # Split into regions
        hands, face = self._split_regions(x)
        
        # Embed each region
        hand_feats = self.hand_embed(hands)  # (B, T, hidden)
        face_feats = self.face_embed(face)   # (B, T, hidden)
        
        # Cross-region attention
        hand_enhanced, face_enhanced = self.cross_attn(hand_feats, face_feats)
        
        # Concatenate and project
        combined = mnp.concatenate([hand_enhanced, face_enhanced], axis=-1)
        output = self.output_proj(combined)
        output = self.output_norm(output)
        
        return output


if __name__ == "__main__":
    print("Testing Cross-Region Attention")
    print("=" * 60)
    
    # Test CrossRegionAttention
    print("\n1. Testing CrossRegionAttention...")
    cross_attn = CrossRegionAttention(dim=256, num_heads=4)
    
    hand_feats = Tensor(ops.StandardNormal()((2, 100, 256)).asnumpy().astype('float32'))
    face_feats = Tensor(ops.StandardNormal()((2, 100, 256)).asnumpy().astype('float32'))
    
    hand_out, face_out = cross_attn(hand_feats, face_feats)
    print(f"   Hand: {hand_feats.shape} -> {hand_out.shape}")
    print(f"   Face: {face_feats.shape} -> {face_out.shape}")
    assert hand_out.shape == hand_feats.shape
    assert face_out.shape == face_feats.shape
    print("   ✓ CrossRegionAttention OK")
    
    # Test RegionalProcessor
    print("\n2. Testing RegionalProcessor...")
    processor = RegionalProcessor(input_dim=708, hidden_dim=256)
    
    x = Tensor(ops.StandardNormal()((2, 100, 708)).asnumpy().astype('float32'))
    out = processor(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape}")
    assert out.shape == (2, 100, 256)
    print("   ✓ RegionalProcessor OK")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
