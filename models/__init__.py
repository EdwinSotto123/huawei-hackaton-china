"""
Models Package
==============
Squeezeformer sign language recognition models using MindSpore.
"""

# Core model components
from .rotary_embedding import RotaryEmbedding, RotaryMultiHeadAttention
from .swiglu import SwiGLU, SwiGLUFFN, GeGLU, GeGLUFFN
from .squeezeformer_block import SqueezeformerBlock, SqueezeformerEncoder, ConvolutionModule, TemporalDownsample
from .cross_attention import CrossRegionAttention, RegionalProcessor

# Main model
from .islr_model_v2 import ISLRModelV2, ISLRModelV2Lite, create_islr_model_v2

# Utilities
from .model_utils import LabelSmoothing, FocalLoss, TopKAccuracy

# Default model alias
ISLRModel = ISLRModelV2
create_islr_model = create_islr_model_v2

__all__ = [
    # Core components
    'RotaryEmbedding',
    'RotaryMultiHeadAttention',
    'SwiGLU',
    'SwiGLUFFN',
    'GeGLU',
    'GeGLUFFN',
    'SqueezeformerBlock',
    'SqueezeformerEncoder',
    'ConvolutionModule',
    'TemporalDownsample',
    'CrossRegionAttention',
    'RegionalProcessor',
    
    # Main models
    'ISLRModelV2',
    'ISLRModelV2Lite',
    'create_islr_model_v2',
    
    # Aliases (for backward compatibility)
    'ISLRModel',
    'create_islr_model',
    
    # Utilities
    'LabelSmoothing',
    'FocalLoss',
    'TopKAccuracy',
]
