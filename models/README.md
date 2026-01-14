# Models

Arquitectura Squeezeformer para reconocimiento de señas en MindSpore.

## Componentes

| Archivo | Descripción |
|---------|-------------|
| `rotary_embedding.py` | RoPE - Rotary Position Embeddings |
| `swiglu.py` | SwiGLU/GeGLU activations |
| `squeezeformer_block.py` | Bloques Squeezeformer (Macaron FFN) |
| `cross_attention.py` | Cross-attention entre regiones anatómicas |
| `islr_model_v2.py` | Modelo principal |
| `model_utils.py` | Loss functions y métricas |

## Arquitectura

```
SqueezeformerBlock (Macaron Structure):
    x → LayerNorm → 1/2 FFN → +x
      → LayerNorm → MHSA(RoPE) → +x
      → LayerNorm → Conv → +x
      → LayerNorm → 1/2 FFN → +x
```

## Uso

```python
from models import ISLRModelV2, create_islr_model_v2

# Modelo base
model = ISLRModelV2(
    input_dim=708,
    num_classes=250,
    dim=256,
    num_blocks=6,
    num_heads=8
)

# Factory con variantes
model = create_islr_model_v2(variant='base')   # ~3M params
model = create_islr_model_v2(variant='lite')   # ~2M params
model = create_islr_model_v2(variant='large')  # ~5M params
```

## APIs de MindSpore

### `mindspore.nn`
- `nn.Cell`: Base class
- `nn.Conv1d`, `nn.Dense`: Layers
- `nn.LayerNorm`, `nn.BatchNorm1d`: Normalization
- `nn.GLU`, `nn.SiLU`: Activations

### `mindspore.ops`
- `ops.BatchMatMul`: Attention computation
- `ops.Softmax`: Attention weights
- `ops.ReduceMean`: Pooling
