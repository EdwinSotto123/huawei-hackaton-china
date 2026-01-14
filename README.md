# MindSpore Sign Language Recognition

Sistema de reconocimiento de lenguaje de señas usando **MindSpore** con arquitectura **Squeezeformer**.

## Características

- **Squeezeformer**: Arquitectura moderna con estructura Macaron FFN
- **RoPE**: Rotary Position Embeddings para codificación temporal
- **Cross-Region Attention**: Atención cruzada entre manos y cara
- **SwiGLU**: Activación mejorada para feed-forward networks

## Arquitectura

```
Input: (batch, T, 708)
         │
    ┌────┴────────────┐
    │RegionalProcessor│ → Cross-Attention Manos↔Cara
    └────┬────────────┘
         │
    ┌────┴─────────────┐
    │Squeezeformer ×6  │ → Macaron FFN + RoPE + Conv
    │+ Temporal Squeeze│
    └────┬─────────────┘
         │
    ┌────┴────────┐
    │GlobalPool   │ → ops.ReduceMean
    │+ Classifier │ → nn.Dense(256, 250)
    └────┬────────┘
         │
    Output: 250 clases
```

## Estructura del Proyecto

```
├── dataset/           # Datos y vocabulario
├── preprocessing/     # Extracción y normalización de landmarks
├── models/            # Arquitectura Squeezeformer
├── training/          # Scripts de entrenamiento
├── inference/         # Demo en tiempo real
└── outputs/           # Checkpoints y logs
```

## Instalación

```bash
pip install mindspore mediapipe opencv-python
```

## Uso

### Entrenamiento
```bash
python training/train.py --epochs 50
```

### Demo
```bash
python inference/webcam_demo.py
```

## APIs de MindSpore

| Módulo | APIs Utilizadas |
|--------|-----------------|
| `mindspore.nn` | Cell, Conv1d, Dense, LayerNorm, GLU, SiLU |
| `mindspore.ops` | BatchMatMul, Gather, ReduceMean, TopK |
| `mindspore.numpy` | where, isnan, concatenate, reshape |
| `mindspore.dataset` | GeneratorDataset, NumpySlicesDataset |
| `mindspore.train` | Model, callbacks |

## Modelo

```python
from models import ISLRModelV2, create_islr_model_v2

model = ISLRModelV2(num_classes=250)
# o
model = create_islr_model_v2(variant='base')  # 'lite', 'large'
```

## Licencia

Apache 2.0
