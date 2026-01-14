# Inference

Demo de inferencia en tiempo real.

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `webcam_demo.py` | Demo de webcam en tiempo real |

## Uso

```bash
# Demo básica
python inference/webcam_demo.py

# Con checkpoint
python inference/webcam_demo.py --checkpoint outputs/checkpoints/model.ckpt

# Con cámara específica
python inference/webcam_demo.py --camera 1
```

## Controles

| Tecla | Acción |
|-------|--------|
| `Q` | Salir |

## Pipeline

```
Webcam → MediaPipe → Landmarks
                         │
                         ▼
              MindSpore Preprocessing
                         │
                         ▼
              Squeezeformer Model
                         │
                         ▼
              Predicción (250 clases)
```

## Carga del Modelo

```python
from models import ISLRModelV2
from mindspore import load_checkpoint, load_param_into_net

model = ISLRModelV2(num_classes=250)
param_dict = load_checkpoint("model.ckpt")
load_param_into_net(model, param_dict)
```
