# Preprocessing

Módulos de preprocesamiento de landmarks usando MindSpore.

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `extract_landmarks.py` | Extracción de landmarks con MediaPipe |
| `normalize_data.py` | Normalización usando `mindspore.numpy` |
| `create_dataset.py` | Pipeline de datos con `mindspore.dataset` |
| `preprocess_layer.py` | Capa de preprocesamiento con `mindspore.ops` |

## Pipeline

```
Video → MediaPipe → Landmarks (543×3)
                        │
                        ▼
              Selección (118 landmarks)
                        │
                        ▼
              Normalización (centrado + std)
                        │
                        ▼
              Velocidades (dx, dx2)
                        │
                        ▼
              Features (T × 708)
```

## APIs de MindSpore Utilizadas

### `mindspore.numpy`
```python
import mindspore.numpy as mnp

# Operaciones tipo NumPy en GPU/Ascend
x = mnp.where(mnp.isnan(x), mnp.zeros_like(x), x)
mean = ms_nan_mean(x, axis=(1, 2))
```

### `mindspore.ops`
```python
import mindspore.ops as ops

self.gather = ops.Gather()
self.isnan = ops.IsNan()
self.reduce_mean = ops.ReduceMean()
```

### `mindspore.dataset`
```python
import mindspore.dataset as ds

dataset = ds.GeneratorDataset(source, column_names=["features", "labels"])
dataset = dataset.batch(64, drop_remainder=True)
```

## Uso

```python
from preprocessing.normalize_data import normalize_landmarks_mindspore
from mindspore import Tensor

landmarks = Tensor(...)  # (T, 543, 3)
features = normalize_landmarks_mindspore(landmarks)  # (T, 708)
```
