# Dataset

Datos y vocabulario para el modelo de reconocimiento de señas.

## Contenido

| Archivo | Descripción |
|---------|-------------|
| `sign_vocabulary.json` | Vocabulario de 250 señas con índices |
| `sign_to_prediction_index_map.json` | Mapeo de señas a índices |

## Formato de Landmarks

Los datos de entrada son landmarks extraídos con MediaPipe:

| Campo | Valor |
|-------|-------|
| Total landmarks | 543 |
| Landmarks seleccionados | 118 |
| Features por landmark | 6 (x, y, dx, dy, dx2, dy2) |
| Dimensión final | 708 |

### Distribución de Landmarks

| Región | Cantidad |
|--------|----------|
| Labios | 40 |
| Mano izquierda | 21 |
| Mano derecha | 21 |
| Nariz | 4 |
| Ojo derecho | 16 |
| Ojo izquierdo | 16 |

## Uso

```python
import json

with open('dataset/sign_vocabulary.json') as f:
    data = json.load(f)
    
signs = data['signs']  # Lista de 250 señas
```
