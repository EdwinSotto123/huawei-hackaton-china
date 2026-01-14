# Outputs

Directorio para checkpoints y logs de entrenamiento.

## Estructura

```
outputs/
├── checkpoints/   # Modelos guardados (.ckpt)
├── logs/          # Logs de entrenamiento
└── results/       # Resultados de evaluación
```

## Checkpoints

Los checkpoints se guardan automáticamente durante el entrenamiento:

```python
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

config = CheckpointConfig(
    save_checkpoint_steps=1000,
    keep_checkpoint_max=5
)
ckpt_cb = ModelCheckpoint(prefix="squeezeformer", directory="outputs/checkpoints")
```

## Cargar Checkpoint

```python
from mindspore import load_checkpoint, load_param_into_net
from models import ISLRModelV2

model = ISLRModelV2()
param_dict = load_checkpoint("outputs/checkpoints/squeezeformer-10_1000.ckpt")
load_param_into_net(model, param_dict)
```
