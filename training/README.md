# Training

Scripts de entrenamiento usando MindSpore.

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `config.py` | Configuración de hiperparámetros |
| `train.py` | Script principal de entrenamiento |

## Uso

```bash
# Entrenamiento básico
python training/train.py

# Con parámetros
python training/train.py --epochs 50 --lr 0.001 --device GPU
```

## Configuración

```python
from training.config import get_default_config

config = get_default_config()
config.train.epochs = 50
config.train.learning_rate = 1e-3
config.train.batch_size = 32
```

## APIs de MindSpore

### `mindspore.train`
```python
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, LossMonitor

trainer = Model(model, loss_fn, optimizer, metrics={'acc': nn.Accuracy()})
trainer.train(epochs, dataset, callbacks=[ModelCheckpoint(), LossMonitor()])
```

### `mindspore.nn.optim`
```python
from mindspore.nn.optim import AdamWeightDecay

optimizer = AdamWeightDecay(
    params=model.trainable_params(),
    learning_rate=1e-3,
    weight_decay=0.05
)
```

### `mindspore.context`
```python
from mindspore import context

context.set_context(
    mode=context.GRAPH_MODE,
    device_target="GPU"  # "CPU", "Ascend"
)
```
