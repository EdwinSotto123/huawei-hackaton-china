# Tests

Tests unitarios para los componentes del modelo.

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `test_model.py` | Tests de componentes Squeezeformer |

## Uso

```bash
# Ejecutar todos los tests
python tests/test_model.py

# O con pytest
pytest tests/ -v
```

## Cobertura

| Componente | Tests |
|------------|-------|
| RotaryEmbedding | ✓ |
| SwiGLU | ✓ |
| SqueezeformerBlock | ✓ |
| CrossRegionAttention | ✓ |
| ISLRModelV2 | ✓ |
| ISLRModelV2Lite | ✓ |
