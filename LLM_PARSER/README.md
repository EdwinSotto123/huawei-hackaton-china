# 🗣️ LLM Parser - Sign to Text

Convierte predicciones de lenguaje de señas en texto natural y enriquecido usando **DeepSeek LLM**.

## 📋 Descripción

El modelo de reconocimiento de señas produce secuencias de palabras como:
```
YO NOMBRE EDWIN
```

Este módulo las transforma en oraciones naturales:
```
Me llamo Edwin
```

## 🚀 Instalación

```bash
pip install requests python-dotenv
```

## 🔑 Configuración

Configura tu API key de DeepSeek:

```bash
# Opción 1: Variable de entorno
export DEEPSEEK_API_KEY="tu-api-key"

# Opción 2: Archivo .env
echo "DEEPSEEK_API_KEY=tu-api-key" > .env
```

Obtén tu API key en: https://platform.deepseek.com/

## 📖 Uso

### Uso Básico

```python
from LLM_PARSER import SignToTextParser, parse_signs

# Función rápida
result = parse_signs("YO NOMBRE EDWIN")
print(result)  # "Me llamo Edwin"

# Con parser
parser = SignToTextParser()
result = parser.parse("HOLA COMO ESTAR TU")
print(result)  # "¡Hola! ¿Cómo estás?"
```

### Con Contexto

```python
parser = SignToTextParser()

# El contexto ayuda al LLM a dar mejor respuesta
result = parser.parse("DONDE BAÑO", context="en un restaurante")
print(result)  # "¿Dónde está el baño, por favor?"

result = parser.parse("QUERER COMER", context="es mediodía")
print(result)  # "Me gustaría comer algo, es hora del almuerzo"
```

### Estilos de Salida

```python
from LLM_PARSER import SignToTextParser, OutputStyle

# Casual (default)
parser = SignToTextParser(style=OutputStyle.CASUAL)
print(parser.parse("HOLA"))  # "¡Hola!"

# Formal
parser = SignToTextParser(style=OutputStyle.FORMAL)
print(parser.parse("HOLA"))  # "Buenos días"

# Expresivo (con emojis)
parser = SignToTextParser(style=OutputStyle.EXPRESSIVE)
print(parser.parse("HOLA"))  # "¡¡Holaaaa!! 👋😄"

# Mínimo
parser = SignToTextParser(style=OutputStyle.MINIMAL)
print(parser.parse("HOLA"))  # "Hola"
```

### Múltiples Idiomas

```python
# Español (default)
parser = SignToTextParser(language="es")
print(parser.parse("YO NOMBRE EDWIN"))  # "Me llamo Edwin"

# Inglés
parser = SignToTextParser(language="en")
print(parser.parse("I NAME EDWIN"))  # "My name is Edwin"
```

### Procesamiento por Lotes

```python
parser = SignToTextParser()

predictions = [
    "YO NOMBRE EDWIN",
    "HOLA COMO ESTAR",
    "GRACIAS AYUDA"
]

results = parser.parse_batch(predictions)
for pred, result in zip(predictions, results):
    print(f"{pred} → {result}")
```

### Streaming (tiempo real)

```python
parser = SignToTextParser()

print("Respuesta: ", end="")
for chunk in parser.parse_stream("HOLA COMO ESTAR TU"):
    print(chunk, end="", flush=True)
print()
```

## 🔗 Integración con el Modelo

```python
from mindspore_hackaton.models import ISLRModelV2
from mindspore_hackaton.LLM_PARSER import SignToTextParser
import numpy as np

# Cargar modelo
model = ISLRModelV2(num_classes=250)
parser = SignToTextParser()

# Vocabulario (250 clases)
vocab = ["HOLA", "GRACIAS", "YO", "TU", "NOMBRE", ...]  # tu vocabulario

def predict_and_parse(landmarks):
    # Predicción del modelo
    logits = model(landmarks)
    indices = np.argsort(logits.asnumpy(), axis=-1)[:, -5:]  # Top-5
    
    # Construir secuencia de palabras
    words = [vocab[i] for i in indices.flatten() if logits[0, i] > threshold]
    raw_prediction = " ".join(words)
    
    # Convertir a texto natural
    natural_text = parser.parse(raw_prediction)
    
    return natural_text

# Uso
text = predict_and_parse(input_landmarks)
print(text)  # "¡Hola! Me llamo Edwin"
```

## 📊 Ejemplos de Conversión

| Input (Predicción) | Output (Texto Natural) |
|--------------------|------------------------|
| `YO NOMBRE EDWIN` | Me llamo Edwin |
| `HOLA COMO ESTAR TU` | ¡Hola! ¿Cómo estás? |
| `GRACIAS MUCHO AYUDA` | ¡Muchas gracias por tu ayuda! |
| `DONDE BAÑO` | ¿Dónde está el baño? |
| `YO QUERER AGUA` | Quiero agua, por favor |
| `NO ENTENDER YO` | No entiendo |
| `TU BONITO` | Eres muy bonito/a |
| `YO IR CASA` | Me voy a casa |
| `MUCHO GUSTO CONOCER` | ¡Mucho gusto en conocerte! |

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────┐
│                    LLM_PARSER                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌───────────────┐      ┌──────────────────────┐   │
│  │ Raw Prediction│ ───▶ │   SignToTextParser   │   │
│  │ "YO EDWIN"    │      │                      │   │
│  └───────────────┘      │  • preprocess()      │   │
│                         │  • build_prompt()    │   │
│                         │  • call_llm()        │   │
│                         └──────────┬───────────┘   │
│                                    │               │
│                                    ▼               │
│                         ┌──────────────────────┐   │
│                         │   DeepSeekClient     │   │
│                         │                      │   │
│                         │  • chat()            │   │
│                         │  • chat_stream()     │   │
│                         └──────────┬───────────┘   │
│                                    │               │
│                                    ▼               │
│                         ┌──────────────────────┐   │
│                         │   DeepSeek API       │   │
│                         │   (deepseek-chat)    │   │
│                         └──────────┬───────────┘   │
│                                    │               │
│                                    ▼               │
│                         ┌──────────────────────┐   │
│                         │   Natural Text       │   │
│                         │ "Me llamo Edwin"     │   │
│                         └──────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## 📁 Estructura

```
LLM_PARSER/
├── __init__.py           # Exports principales
├── deepseek_client.py    # Cliente API DeepSeek
├── parser.py             # SignToTextParser principal
└── README.md             # Esta documentación
```

## ⚙️ Configuración Avanzada

```python
parser = SignToTextParser(
    api_key="sk-...",           # API key (o usar env var)
    language="es",              # Idioma de salida
    style=OutputStyle.CASUAL,   # Estilo de texto
    model="deepseek-chat",      # Modelo a usar
    temperature=0.7             # Creatividad (0.0-1.0)
)
```

## 🧪 Testing

```bash
# Configurar API key
export DEEPSEEK_API_KEY="tu-api-key"

# Ejecutar tests
python -m LLM_PARSER.parser
```

## 📄 Licencia

Apache 2.0
