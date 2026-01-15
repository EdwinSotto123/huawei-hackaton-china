# 🎤 TTS Service - Text to Speech

Convierte texto natural a audio usando **ElevenLabs API**.

## 📋 Descripción

Este servicio toma el texto generado por el LLM Parser y lo convierte en audio hablado de alta calidad.

```
"Me llamo Edwin" → 🔊 Audio MP3
```

## 🚀 Instalación

```bash
pip install requests
```

## 🔑 Configuración

Configura tu API key de ElevenLabs:

```bash
# Variable de entorno
export ELEVENLABS_API_KEY="tu-api-key"

# O archivo .env
echo "ELEVENLABS_API_KEY=tu-api-key" >> .env
```

Obtén tu API key en: https://elevenlabs.io/

## 📖 Uso

### Uso Rápido

```python
from TTS_SERVICE import speak, speak_to_file

# Obtener audio en base64 (para web)
audio_b64 = speak("¡Hola! Me llamo Edwin")

# Guardar a archivo
speak_to_file("Gracias por tu ayuda", "output.mp3")
```

### Servicio Completo

```python
from TTS_SERVICE import TextToSpeech

tts = TextToSpeech(voice="spanish_female")

# Audio bytes
audio = tts.synthesize("¿Cómo estás?")

# Base64 para web
audio_b64 = tts.synthesize_base64("Hola mundo")

# Resultado completo con metadata
result = tts.synthesize_full("Muchas gracias")
print(result.duration_estimate)  # Duración estimada
print(result.to_data_url())      # Data URL para HTML

# Guardar a archivo
tts.synthesize_to_file("Adiós", "farewell.mp3")
```

### Voces Disponibles

```python
from TTS_SERVICE import TextToSpeech, VOICES

# Ver voces disponibles
print(VOICES)

# Crear con voz específica
tts = TextToSpeech(voice="spanish_male")      # Voz masculina español
tts = TextToSpeech(voice="spanish_female")    # Voz femenina español
tts = TextToSpeech(voice="rachel")            # Rachel (inglés)
tts = TextToSpeech(voice="josh")              # Josh (inglés, grave)

# Cambiar voz
tts.set_voice("spanish_male")
```

### Voces Pre-configuradas

| Nombre | Género | Descripción |
|--------|--------|-------------|
| `spanish_female` | Femenino | Matilda - Cálida, ideal para español |
| `spanish_male` | Masculino | Antoni - Versátil, buen español |
| `rachel` | Femenino | Calmada, americana |
| `josh` | Masculino | Voz grave, americana |
| `bella` | Femenino | Suave, americana |
| `adam` | Masculino | Profunda, americana |

### Streaming (Tiempo Real)

```python
tts = TextToSpeech()

# Para audio en tiempo real
for chunk in tts.synthesize_stream("Este es un texto largo..."):
    # Procesar/reproducir chunk
    audio_player.feed(chunk)
```

### Uso en Web (HTML)

```python
result = tts.synthesize_full("¡Hola!")

# Generar HTML con audio
html = f'''
<audio controls autoplay>
    <source src="{result.to_data_url()}" type="audio/mpeg">
</audio>
'''
```

## 🔗 Pipeline Completo: Señas → Texto → Audio

```python
from mindspore_hackaton.models import ISLRModelV2
from mindspore_hackaton.LLM_PARSER import SignToTextParser
from mindspore_hackaton.TTS_SERVICE import TextToSpeech

# Inicializar servicios
model = ISLRModelV2(num_classes=250)
parser = SignToTextParser()
tts = TextToSpeech(voice="spanish_female")

def sign_to_speech(landmarks):
    # 1. Reconocer señas
    logits = model(landmarks)
    raw_prediction = decode_to_words(logits)  # "YO NOMBRE EDWIN"
    
    # 2. Convertir a texto natural
    natural_text = parser.parse(raw_prediction)  # "Me llamo Edwin"
    
    # 3. Generar audio
    audio = tts.synthesize(natural_text)
    
    return audio, natural_text

# Uso
audio, text = sign_to_speech(input_landmarks)
print(f"Texto: {text}")
# Reproducir audio...
```

## 📊 Arquitectura

```
┌───────────────────────────────────────────────────────┐
│                    TTS_SERVICE                        │
├───────────────────────────────────────────────────────┤
│                                                       │
│  ┌───────────────┐      ┌──────────────────────────┐ │
│  │  Natural Text │ ───▶ │    TextToSpeech          │ │
│  │ "Me llamo..." │      │                          │ │
│  └───────────────┘      │  • synthesize()          │ │
│                         │  • synthesize_base64()   │ │
│                         │  • synthesize_stream()   │ │
│                         └───────────┬──────────────┘ │
│                                     │                │
│                                     ▼                │
│                         ┌──────────────────────────┐ │
│                         │   ElevenLabsClient       │ │
│                         │                          │ │
│                         │  • eleven_multilingual   │ │
│                         │  • Voice selection       │ │
│                         └───────────┬──────────────┘ │
│                                     │                │
│                                     ▼                │
│                         ┌──────────────────────────┐ │
│                         │   ElevenLabs API         │ │
│                         └───────────┬──────────────┘ │
│                                     │                │
│                                     ▼                │
│                         ┌──────────────────────────┐ │
│                         │   Audio Output           │ │
│                         │ • bytes (MP3)            │ │
│                         │ • base64 (web)           │ │
│                         │ • file (.mp3)            │ │
│                         └──────────────────────────┘ │
│                                                       │
└───────────────────────────────────────────────────────┘
```

## 📁 Estructura

```
TTS_SERVICE/
├── __init__.py           # Exports principales
├── elevenlabs_client.py  # Cliente API ElevenLabs
├── tts.py                # TextToSpeech principal
└── README.md             # Esta documentación
```

## ⚙️ Configuración Avanzada

```python
tts = TextToSpeech(
    api_key="tu-key",                    # API key
    voice="spanish_female",              # Voz por defecto
    model="eleven_multilingual_v2",      # Modelo (mejor para español)
    stability=0.5,                       # Estabilidad de voz (0-1)
    similarity_boost=0.75                # Claridad de voz (0-1)
)
```

### Modelos Disponibles

| Modelo | Descripción |
|--------|-------------|
| `eleven_multilingual_v2` | Mejor para español y multilingüe |
| `eleven_turbo_v2` | Más rápido, buena calidad |
| `eleven_monolingual_v1` | Solo inglés, muy rápido |

## 🧪 Testing

```bash
# Configurar API key
export ELEVENLABS_API_KEY="tu-api-key"

# Ejecutar tests
python -m TTS_SERVICE.tts
```

## 💰 Costos

ElevenLabs tiene plan gratuito con ~10,000 caracteres/mes. 
Más info: https://elevenlabs.io/pricing

## 📄 Licencia

Apache 2.0
