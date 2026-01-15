# 🤟 Sign Language Recognition App

Aplicación completa de reconocimiento de lenguaje de señas con salida de voz en tiempo real.

## 📋 Pipeline Completo

```
📹 Cámara → 🖐️ MediaPipe → 🧠 Squeezeformer → 📝 DeepSeek LLM → 🔊 ElevenLabs TTS
```

## 🚀 Instalación

```bash
# Dependencias base
pip install opencv-python mediapipe numpy

# Para audio
pip install pygame

# Para LLM y TTS (opcional, mejora la experiencia)
pip install openai requests
```

## 🔑 Configuración de API Keys

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "tu-deepseek-key"
$env:ELEVENLABS_API_KEY = "tu-elevenlabs-key"

# Linux/Mac
export DEEPSEEK_API_KEY="tu-deepseek-key"
export ELEVENLABS_API_KEY="tu-elevenlabs-key"
```

## ▶️ Ejecución

```bash
# Desde la carpeta mindspore_hackaton
cd mindspore_hackaton

# Ejecutar app
python -m app.main

# Con opciones
python -m app.main --camera 0 --language es --voice spanish_female
```

## 🎮 Controles

| Tecla | Acción |
|-------|--------|
| `SPACE` | Forzar predicción ahora |
| `R` | Reiniciar buffer |
| `M` | Silenciar/Activar audio |
| `S` | Cambiar estilo (casual/formal/expresivo) |
| `Q` / `ESC` | Salir |

## 📊 Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                         SIGN LANGUAGE APP                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐      ┌─────────────────────────────────────┐   │
│  │   CAMERA    │      │           CAMERA CAPTURE            │   │
│  │   📹        │ ───▶ │  • OpenCV VideoCapture              │   │
│  │             │      │  • 640x480 @ 30fps                  │   │
│  └─────────────┘      └──────────────┬──────────────────────┘   │
│                                      │                           │
│                                      ▼                           │
│                       ┌─────────────────────────────────────┐   │
│                       │         MEDIAPIPE HOLISTIC          │   │
│                       │  • Face: 468 landmarks              │   │
│                       │  • Hands: 21×2 landmarks            │   │
│                       │  • Pose: 33 landmarks               │   │
│                       │  → Selected: 118 landmarks          │   │
│                       └──────────────┬──────────────────────┘   │
│                                      │                           │
│                                      ▼                           │
│                       ┌─────────────────────────────────────┐   │
│                       │        FEATURE EXTRACTION           │   │
│                       │  • Position (x, y): 236 features    │   │
│                       │  • Velocity (dx, dy): 236 features  │   │
│                       │  • Acceleration: 236 features       │   │
│                       │  → Total: 708 features/frame        │   │
│                       └──────────────┬──────────────────────┘   │
│                                      │                           │
│                                      ▼                           │
│                       ┌─────────────────────────────────────┐   │
│                       │       FRAME BUFFER (64 frames)      │   │
│                       │  Input: (64, 708) → (1, 64, 708)    │   │
│                       └──────────────┬──────────────────────┘   │
│                                      │                           │
│                                      ▼                           │
│                       ┌─────────────────────────────────────┐   │
│                       │      SQUEEZEFORMER MODEL 🧠         │   │
│                       │  • RegionalProcessor                │   │
│                       │  • Squeezeformer ×6 blocks          │   │
│                       │  • Classification → 250 classes     │   │
│                       │  → Output: "YO NOMBRE EDWIN"        │   │
│                       └──────────────┬──────────────────────┘   │
│                                      │                           │
│                                      ▼                           │
│                       ┌─────────────────────────────────────┐   │
│                       │        LLM PARSER (DeepSeek) 📝     │   │
│                       │  • Input: "YO NOMBRE EDWIN"         │   │
│                       │  • Output: "Me llamo Edwin"         │   │
│                       │  • Styles: casual/formal/expressive │   │
│                       └──────────────┬──────────────────────┘   │
│                                      │                           │
│                                      ▼                           │
│                       ┌─────────────────────────────────────┐   │
│                       │      TTS SERVICE (ElevenLabs) 🔊    │   │
│                       │  • Input: "Me llamo Edwin"          │   │
│                       │  • Output: Audio MP3                │   │
│                       │  • Voice: spanish_female            │   │
│                       └──────────────┬──────────────────────┘   │
│                                      │                           │
│                                      ▼                           │
│                       ┌─────────────────────────────────────┐   │
│                       │         AUDIO PLAYBACK 🎵           │   │
│                       │  • pygame.mixer                     │   │
│                       │  • Real-time playback               │   │
│                       └─────────────────────────────────────┘   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

## 📁 Estructura

```
app/
├── __init__.py      # Module exports
├── main.py          # SignLanguageApp - UI principal
├── camera.py        # CameraCapture - Webcam + MediaPipe
├── pipeline.py      # SignToSpeechPipeline - Model→LLM→TTS
└── README.md        # Esta documentación
```

## 🔧 Uso Programático

```python
from app import SignLanguageApp, SignToSpeechPipeline, CameraCapture

# Opción 1: App completa con UI
app = SignLanguageApp(language="es", voice="spanish_female")
app.run()

# Opción 2: Solo pipeline
pipeline = SignToSpeechPipeline()

# Añadir frames
for features in feature_generator:
    pipeline.add_frame(features)

# Obtener predicción
result = pipeline.finalize()
print(result.natural_text)   # "Me llamo Edwin"
play(result.audio_data)      # Reproducir audio

# Opción 3: Solo cámara
camera = CameraCapture()
for frame_data in camera.stream():
    print(frame_data.features.shape)  # (708,)
    if frame_data.has_hands:
        process(frame_data)
```

## 🎤 Voces Disponibles

| Voz | Idioma | Género |
|-----|--------|--------|
| `spanish_female` | Español | Femenino |
| `spanish_male` | Español | Masculino |
| `rachel` | Inglés | Femenino |
| `josh` | Inglés | Masculino |

## ✨ Estilos de Texto

| Estilo | Ejemplo |
|--------|---------|
| `casual` | "¡Hola! Me llamo Edwin" |
| `formal` | "Buenos días. Mi nombre es Edwin" |
| `expressive` | "¡¡Holaaaa!! 👋 Soy Edwin!" |
| `minimal` | "Soy Edwin" |

## 🐛 Troubleshooting

### Cámara no detectada
```bash
# Probar diferentes IDs
python -m app.main --camera 1
python -m app.main --camera 2
```

### Sin audio
```bash
# Instalar pygame
pip install pygame

# Verificar API key
echo $ELEVENLABS_API_KEY
```

### LLM no funciona
```bash
# Verificar API key
echo $DEEPSEEK_API_KEY

# Probar directamente
python -m LLM_PARSER.parser
```

### Modelo lento
- La primera predicción es más lenta (carga del modelo)
- Subsecuentes son más rápidas
- Usar GPU mejora rendimiento

## 📄 Licencia

Apache 2.0
