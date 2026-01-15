"""
TTS Service Module
===================
Text-to-Speech using ElevenLabs API.

Example:
    from TTS_SERVICE import TextToSpeech, speak
    
    # Quick function
    audio_b64 = speak("Hola, me llamo Edwin")
    
    # Full service
    tts = TextToSpeech()
    audio_bytes = tts.synthesize("¡Hola! ¿Cómo estás?", voice="spanish_female")
"""

from .elevenlabs_client import ElevenLabsClient, Voice, VOICES
from .tts import TextToSpeech, speak, speak_to_file

__all__ = [
    "TextToSpeech",
    "ElevenLabsClient",
    "Voice",
    "VOICES",
    "speak",
    "speak_to_file"
]
