"""
Text-to-Speech Service
======================
High-level TTS service using ElevenLabs.

Examples:
    # Quick usage
    from TTS_SERVICE import speak, speak_to_file
    
    audio_b64 = speak("¡Hola! ¿Cómo estás?")
    speak_to_file("Gracias por tu ayuda", "output.mp3")
    
    # Full service
    from TTS_SERVICE import TextToSpeech
    
    tts = TextToSpeech(voice="spanish_female")
    audio = tts.synthesize("Me llamo Edwin")
"""

import os
import base64
from typing import Optional, Union, Dict, List
from dataclasses import dataclass
from pathlib import Path

from .elevenlabs_client import ElevenLabsClient, Voice, VOICES


@dataclass
class TTSResult:
    """Result from TTS synthesis."""
    audio_bytes: bytes
    audio_base64: str
    format: str
    duration_estimate: float  # Rough estimate in seconds
    text: str
    voice: str
    
    def save(self, path: Union[str, Path]) -> Path:
        """Save audio to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(self.audio_bytes)
        return path
    
    def to_data_url(self) -> str:
        """Get data URL for embedding in HTML/web."""
        mime = "audio/mpeg" if self.format == "mp3" else f"audio/{self.format}"
        return f"data:{mime};base64,{self.audio_base64}"


class TextToSpeech:
    """
    Text-to-Speech service using ElevenLabs.
    
    Usage:
        tts = TextToSpeech()
        
        # Get audio bytes
        audio = tts.synthesize("¡Hola mundo!")
        
        # Get base64 for web
        result = tts.synthesize_full("Hola")
        html_audio = f'<audio src="{result.to_data_url()}" controls></audio>'
        
        # Save to file
        tts.synthesize_to_file("Gracias", "output.mp3")
        
        # Different voices
        tts = TextToSpeech(voice="spanish_male")
        tts.set_voice("rachel")  # Change voice
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: str = "spanish_female",
        model: str = "eleven_multilingual_v2",
        stability: float = 0.5,
        similarity_boost: float = 0.75
    ):
        """
        Initialize TTS service.
        
        Args:
            api_key: ElevenLabs API key (or set ELEVENLABS_API_KEY env var)
            voice: Voice name (see VOICES dict) or voice ID
            model: ElevenLabs model:
                - eleven_multilingual_v2 (best for Spanish)
                - eleven_turbo_v2 (faster, good quality)
            stability: Voice stability (0.0-1.0)
            similarity_boost: Voice clarity (0.0-1.0)
        """
        self.client = ElevenLabsClient(
            api_key=api_key,
            model_id=model
        )
        
        self.voice = voice
        self.stability = stability
        self.similarity_boost = similarity_boost
        
        # Resolve voice ID
        self._voice_id = self._resolve_voice_id(voice)
    
    def _resolve_voice_id(self, voice: str) -> str:
        """Resolve voice name to ID."""
        if voice.lower() in VOICES:
            return VOICES[voice.lower()]["id"]
        # Assume it's a voice ID
        return voice
    
    def set_voice(self, voice: str):
        """Change the voice."""
        self.voice = voice
        self._voice_id = self._resolve_voice_id(voice)
    
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None
    ) -> bytes:
        """
        Synthesize text to audio bytes.
        
        Args:
            text: Text to synthesize
            voice: Override voice for this call
            
        Returns:
            Audio bytes (MP3 format)
        """
        voice_id = self._resolve_voice_id(voice) if voice else self._voice_id
        
        return self.client.synthesize(
            text=text,
            voice_id=voice_id,
            stability=self.stability,
            similarity_boost=self.similarity_boost
        )
    
    def synthesize_base64(
        self,
        text: str,
        voice: Optional[str] = None
    ) -> str:
        """
        Synthesize text and return base64-encoded audio.
        
        Args:
            text: Text to synthesize
            voice: Override voice
            
        Returns:
            Base64-encoded audio string
        """
        audio_bytes = self.synthesize(text, voice)
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    def synthesize_full(
        self,
        text: str,
        voice: Optional[str] = None
    ) -> TTSResult:
        """
        Synthesize text and return full result with metadata.
        
        Args:
            text: Text to synthesize
            voice: Override voice
            
        Returns:
            TTSResult with audio bytes, base64, and metadata
        """
        voice_name = voice or self.voice
        audio_bytes = self.synthesize(text, voice)
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Rough duration estimate (average speaking rate ~150 words/min)
        word_count = len(text.split())
        duration_estimate = word_count / 2.5  # ~2.5 words per second
        
        return TTSResult(
            audio_bytes=audio_bytes,
            audio_base64=audio_b64,
            format="mp3",
            duration_estimate=duration_estimate,
            text=text,
            voice=voice_name
        )
    
    def synthesize_to_file(
        self,
        text: str,
        output_path: Union[str, Path],
        voice: Optional[str] = None
    ) -> Path:
        """
        Synthesize text and save to file.
        
        Args:
            text: Text to synthesize
            output_path: Output file path
            voice: Override voice
            
        Returns:
            Path to saved file
        """
        audio_bytes = self.synthesize(text, voice)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        
        return output_path
    
    def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None
    ):
        """
        Stream synthesized audio (generator).
        
        Args:
            text: Text to synthesize
            voice: Override voice
            
        Yields:
            Audio chunks (bytes)
        """
        voice_id = self._resolve_voice_id(voice) if voice else self._voice_id
        
        for chunk in self.client.synthesize_stream(
            text=text,
            voice_id=voice_id,
            stability=self.stability,
            similarity_boost=self.similarity_boost
        ):
            yield chunk
    
    def list_voices(self) -> Dict[str, Dict]:
        """Get available voice presets."""
        return VOICES
    
    def get_available_voices(self) -> List[Dict]:
        """Fetch all available voices from ElevenLabs."""
        return self.client.get_voices()


# ============================================================================
# Convenience Functions
# ============================================================================

def speak(
    text: str,
    voice: str = "spanish_female",
    api_key: Optional[str] = None
) -> str:
    """
    Quick function to convert text to speech (base64).
    
    Args:
        text: Text to synthesize
        voice: Voice name or ID
        api_key: Optional API key
        
    Returns:
        Base64-encoded audio string
        
    Example:
        audio_b64 = speak("¡Hola! Me llamo Edwin")
    """
    tts = TextToSpeech(api_key=api_key, voice=voice)
    return tts.synthesize_base64(text)


def speak_to_file(
    text: str,
    output_path: Union[str, Path],
    voice: str = "spanish_female",
    api_key: Optional[str] = None
) -> Path:
    """
    Quick function to convert text to speech and save to file.
    
    Args:
        text: Text to synthesize
        output_path: Output file path
        voice: Voice name or ID
        api_key: Optional API key
        
    Returns:
        Path to saved file
        
    Example:
        speak_to_file("Hola mundo", "hello.mp3")
    """
    tts = TextToSpeech(api_key=api_key, voice=voice)
    return tts.synthesize_to_file(text, output_path)


if __name__ == "__main__":
    print("=" * 60)
    print("Text-to-Speech Service Test")
    print("=" * 60)
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    
    if api_key:
        print("\n✅ API key found. Testing...\n")
        
        tts = TextToSpeech()
        
        # Test basic synthesis
        print("Test 1: Basic synthesis")
        audio = tts.synthesize("¡Hola! Me llamo Edwin. ¿Cómo estás?")
        print(f"  Audio size: {len(audio)} bytes\n")
        
        # Test base64
        print("Test 2: Base64 output")
        audio_b64 = tts.synthesize_base64("Muchas gracias por tu ayuda")
        print(f"  Base64 length: {len(audio_b64)} chars")
        print(f"  Preview: {audio_b64[:50]}...\n")
        
        # Test full result
        print("Test 3: Full result with metadata")
        result = tts.synthesize_full("Este es un texto de prueba para el servicio de voz")
        print(f"  Text: {result.text}")
        print(f"  Voice: {result.voice}")
        print(f"  Format: {result.format}")
        print(f"  Duration estimate: {result.duration_estimate:.1f}s")
        print(f"  Data URL preview: {result.to_data_url()[:60]}...\n")
        
        # Test save to file
        print("Test 4: Save to file")
        output_path = tts.synthesize_to_file(
            "Este archivo se guardó correctamente",
            "test_output.mp3"
        )
        print(f"  Saved to: {output_path}\n")
        
        # Test different voices
        print("Test 5: Different voices")
        for voice_name in ["spanish_female", "spanish_male", "rachel"]:
            tts.set_voice(voice_name)
            audio = tts.synthesize("Hola")
            print(f"  {voice_name}: {len(audio)} bytes")
        
        print("\n✅ All tests passed!")
        
    else:
        print("\n⚠️  ELEVENLABS_API_KEY not set.")
        print("Set it with: export ELEVENLABS_API_KEY='your-key'")
        print("Get your key at: https://elevenlabs.io/")
        print("\nAvailable voice presets:")
        for name, info in VOICES.items():
            print(f"  • {name}: {info['name']} ({info['gender']}, {info['accent']})")
    
    print("\n" + "=" * 60)
