"""
ElevenLabs API Client
======================
Client for ElevenLabs Text-to-Speech API.

Installation:
    pip install requests

Get your API key at: https://elevenlabs.io/
"""

import os
import base64
import requests
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# Voice Presets
# ============================================================================

class Voice(Enum):
    """Pre-defined voice options."""
    # Multilingual voices (support Spanish)
    RACHEL = "21m00Tcm4TlvDq8ikWAM"      # Female, calm
    DOMI = "AZnzlk1XvdvUeBnXmlld"         # Female, strong
    BELLA = "EXAVITQu4vr4xnSDxMaL"        # Female, soft
    ANTONI = "ErXwobaYiN019PkySvjV"       # Male, well-rounded
    ELLI = "MF3mGyEYCl7XYWbV9V6O"         # Female, young
    JOSH = "TxGEqnHWrfWFTfGW9XjX"         # Male, deep
    ARNOLD = "VR6AewLTigWG4xSOukaG"       # Male, crisp
    ADAM = "pNInz6obpgDQGcFmaJgB"         # Male, deep
    SAM = "yoZ06aMxZJJ28mfd3POQ"          # Male, raspy
    
    # Spanish-optimized
    MATILDA = "XrExE9yKIg1WjnnlVkGX"      # Female, warm (good for Spanish)
    CHARLOTTE = "XB0fDUnXU5powFXDhCwa"    # Female, seductive
    CLYDE = "2EiwWnXFnvU5JabPnv8n"        # Male, war veteran
    DAVE = "CYw3kZ02Hs0563khs1Fj"         # Male, conversational
    FIN = "D38z5RcWu1voky8WS1ja"          # Male, sailor
    GLINDA = "z9fAnlkpzviPz146aGWa"       # Female, witch
    GRACE = "oWAxZDx7w5VEj9dCyTzz"        # Female, southern
    
    # Default for Spanish
    SPANISH_FEMALE = "XrExE9yKIg1WjnnlVkGX"  # Matilda - warm, good for Spanish
    SPANISH_MALE = "ErXwobaYiN019PkySvjV"    # Antoni - well-rounded


# Voice metadata for easy access
VOICES = {
    "rachel": {"id": Voice.RACHEL.value, "name": "Rachel", "gender": "female", "accent": "american"},
    "domi": {"id": Voice.DOMI.value, "name": "Domi", "gender": "female", "accent": "american"},
    "bella": {"id": Voice.BELLA.value, "name": "Bella", "gender": "female", "accent": "american"},
    "antoni": {"id": Voice.ANTONI.value, "name": "Antoni", "gender": "male", "accent": "american"},
    "elli": {"id": Voice.ELLI.value, "name": "Elli", "gender": "female", "accent": "american"},
    "josh": {"id": Voice.JOSH.value, "name": "Josh", "gender": "male", "accent": "american"},
    "arnold": {"id": Voice.ARNOLD.value, "name": "Arnold", "gender": "male", "accent": "american"},
    "adam": {"id": Voice.ADAM.value, "name": "Adam", "gender": "male", "accent": "american"},
    "sam": {"id": Voice.SAM.value, "name": "Sam", "gender": "male", "accent": "american"},
    "matilda": {"id": Voice.MATILDA.value, "name": "Matilda", "gender": "female", "accent": "neutral"},
    "charlotte": {"id": Voice.CHARLOTTE.value, "name": "Charlotte", "gender": "female", "accent": "neutral"},
    "spanish_female": {"id": Voice.SPANISH_FEMALE.value, "name": "Matilda", "gender": "female", "accent": "spanish"},
    "spanish_male": {"id": Voice.SPANISH_MALE.value, "name": "Antoni", "gender": "male", "accent": "spanish"},
}


@dataclass
class ElevenLabsConfig:
    """Configuration for ElevenLabs API."""
    api_key: str
    base_url: str = "https://api.elevenlabs.io/v1"
    model_id: str = "eleven_multilingual_v2"  # Best for Spanish
    output_format: str = "mp3_44100_128"      # High quality MP3


class ElevenLabsClient:
    """
    Client for ElevenLabs Text-to-Speech API.
    
    Usage:
        client = ElevenLabsClient(api_key="your-api-key")
        
        # Get audio bytes
        audio = client.synthesize("Hola mundo")
        
        # Get base64
        audio_b64 = client.synthesize_base64("Hola mundo")
        
        # With specific voice
        audio = client.synthesize("Hello", voice_id="21m00Tcm4TlvDq8ikWAM")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "eleven_multilingual_v2",
        output_format: str = "mp3_44100_128"
    ):
        """
        Initialize ElevenLabs client.
        
        Args:
            api_key: ElevenLabs API key (or set ELEVENLABS_API_KEY env var)
            model_id: Model to use:
                - eleven_multilingual_v2 (best for Spanish/multilingual)
                - eleven_monolingual_v1 (English only, faster)
                - eleven_turbo_v2 (fastest, good quality)
            output_format: Audio format:
                - mp3_44100_128 (high quality MP3)
                - mp3_44100_64 (standard MP3)
                - pcm_16000 (raw PCM)
                - pcm_22050 (raw PCM higher quality)
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ElevenLabs API key required. "
                "Set ELEVENLABS_API_KEY environment variable or pass api_key parameter."
            )
        
        self.config = ElevenLabsConfig(
            api_key=self.api_key,
            model_id=model_id,
            output_format=output_format
        )
        
        self.headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
    
    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        voice_name: Optional[str] = None,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        use_speaker_boost: bool = True
    ) -> bytes:
        """
        Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            voice_id: ElevenLabs voice ID (takes priority)
            voice_name: Voice name from VOICES dict (e.g., "spanish_female")
            stability: Voice stability (0.0-1.0). Lower = more expressive
            similarity_boost: Voice clarity (0.0-1.0). Higher = clearer
            style: Style exaggeration (0.0-1.0). Higher = more stylized
            use_speaker_boost: Boost speaker similarity
            
        Returns:
            Audio bytes (MP3 format by default)
        """
        # Resolve voice ID
        if voice_id is None:
            if voice_name and voice_name.lower() in VOICES:
                voice_id = VOICES[voice_name.lower()]["id"]
            else:
                voice_id = Voice.SPANISH_FEMALE.value  # Default
        
        url = f"{self.config.base_url}/text-to-speech/{voice_id}"
        
        payload = {
            "text": text,
            "model_id": self.config.model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "use_speaker_boost": use_speaker_boost
            }
        }
        
        # Add output format as query param
        params = {"output_format": self.config.output_format}
        
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            return response.content
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise ValueError("Invalid ElevenLabs API key")
            elif response.status_code == 422:
                raise ValueError(f"Invalid request: {response.text}")
            else:
                raise ConnectionError(f"ElevenLabs API error: {e}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"ElevenLabs API error: {e}")
    
    def synthesize_base64(
        self,
        text: str,
        voice_id: Optional[str] = None,
        voice_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Synthesize text and return base64-encoded audio.
        
        Args:
            text: Text to synthesize
            voice_id: Voice ID
            voice_name: Voice name
            **kwargs: Additional parameters for synthesize()
            
        Returns:
            Base64-encoded audio string
        """
        audio_bytes = self.synthesize(text, voice_id, voice_name, **kwargs)
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    def synthesize_stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        voice_name: Optional[str] = None,
        **kwargs
    ):
        """
        Stream synthesized audio (generator).
        
        Args:
            text: Text to synthesize
            voice_id: Voice ID
            voice_name: Voice name
            
        Yields:
            Audio chunks (bytes)
        """
        if voice_id is None:
            if voice_name and voice_name.lower() in VOICES:
                voice_id = VOICES[voice_name.lower()]["id"]
            else:
                voice_id = Voice.SPANISH_FEMALE.value
        
        url = f"{self.config.base_url}/text-to-speech/{voice_id}/stream"
        
        payload = {
            "text": text,
            "model_id": self.config.model_id,
            "voice_settings": {
                "stability": kwargs.get("stability", 0.5),
                "similarity_boost": kwargs.get("similarity_boost", 0.75)
            }
        }
        
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk
                    
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"ElevenLabs streaming error: {e}")
    
    def get_voices(self) -> List[Dict]:
        """
        Get list of available voices.
        
        Returns:
            List of voice dictionaries with id, name, labels, etc.
        """
        url = f"{self.config.base_url}/voices"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json().get("voices", [])
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error fetching voices: {e}")
    
    def get_user_info(self) -> Dict:
        """
        Get user subscription info (character quota, etc.).
        
        Returns:
            User info dictionary
        """
        url = f"{self.config.base_url}/user"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error fetching user info: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("ElevenLabs Client Test")
    print("=" * 50)
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    
    if api_key:
        print("\n✅ API key found. Testing...\n")
        
        client = ElevenLabsClient()
        
        # Test synthesis
        print("Test 1: Synthesize Spanish text")
        audio = client.synthesize(
            "¡Hola! Me llamo Edwin. ¿Cómo estás?",
            voice_name="spanish_female"
        )
        print(f"  Audio size: {len(audio)} bytes")
        
        # Save to file
        with open("test_audio.mp3", "wb") as f:
            f.write(audio)
        print("  Saved to: test_audio.mp3\n")
        
        # Test base64
        print("Test 2: Base64 output")
        audio_b64 = client.synthesize_base64("Gracias por tu ayuda")
        print(f"  Base64 length: {len(audio_b64)} chars")
        print(f"  Preview: {audio_b64[:50]}...\n")
        
        # List voices
        print("Test 3: Available voices")
        voices = client.get_voices()
        print(f"  Found {len(voices)} voices")
        for v in voices[:5]:
            print(f"    • {v['name']} ({v['voice_id'][:8]}...)")
        
        print("\n✅ All tests passed!")
    else:
        print("\n⚠️  ELEVENLABS_API_KEY not set.")
        print("Set it with: export ELEVENLABS_API_KEY='your-key'")
        print("Get your key at: https://elevenlabs.io/")
        print("\nAvailable voice presets:")
        for name, info in VOICES.items():
            print(f"  • {name}: {info['name']} ({info['gender']}, {info['accent']})")
    
    print("\n" + "=" * 50)
