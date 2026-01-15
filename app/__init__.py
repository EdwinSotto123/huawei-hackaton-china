"""
Sign Language App
==================
Complete application for sign language recognition with voice output.

Pipeline:
    📹 Camera → 🖐️ MediaPipe → 🧠 Model → 📝 LLM → 🔊 Audio

Usage:
    python -m app.main
    # or
    from app import SignLanguageApp
    app = SignLanguageApp()
    app.run()
"""

from .main import SignLanguageApp, run_app
from .camera import CameraCapture
from .pipeline import SignToSpeechPipeline

__all__ = [
    "SignLanguageApp",
    "CameraCapture", 
    "SignToSpeechPipeline",
    "run_app"
]
