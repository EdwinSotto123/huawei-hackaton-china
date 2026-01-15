"""
Sign Language App - Main Application
=====================================
Complete sign language recognition app with real-time camera,
model inference, LLM text enhancement, and voice output.

Usage:
    python -m app.main
    
Controls:
    SPACE - Force prediction now
    R     - Reset/clear buffer
    M     - Toggle mute audio
    S     - Change style (casual/formal/expressive)
    Q     - Quit
"""

import os
import sys
import time
import threading
import wave
import tempfile
from pathlib import Path
from typing import Optional
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

# Try to import audio playback
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  pygame not available. Audio playback disabled.")
    print("   Install with: pip install pygame")

from .camera import CameraCapture, FrameData
from .pipeline import SignToSpeechPipeline, PipelineConfig, PipelineResult


class SignLanguageApp:
    """
    Complete Sign Language Recognition Application.
    
    Pipeline:
        📹 Camera → 🖐️ MediaPipe → 🧠 Model → 📝 LLM → 🔊 Audio
    
    Usage:
        app = SignLanguageApp()
        app.run()
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        model_path: Optional[str] = None,
        language: str = "es",
        voice: str = "spanish_female"
    ):
        """
        Initialize application.
        
        Args:
            camera_id: Camera device ID
            model_path: Path to model checkpoint
            language: Output language (es/en)
            voice: TTS voice name
        """
        print("\n" + "=" * 60)
        print("🤟 SIGN LANGUAGE RECOGNITION APP")
        print("=" * 60 + "\n")
        
        # Configuration
        self.config = PipelineConfig(
            language=language,
            voice=voice,
            min_prediction_interval=3.0,
            enable_llm=os.getenv("DEEPSEEK_API_KEY") is not None,
            enable_tts=os.getenv("ELEVENLABS_API_KEY") is not None
        )
        
        # Initialize components
        print("📷 Initializing camera...")
        self.camera = CameraCapture(camera_id=camera_id)
        
        print("🧠 Initializing pipeline...")
        self.pipeline = SignToSpeechPipeline(
            config=self.config,
            model_path=model_path
        )
        
        # State
        self.running = False
        self.muted = False
        self.last_result: Optional[PipelineResult] = None
        self.predictions_history: list = []
        
        # UI settings
        self.window_name = "Sign Language Recognition"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Audio player
        self.audio_thread: Optional[threading.Thread] = None
        
        print("\n✅ App initialized!")
        print(f"   LLM: {'Enabled' if self.config.enable_llm else 'Disabled (set DEEPSEEK_API_KEY)'}")
        print(f"   TTS: {'Enabled' if self.config.enable_tts else 'Disabled (set ELEVENLABS_API_KEY)'}")
        print()
    
    def play_audio(self, audio_data: bytes):
        """Play audio in background thread."""
        if not PYGAME_AVAILABLE or self.muted or not audio_data:
            return
        
        def player():
            try:
                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(audio_data)
                    temp_path = f.name
                
                # Play with pygame
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                # Cleanup
                os.unlink(temp_path)
                
            except Exception as e:
                print(f"Audio error: {e}")
        
        self.audio_thread = threading.Thread(target=player)
        self.audio_thread.start()
    
    def process_frame(self, frame_data: FrameData):
        """Process a single frame."""
        # Add to buffer
        self.pipeline.add_frame(frame_data.features)
        
        # Check if we should predict
        if frame_data.has_hands and self.pipeline.can_predict():
            self._do_prediction()
    
    def _do_prediction(self):
        """Run prediction on current buffer."""
        result = self.pipeline.finalize()
        
        if result:
            self.last_result = result
            self.predictions_history.append(result)
            
            # Keep last 5 predictions
            if len(self.predictions_history) > 5:
                self.predictions_history.pop(0)
            
            print(f"\n🎯 Prediction:")
            print(f"   Raw: {result.raw_prediction}")
            print(f"   Text: {result.natural_text}")
            print(f"   Confidence: {result.confidence:.1%}")
            
            # Play audio
            if result.audio_data:
                self.play_audio(result.audio_data)
    
    def draw_ui(self, frame: np.ndarray, frame_data: FrameData) -> np.ndarray:
        """Draw user interface overlay."""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Semi-transparent overlay for text areas
        overlay = display.copy()
        
        # Top bar
        cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
        
        # Bottom bar
        cv2.rectangle(overlay, (0, h-120), (w, h), (20, 20, 20), -1)
        
        # Blend
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        
        # Title
        cv2.putText(display, "Sign Language Recognition", (10, 30),
                    self.font, 0.8, (255, 255, 255), 2)
        
        # Status indicators
        status_color = (0, 255, 0) if frame_data.has_hands else (100, 100, 100)
        status_text = "HANDS DETECTED" if frame_data.has_hands else "Show your hands"
        cv2.putText(display, status_text, (10, 60),
                    self.font, 0.5, status_color, 1)
        
        # Buffer progress
        buffer_pct = len(self.pipeline.frame_buffer) / self.config.sequence_length
        buffer_pct = min(1.0, buffer_pct)
        bar_w = 200
        cv2.rectangle(display, (w-220, 20), (w-20, 40), (50, 50, 50), -1)
        cv2.rectangle(display, (w-220, 20), (w-220+int(bar_w*buffer_pct), 40), (0, 200, 0), -1)
        cv2.putText(display, f"Buffer: {int(buffer_pct*100)}%", (w-220, 60),
                    self.font, 0.4, (200, 200, 200), 1)
        
        # Mute indicator
        if self.muted:
            cv2.putText(display, "🔇 MUTED", (w-100, 80),
                        self.font, 0.5, (0, 0, 255), 1)
        
        # Last prediction
        if self.last_result:
            # Natural text (large)
            text = self.last_result.natural_text
            if len(text) > 50:
                text = text[:47] + "..."
            cv2.putText(display, text, (20, h-80),
                        self.font, 0.9, (255, 255, 255), 2)
            
            # Raw prediction (small)
            cv2.putText(display, f"Raw: {self.last_result.raw_prediction}", (20, h-50),
                        self.font, 0.5, (150, 150, 150), 1)
            
            # Confidence
            conf_color = (0, 255, 0) if self.last_result.confidence > 0.5 else (0, 200, 255)
            cv2.putText(display, f"Conf: {self.last_result.confidence:.0%}", (20, h-20),
                        self.font, 0.5, conf_color, 1)
        else:
            cv2.putText(display, "Waiting for signs...", (20, h-80),
                        self.font, 0.7, (150, 150, 150), 1)
        
        # Controls hint
        cv2.putText(display, "SPACE=predict | R=reset | M=mute | Q=quit", (w-350, h-20),
                    self.font, 0.4, (100, 100, 100), 1)
        
        return display
    
    def handle_key(self, key: int) -> bool:
        """
        Handle keyboard input.
        
        Returns:
            False if should quit, True otherwise
        """
        if key == ord('q') or key == 27:  # Q or ESC
            return False
        
        elif key == ord(' '):  # Space - force prediction
            print("⚡ Forcing prediction...")
            self.pipeline.last_prediction_time = 0  # Reset timer
            self._do_prediction()
        
        elif key == ord('r'):  # R - reset
            print("🔄 Resetting buffer...")
            self.pipeline.frame_buffer.clear()
            self.last_result = None
        
        elif key == ord('m'):  # M - toggle mute
            self.muted = not self.muted
            print(f"🔊 Audio: {'Muted' if self.muted else 'Enabled'}")
        
        elif key == ord('s'):  # S - change style
            styles = ["casual", "formal", "expressive", "minimal"]
            current = self.config.style
            idx = (styles.index(current) + 1) % len(styles)
            self.config.style = styles[idx]
            print(f"✨ Style: {self.config.style}")
            
            # Reinit LLM with new style
            if self.pipeline.llm:
                from LLM_PARSER import OutputStyle
                self.pipeline.llm.set_style(OutputStyle(self.config.style))
        
        return True
    
    def run(self):
        """Run the main application loop."""
        print("\n" + "-" * 40)
        print("CONTROLS:")
        print("  SPACE - Force prediction now")
        print("  R     - Reset buffer")
        print("  M     - Toggle mute")
        print("  S     - Change style")
        print("  Q     - Quit")
        print("-" * 40 + "\n")
        
        # Start camera
        if not self.camera.start():
            print("❌ Failed to start camera")
            return
        
        self.running = True
        
        try:
            while self.running:
                # Read frame
                frame_data = self.camera.read_frame()
                
                if frame_data is None:
                    continue
                
                # Draw landmarks on frame
                display = self.camera.draw_landmarks(frame_data.frame, frame_data.landmarks)
                
                # Draw UI
                display = self.draw_ui(display, frame_data)
                
                # Show
                cv2.imshow(self.window_name, display)
                
                # Process (add to buffer, maybe predict)
                self.process_frame(frame_data)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key(key):
                    break
                
        except KeyboardInterrupt:
            print("\n⚡ Interrupted by user")
        
        finally:
            self.running = False
            self.camera.stop()
            cv2.destroyAllWindows()
            
            if PYGAME_AVAILABLE:
                pygame.mixer.quit()
            
            print("\n👋 Application closed")
            print(f"   Total predictions: {len(self.predictions_history)}")


def run_app():
    """Entry point for running the app."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sign Language Recognition App")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera ID")
    parser.add_argument("--model", "-m", type=str, help="Model checkpoint path")
    parser.add_argument("--language", "-l", type=str, default="es", choices=["es", "en"])
    parser.add_argument("--voice", "-v", type=str, default="spanish_female")
    
    args = parser.parse_args()
    
    app = SignLanguageApp(
        camera_id=args.camera,
        model_path=args.model,
        language=args.language,
        voice=args.voice
    )
    app.run()


if __name__ == "__main__":
    run_app()
