"""
Sign to Speech Pipeline
========================
Complete pipeline from landmarks to spoken audio.

Pipeline:
    Features (708) → Model → Prediction → LLM → Text → TTS → Audio
"""

import os
import sys
import time
import threading
import queue
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mindspore
    from mindspore import Tensor
    import mindspore.ops as ops
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False
    print("⚠️  MindSpore not available. Using mock model.")

# Import our services
try:
    from LLM_PARSER import SignToTextParser, OutputStyle
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️  LLM Parser not available.")

try:
    from TTS_SERVICE import TextToSpeech
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️  TTS Service not available.")


@dataclass
class PipelineResult:
    """Result from the pipeline."""
    raw_prediction: str          # Raw model output (e.g., "YO NOMBRE EDWIN")
    natural_text: str            # LLM processed text (e.g., "Me llamo Edwin")
    audio_data: Optional[bytes]  # Audio bytes (MP3)
    audio_base64: Optional[str]  # Base64 encoded audio
    confidence: float            # Prediction confidence
    processing_time: float       # Total processing time in seconds
    timestamp: float             # When processed


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    # Model settings
    num_classes: int = 250
    sequence_length: int = 64    # Frames needed for prediction
    confidence_threshold: float = 0.3
    
    # LLM settings
    language: str = "es"
    style: str = "casual"
    
    # TTS settings
    voice: str = "spanish_female"
    
    # Behavior
    min_prediction_interval: float = 2.0  # Min seconds between predictions
    enable_tts: bool = True
    enable_llm: bool = True


# Vocabulary - 250 common signs (simplified example)
# In production, load from sign_vocabulary.json
DEFAULT_VOCABULARY = [
    "HOLA", "ADIOS", "GRACIAS", "POR_FAVOR", "SI", "NO", "YO", "TU", "EL", "ELLA",
    "NOSOTROS", "USTEDES", "ELLOS", "NOMBRE", "LLAMAR", "SER", "ESTAR", "TENER",
    "QUERER", "PODER", "DEBER", "IR", "VENIR", "CASA", "TRABAJO", "ESCUELA",
    "FAMILIA", "AMIGO", "AMOR", "FELIZ", "TRISTE", "BIEN", "MAL", "BUENO", "MALO",
    "GRANDE", "PEQUEÑO", "MUCHO", "POCO", "TODO", "NADA", "AQUI", "ALLA", "DONDE",
    "CUANDO", "COMO", "QUE", "QUIEN", "PORQUE", "AGUA", "COMIDA", "COMER", "BEBER",
    "DORMIR", "DESPERTAR", "MAÑANA", "TARDE", "NOCHE", "HOY", "AYER", "MAÑANA_TIEMPO",
    "SEMANA", "MES", "AÑO", "TIEMPO", "AHORA", "DESPUES", "ANTES", "SIEMPRE", "NUNCA",
    "NUMERO", "UNO", "DOS", "TRES", "CUATRO", "CINCO", "SEIS", "SIETE", "OCHO", 
    "NUEVE", "DIEZ", "MADRE", "PADRE", "HIJO", "HIJA", "HERMANO", "HERMANA",
    "ABUELO", "ABUELA", "TIO", "TIA", "PRIMO", "ESPOSO", "ESPOSA", "BEBE",
    "NIÑO", "ADULTO", "PERSONA", "GENTE", "HOMBRE", "MUJER", "DOCTOR", "MAESTRO",
    "ESTUDIANTE", "AYUDA", "NECESITAR", "ENTENDER", "SABER", "PENSAR", "SENTIR",
    "VER", "OIR", "HABLAR", "DECIR", "LEER", "ESCRIBIR", "APRENDER", "ENSEÑAR",
    "TRABAJAR", "JUGAR", "CAMINAR", "CORRER", "SALTAR", "BAILAR", "CANTAR",
    "COCINAR", "LIMPIAR", "COMPRAR", "VENDER", "PAGAR", "DINERO", "PRECIO",
    "CARO", "BARATO", "NUEVO", "VIEJO", "BONITO", "FEO", "FACIL", "DIFICIL",
    "RAPIDO", "LENTO", "CALIENTE", "FRIO", "HOSPITAL", "BANCO", "TIENDA",
    "RESTAURANTE", "CINE", "PARQUE", "CALLE", "CIUDAD", "PAIS", "MUNDO",
    "ROJO", "AZUL", "VERDE", "AMARILLO", "BLANCO", "NEGRO", "ROSA", "NARANJA",
    "CARRO", "BUS", "AVION", "TREN", "BICICLETA", "TELEFONO", "COMPUTADORA",
    "INTERNET", "FOTO", "VIDEO", "MUSICA", "PELICULA", "LIBRO", "PAPEL",
    "LAPIZ", "MESA", "SILLA", "PUERTA", "VENTANA", "BAÑO", "COCINA", "CUARTO",
    "CAMA", "ROPA", "ZAPATO", "SOMBRERO", "LENTES", "RELOJ", "PERRO", "GATO",
    "PAJARO", "PEZ", "ARBOL", "FLOR", "SOL", "LUNA", "ESTRELLA", "LLUVIA",
    "NIEVE", "VIENTO", "ENFERMO", "SANO", "MEDICINA", "DOLOR", "CANSADO",
    "ENERGIA", "PROBLEMA", "SOLUCION", "PREGUNTA", "RESPUESTA", "EJEMPLO",
    "HISTORIA", "NOTICIA", "IMPORTANTE", "DIFERENTE", "IGUAL", "ESPECIAL",
    "NORMAL", "POSIBLE", "IMPOSIBLE", "SEGURO", "PELIGROSO", "VERDAD", "MENTIRA",
    "SECRETO", "SORPRESA", "CELEBRAR", "FELICITAR", "CUMPLEAÑOS", "NAVIDAD",
    "FIESTA", "REGALO", "DISCULPA", "PERDON", "PERMISO", "FAVOR", "BIENVENIDO",
    "SUERTE", "EXITO", "ESPERANZA", "FE", "PAZ", "LIBERTAD"
]


class MockModel:
    """Mock model for testing without MindSpore."""
    
    def __init__(self, num_classes: int = 250):
        self.num_classes = num_classes
    
    def __call__(self, x):
        """Return random logits."""
        batch_size = x.shape[0] if len(x.shape) > 2 else 1
        logits = np.random.randn(batch_size, self.num_classes).astype(np.float32)
        return logits


class SignToSpeechPipeline:
    """
    Complete pipeline: Features → Model → LLM → TTS
    
    Usage:
        pipeline = SignToSpeechPipeline()
        
        # Process frame features
        result = pipeline.process(features)
        print(result.natural_text)  # "Me llamo Edwin"
        play_audio(result.audio_data)  # Play MP3
        
        # Or process multiple frames (better accuracy)
        pipeline.add_frame(features1)
        pipeline.add_frame(features2)
        ...
        result = pipeline.finalize()
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        vocabulary: Optional[List[str]] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration
            vocabulary: Sign vocabulary list
            model_path: Path to trained model checkpoint
        """
        self.config = config or PipelineConfig()
        self.vocabulary = vocabulary or DEFAULT_VOCABULARY
        
        # Frame buffer for sequence
        self.frame_buffer: List[np.ndarray] = []
        self.last_prediction_time = 0
        
        # Initialize components
        self._init_model(model_path)
        self._init_llm()
        self._init_tts()
        
        # Results queue for async processing
        self.results_queue = queue.Queue()
    
    def _init_model(self, model_path: Optional[str]):
        """Initialize the sign recognition model."""
        if MINDSPORE_AVAILABLE and model_path and Path(model_path).exists():
            try:
                from models import ISLRModelV2
                self.model = ISLRModelV2(num_classes=self.config.num_classes)
                # Load checkpoint
                param_dict = mindspore.load_checkpoint(model_path)
                mindspore.load_param_into_net(self.model, param_dict)
                self.model.set_train(False)
                print(f"✅ Model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️  Error loading model: {e}. Using mock.")
                self.model = MockModel(self.config.num_classes)
        else:
            self.model = MockModel(self.config.num_classes)
            print("ℹ️  Using mock model (no checkpoint)")
    
    def _init_llm(self):
        """Initialize LLM parser."""
        if LLM_AVAILABLE and self.config.enable_llm:
            try:
                self.llm = SignToTextParser(
                    language=self.config.language,
                    style=OutputStyle(self.config.style)
                )
                print("✅ LLM Parser initialized")
            except Exception as e:
                print(f"⚠️  LLM error: {e}")
                self.llm = None
        else:
            self.llm = None
    
    def _init_tts(self):
        """Initialize TTS service."""
        if TTS_AVAILABLE and self.config.enable_tts:
            try:
                self.tts = TextToSpeech(voice=self.config.voice)
                print("✅ TTS Service initialized")
            except Exception as e:
                print(f"⚠️  TTS error: {e}")
                self.tts = None
        else:
            self.tts = None
    
    def add_frame(self, features: np.ndarray):
        """
        Add frame features to buffer.
        
        Args:
            features: (708,) feature vector
        """
        self.frame_buffer.append(features)
        
        # Keep only last N frames
        max_frames = self.config.sequence_length * 2
        if len(self.frame_buffer) > max_frames:
            self.frame_buffer = self.frame_buffer[-max_frames:]
    
    def can_predict(self) -> bool:
        """Check if we have enough frames and time has passed."""
        has_frames = len(self.frame_buffer) >= self.config.sequence_length
        time_passed = time.time() - self.last_prediction_time >= self.config.min_prediction_interval
        return has_frames and time_passed
    
    def _run_model(self, features: np.ndarray) -> Tuple[List[str], float]:
        """
        Run model inference.
        
        Args:
            features: (seq_len, 708) features
            
        Returns:
            predictions: List of predicted words
            confidence: Average confidence
        """
        # Prepare input: (1, T, 708)
        if len(features.shape) == 2:
            features = features[np.newaxis, ...]
        
        # Run model
        if MINDSPORE_AVAILABLE and not isinstance(self.model, MockModel):
            x = Tensor(features.astype(np.float32))
            logits = self.model(x)
            
            # Post-processing with MindSpore Ops
            # This keeps tensors on device longer
            probs = ops.Softmax(axis=-1)(logits[0])
            top_vals, top_inds = ops.TopK(sorted=True)(probs, 5)
            
            top_indices = top_inds.asnumpy()
            top_probs = top_vals.asnumpy()
        else:
            # Fallback for Mock/NumPy
            logits = self.model(features)
            probs = self._softmax(logits[0])
            top_indices = np.argsort(probs)[-5:][::-1]
            top_probs = probs[top_indices]
        
        predictions = []
        confidences = []
        
        for i, idx in enumerate(top_indices):
            prob = float(top_probs[i])
            if prob > self.config.confidence_threshold:
                if idx < len(self.vocabulary):
                    predictions.append(self.vocabulary[idx])
                    confidences.append(prob)
        
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        
        return predictions, avg_confidence
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _run_llm(self, raw_prediction: str) -> str:
        """Convert raw prediction to natural text."""
        if self.llm:
            try:
                return self.llm.parse(raw_prediction)
            except Exception as e:
                print(f"LLM error: {e}")
        
        # Fallback: simple title case
        return raw_prediction.replace("_", " ").title()
    
    def _run_tts(self, text: str) -> Tuple[Optional[bytes], Optional[str]]:
        """Convert text to speech."""
        if self.tts:
            try:
                audio_bytes = self.tts.synthesize(text)
                audio_b64 = self.tts.synthesize_base64(text)
                return audio_bytes, audio_b64
            except Exception as e:
                print(f"TTS error: {e}")
        
        return None, None
    
    def process(self, features: np.ndarray) -> Optional[PipelineResult]:
        """
        Process features through complete pipeline.
        
        Args:
            features: (T, 708) or (708,) features
            
        Returns:
            PipelineResult or None if processing failed
        """
        start_time = time.time()
        
        # Ensure 2D
        if len(features.shape) == 1:
            features = features[np.newaxis, :]
        
        # Run model
        predictions, confidence = self._run_model(features)
        
        if not predictions:
            return None
        
        raw_prediction = " ".join(predictions)
        
        # Run LLM
        natural_text = self._run_llm(raw_prediction)
        
        # Run TTS
        audio_data, audio_b64 = self._run_tts(natural_text)
        
        processing_time = time.time() - start_time
        
        return PipelineResult(
            raw_prediction=raw_prediction,
            natural_text=natural_text,
            audio_data=audio_data,
            audio_base64=audio_b64,
            confidence=confidence,
            processing_time=processing_time,
            timestamp=time.time()
        )
    
    def finalize(self) -> Optional[PipelineResult]:
        """
        Process buffered frames and get prediction.
        
        Returns:
            PipelineResult or None
        """
        if not self.can_predict():
            return None
        
        # Get sequence from buffer
        seq_len = self.config.sequence_length
        features = np.array(self.frame_buffer[-seq_len:])  # (T, 708)
        
        # Process
        result = self.process(features)
        
        if result:
            self.last_prediction_time = time.time()
            self.frame_buffer.clear()  # Clear buffer after prediction
        
        return result
    
    def process_async(
        self,
        features: np.ndarray,
        callback: Optional[Callable[[PipelineResult], None]] = None
    ):
        """
        Process asynchronously.
        
        Args:
            features: Input features
            callback: Function to call with result
        """
        def worker():
            result = self.process(features)
            if result:
                self.results_queue.put(result)
                if callback:
                    callback(result)
        
        thread = threading.Thread(target=worker)
        thread.start()
    
    def get_result(self, timeout: float = 0.1) -> Optional[PipelineResult]:
        """Get result from async queue."""
        try:
            return self.results_queue.get(timeout=timeout)
        except queue.Empty:
            return None


if __name__ == "__main__":
    print("=" * 60)
    print("Sign to Speech Pipeline Test")
    print("=" * 60)
    
    # Create pipeline
    config = PipelineConfig(
        enable_llm=os.getenv("DEEPSEEK_API_KEY") is not None,
        enable_tts=os.getenv("ELEVENLABS_API_KEY") is not None
    )
    
    pipeline = SignToSpeechPipeline(config=config)
    
    # Test with random features
    print("\n📊 Testing pipeline with random data...\n")
    
    # Simulate 64 frames
    for i in range(64):
        features = np.random.randn(708).astype(np.float32)
        pipeline.add_frame(features)
    
    print(f"Buffer size: {len(pipeline.frame_buffer)}")
    print(f"Can predict: {pipeline.can_predict()}")
    
    # Get prediction
    result = pipeline.finalize()
    
    if result:
        print(f"\n✅ Pipeline Result:")
        print(f"   Raw: {result.raw_prediction}")
        print(f"   Natural: {result.natural_text}")
        print(f"   Confidence: {result.confidence:.2%}")
        print(f"   Audio: {'Yes' if result.audio_data else 'No'}")
        print(f"   Time: {result.processing_time:.3f}s")
    else:
        print("❌ No result (check API keys)")
    
    print("\n" + "=" * 60)
