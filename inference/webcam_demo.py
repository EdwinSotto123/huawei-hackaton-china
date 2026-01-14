"""
Webcam Demo (Squeezeformer)
===========================
Real-time sign language recognition using webcam.

Uses MindSpore Squeezeformer model for inference.
Note: OpenCV and MediaPipe require numpy for I/O.

Usage:
    python webcam_demo.py [--checkpoint model.ckpt]
"""

import cv2
import json
import time
import argparse
from pathlib import Path
from collections import deque
import sys

# MindSpore imports
import mindspore
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import Tensor

# MediaPipe (requires numpy for I/O)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np  # Required for OpenCV/MediaPipe interface only

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Landmark Definitions
# ============================================================================
LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
       291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
       78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
       95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
NOSE = [1, 2, 98, 327]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE
NUM_LANDMARKS = len(POINT_LANDMARKS)

MODELS_DIR = Path(__file__).parent.parent / "models" / "mediapipe"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
FACE_MODEL_PATH = MODELS_DIR / "face_landmarker.task"


def download_model(url: str, path: Path):
    """Download model if not exists."""
    import urllib.request
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {path.name}...")
        urllib.request.urlretrieve(url, str(path))


def load_vocabulary() -> list:
    """Load sign vocabulary."""
    vocab_path = Path(__file__).parent.parent / "dataset" / "sign_vocabulary.json"
    with open(vocab_path) as f:
        data = json.load(f)
    return data["signs"]


# ============================================================================
# MindSpore Processing Functions
# ============================================================================
def ms_nan_mean(x: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    """Compute mean ignoring NaN using mindspore.numpy."""
    mask = ~mnp.isnan(x)
    x_filled = mnp.where(mask, x, mnp.zeros_like(x))
    count = mnp.sum(mask.astype(mnp.float32), axis=axis, keepdims=keepdims)
    total = mnp.sum(x_filled, axis=axis, keepdims=keepdims)
    return total / mnp.maximum(count, mnp.array(1.0))


def normalize_sequence_mindspore(sequence: Tensor) -> Tensor:
    """
    Normalize landmark sequence using MindSpore.
    
    Args:
        sequence: (T, 543, 3) landmark tensor
        
    Returns:
        features: (T, 708) normalized features
    """
    # Select landmarks: (T, 118, 2)
    x = sequence[:, POINT_LANDMARKS, :2]
    T = x.shape[0]
    
    # Center using mean
    mean = ms_nan_mean(x, axis=(0, 1), keepdims=True)
    mean = mnp.where(mnp.isnan(mean), mnp.full_like(mean, 0.5), mean)
    
    # Normalize using std
    diff = x - mean
    std = mnp.sqrt(ms_nan_mean(diff * diff))
    std = mnp.maximum(std, mnp.array(1e-6))
    x = (x - mean) / std
    
    # Calculate velocities
    dx = mnp.zeros_like(x)
    dx2 = mnp.zeros_like(x)
    if T > 1:
        dx_vals = x[1:] - x[:-1]
        dx = mnp.concatenate([dx_vals, mnp.zeros((1, NUM_LANDMARKS, 2))], axis=0)
    if T > 2:
        dx2_vals = x[2:] - x[:-2]
        dx2 = mnp.concatenate([dx2_vals, mnp.zeros((2, NUM_LANDMARKS, 2))], axis=0)
    
    # Flatten and concatenate
    x_flat = mnp.reshape(x, (T, -1))
    dx_flat = mnp.reshape(dx, (T, -1))
    dx2_flat = mnp.reshape(dx2, (T, -1))
    features = mnp.concatenate([x_flat, dx_flat, dx2_flat], axis=-1)
    
    # Replace NaN
    features = mnp.where(mnp.isnan(features), mnp.zeros_like(features), features)
    
    return features


class SignPredictor:
    """
    Real-time sign language predictor.
    
    Uses MediaPipe for landmark extraction and MindSpore for inference.
    """
    
    def __init__(self, checkpoint_path: str = None):
        print("Initializing Sign Predictor...")
        
        # Load vocabulary
        self.vocabulary = load_vocabulary()
        self.num_classes = len(self.vocabulary)
        print(f"  Vocabulary: {self.num_classes} signs")
        
        # Download MediaPipe models
        download_model(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            HAND_MODEL_PATH
        )
        download_model(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            FACE_MODEL_PATH
        )
        
        # Initialize MediaPipe
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        print("  ✓ MediaPipe Hand Landmarker")
        
        face_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )
        self.face_detector = vision.FaceLandmarker.create_from_options(face_options)
        print("  ✓ MediaPipe Face Landmarker")
        
        # Frame buffer
        self.buffer_size = 64
        self.landmark_buffer = deque(maxlen=self.buffer_size)
        
        # Model
        self.model = None
        if checkpoint_path:
            self._load_model(checkpoint_path)
        else:
            print("  ⚠ No checkpoint - using demo mode")
        
        print("Initialization complete!")
    
    def _load_model(self, checkpoint_path: str):
        """Load trained Squeezeformer model from checkpoint."""
        print(f"  Loading checkpoint: {checkpoint_path}")
        try:
            from models import ISLRModelV2
            from mindspore import load_checkpoint, load_param_into_net
            
            self.model = ISLRModelV2(num_classes=self.num_classes)
            param_dict = load_checkpoint(checkpoint_path)
            load_param_into_net(self.model, param_dict)
            print("  ✓ Squeezeformer model loaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")
    
    def extract_landmarks(self, frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """Extract landmarks from frame (requires numpy for MediaPipe)."""
        landmarks = np.full((543, 3), np.nan, dtype=np.float32)
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Face
        try:
            result = self.face_detector.detect_for_video(mp_image, timestamp_ms)
            if result.face_landmarks:
                for lms in result.face_landmarks:
                    for i, lm in enumerate(lms):
                        if i < 468:
                            landmarks[i] = [lm.x, lm.y, lm.z]
        except:
            pass
        
        # Hands
        try:
            result = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
            if result.hand_landmarks:
                for idx, lms in enumerate(result.hand_landmarks):
                    handedness = result.handedness[idx][0].category_name
                    base_idx = 468 if handedness == "Left" else 522
                    for i, lm in enumerate(lms):
                        if i < 21:
                            landmarks[base_idx + i] = [lm.x, lm.y, lm.z]
        except:
            pass
        
        return landmarks
    
    def predict(self) -> tuple:
        """Make prediction from current buffer using MindSpore."""
        if len(self.landmark_buffer) < 16:
            return None, 0.0
        
        # Convert buffer to MindSpore Tensor
        sequence_np = np.stack(list(self.landmark_buffer))
        sequence = Tensor(sequence_np.astype('float32'))
        
        # Normalize using MindSpore
        features = normalize_sequence_mindspore(sequence)
        
        # Predict
        if self.model is not None:
            # Add batch dimension and run inference
            features = mnp.expand_dims(features, axis=0)
            logits = self.model(features)
            probs = ops.Softmax(axis=-1)(logits)
            probs = probs.asnumpy()[0]
        else:
            # Demo mode: random predictions using MindSpore ops
            probs = ops.Softmax(axis=-1)(
                Tensor(ops.StandardNormal()((self.num_classes,)).asnumpy())
            ).asnumpy()
        
        # Get top prediction
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        top_sign = self.vocabulary[top_idx]
        
        return top_sign, top_prob
    
    def draw_ui(self, frame: np.ndarray, sign: str, confidence: float,
                has_hands: bool, fps: float) -> np.ndarray:
        """Draw UI overlay."""
        h, w = frame.shape[:2]
        
        # Top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "MindSpore Sign Language Recognition", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Status
        hand_color = (0, 255, 0) if has_hands else (100, 100, 100)
        cv2.putText(frame, f"Hands: {'Y' if has_hands else 'N'}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)
        cv2.putText(frame, f"FPS: {fps:.0f}", (120, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Buffer: {len(self.landmark_buffer)}/{self.buffer_size}", 
                    (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Bottom bar
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 100), (w, h), (30, 30, 30), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
        
        if sign and confidence > 0.3:
            cv2.putText(frame, f"Prediction: {sign.upper()}", (20, h - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {confidence*100:.0f}%", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "Make a sign...", (20, h - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1)
        
        cv2.putText(frame, "Press Q to quit", (w - 150, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return frame
    
    def run(self, camera_id: int = 0):
        """Run webcam demo."""
        print("\n" + "=" * 60)
        print("WEBCAM DEMO (Squeezeformer Inference)")
        print("=" * 60)
        print("Press Q to quit\n")
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Warmup
        for _ in range(30):
            cap.read()
            time.sleep(0.02)
        
        start_time = time.time()
        frame_count = 0
        fps = 30
        fps_time = time.time()
        
        current_sign = None
        current_conf = 0.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            timestamp_ms = int((time.time() - start_time) * 1000)
            
            # Extract landmarks
            landmarks = self.extract_landmarks(frame, timestamp_ms)
            has_hands = not np.isnan(landmarks[LHAND[0], 0]) or not np.isnan(landmarks[RHAND[0], 0])
            
            # Add to buffer
            self.landmark_buffer.append(landmarks)
            
            # Predict every 10 frames
            frame_count += 1
            if frame_count % 10 == 0:
                sign, conf = self.predict()
                if conf > 0.3:
                    current_sign = sign
                    current_conf = conf
            
            # Calculate FPS
            if frame_count % 30 == 0:
                now = time.time()
                fps = 30 / (now - fps_time)
                fps_time = now
            
            # Draw UI
            frame = self.draw_ui(frame, current_sign, current_conf, has_hands, fps)
            
            cv2.imshow("MindSpore Sign Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nDemo ended.")


def main():
    parser = argparse.ArgumentParser(description="Webcam sign recognition demo")
    parser.add_argument("--checkpoint", "-c", type=str, help="Model checkpoint path")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    
    args = parser.parse_args()
    
    predictor = SignPredictor(checkpoint_path=args.checkpoint)
    predictor.run(camera_id=args.camera)


if __name__ == "__main__":
    main()
