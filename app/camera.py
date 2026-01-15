"""
Camera Capture Module
=====================
Handles webcam capture and MediaPipe landmark extraction.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from typing import Optional, Tuple, List, Generator
from dataclasses import dataclass
import time


# Landmark indices for sign recognition (118 total)
LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
       291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
       78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
       95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
NOSE = [1, 2, 98, 327]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133,
        246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362,
        466, 388, 387, 386, 385, 384, 398]

SELECTED_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE  # 118 landmarks


@dataclass
class FrameData:
    """Data extracted from a single frame."""
    frame: np.ndarray           # Original BGR frame
    landmarks: np.ndarray       # (543, 3) raw landmarks
    selected: np.ndarray        # (118, 3) selected landmarks
    features: np.ndarray        # (708,) model input features
    timestamp: float            # Frame timestamp
    has_hands: bool             # Whether hands were detected


class CameraCapture:
    """
    Webcam capture with MediaPipe landmark extraction.
    
    Usage:
        camera = CameraCapture()
        camera.start()
        
        for frame_data in camera.stream():
            # frame_data.features contains (708,) model input
            process(frame_data)
            
        camera.stop()
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        buffer_size: int = 30  # Frames to buffer for temporal features
    ):
        """
        Initialize camera capture.
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
            width: Frame width
            height: Frame height
            fps: Target FPS
            buffer_size: Number of frames to buffer for velocity/acceleration
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        
        self.cap = None
        self.running = False
        
        # Landmark buffer for temporal features
        self.landmark_buffer: List[np.ndarray] = []
        
        # MediaPipe setup
        self._setup_mediapipe()
    
    def _setup_mediapipe(self):
        """Initialize MediaPipe solutions."""
        # Holistic solution for full body tracking
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
    
    def start(self) -> bool:
        """Start camera capture."""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"❌ Error: Cannot open camera {self.camera_id}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self.running = True
        self.landmark_buffer.clear()
        
        print(f"✅ Camera started: {self.width}x{self.height} @ {self.fps}fps")
        return True
    
    def stop(self):
        """Stop camera capture."""
        self.running = False
        if self.cap:
            self.cap.release()
        self.holistic.close()
        print("📷 Camera stopped")
    
    def _extract_landmarks(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Extract landmarks from frame using MediaPipe.
        
        Returns:
            landmarks: (543, 3) array
            has_hands: bool indicating if hands were detected
        """
        # Initialize with NaN
        landmarks = np.full((543, 3), np.nan, dtype=np.float32)
        has_hands = False
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.holistic.process(frame_rgb)
        
        # Face landmarks (0-467)
        if results.face_landmarks:
            for i, lm in enumerate(results.face_landmarks.landmark):
                if i < 468:
                    landmarks[i] = [lm.x, lm.y, lm.z]
        
        # Left hand (468-488)
        if results.left_hand_landmarks:
            has_hands = True
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                if i < 21:
                    landmarks[468 + i] = [lm.x, lm.y, lm.z]
        
        # Pose (489-521)
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                if i < 33:
                    landmarks[489 + i] = [lm.x, lm.y, lm.z]
        
        # Right hand (522-542)
        if results.right_hand_landmarks:
            has_hands = True
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                if i < 21:
                    landmarks[522 + i] = [lm.x, lm.y, lm.z]
        
        return landmarks, has_hands
    
    def _compute_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute model input features from landmarks.
        
        Features: position (x,y), velocity (dx,dy), acceleration (d2x,d2y)
        Output: (118 * 6) = 708 features
        """
        # Select landmarks
        selected = landmarks[SELECTED_LANDMARKS]  # (118, 3)
        
        # Add to buffer
        self.landmark_buffer.append(selected[:, :2])  # Only x, y
        if len(self.landmark_buffer) > self.buffer_size:
            self.landmark_buffer.pop(0)
        
        # Current position (x, y)
        pos = selected[:, :2].flatten()  # (236,)
        
        # Velocity (dx, dy)
        if len(self.landmark_buffer) >= 2:
            vel = (self.landmark_buffer[-1] - self.landmark_buffer[-2]).flatten()
        else:
            vel = np.zeros(236, dtype=np.float32)
        
        # Acceleration (d2x, d2y)
        if len(self.landmark_buffer) >= 3:
            vel_prev = self.landmark_buffer[-2] - self.landmark_buffer[-3]
            vel_curr = self.landmark_buffer[-1] - self.landmark_buffer[-2]
            acc = (vel_curr - vel_prev).flatten()
        else:
            acc = np.zeros(236, dtype=np.float32)
        
        # Concatenate: (236 + 236 + 236) = 708
        features = np.concatenate([pos, vel, acc]).astype(np.float32)
        
        # Replace NaN with 0
        features = np.nan_to_num(features, nan=0.0)
        
        return features
    
    def read_frame(self) -> Optional[FrameData]:
        """Read and process a single frame."""
        if not self.cap or not self.running:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Extract landmarks
        landmarks, has_hands = self._extract_landmarks(frame)
        
        # Get selected landmarks
        selected = landmarks[SELECTED_LANDMARKS]
        
        # Compute features
        features = self._compute_features(landmarks)
        
        return FrameData(
            frame=frame,
            landmarks=landmarks,
            selected=selected,
            features=features,
            timestamp=time.time(),
            has_hands=has_hands
        )
    
    def stream(self) -> Generator[FrameData, None, None]:
        """
        Stream frames continuously.
        
        Yields:
            FrameData for each frame
        """
        if not self.start():
            return
        
        try:
            while self.running:
                frame_data = self.read_frame()
                if frame_data:
                    yield frame_data
        finally:
            self.stop()
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Draw landmarks on frame for visualization."""
        frame_copy = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw selected landmarks
        for idx in SELECTED_LANDMARKS:
            if not np.isnan(landmarks[idx, 0]):
                x = int(landmarks[idx, 0] * w)
                y = int(landmarks[idx, 1] * h)
                
                # Color by region
                if idx in LHAND or idx in RHAND:
                    color = (0, 255, 0)  # Green for hands
                elif idx in LIP:
                    color = (255, 0, 0)  # Blue for lips
                else:
                    color = (0, 255, 255)  # Yellow for eyes/nose
                
                cv2.circle(frame_copy, (x, y), 3, color, -1)
        
        return frame_copy


if __name__ == "__main__":
    print("Camera Capture Test")
    print("=" * 40)
    
    camera = CameraCapture()
    
    print("\nPress 'q' to quit\n")
    
    frame_count = 0
    for frame_data in camera.stream():
        # Draw landmarks
        display = camera.draw_landmarks(frame_data.frame, frame_data.landmarks)
        
        # Add info text
        status = "✋ Hands detected" if frame_data.has_hands else "👋 Show hands"
        cv2.putText(display, status, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Features: {frame_data.features.shape}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Sign Language Camera", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames")
