"""
Landmark Extraction Module
==========================
Extracts body, hand, and face landmarks from video frames using MediaPipe.

Usage:
    python extract_landmarks.py --input video.mp4 --output landmarks.npy
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import urllib.request
from tqdm import tqdm
import argparse


# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR = Path(__file__).parent.parent
MODELS_DIR = SCRIPT_DIR / "models" / "mediapipe"

# MediaPipe model URLs
MODELS = {
    "hand": {
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        "path": MODELS_DIR / "hand_landmarker.task"
    },
    "face": {
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "path": MODELS_DIR / "face_landmarker.task"
    },
    "pose": {
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "path": MODELS_DIR / "pose_landmarker.task"
    }
}

# Landmark indices (matching original ISLR competition format)
# Total: 543 landmarks
# Face: 0-467 (468 landmarks)
# Left Hand: 468-488 (21 landmarks)
# Pose: 489-521 (33 landmarks)
# Right Hand: 522-542 (21 landmarks)

LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
       291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
       78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
       95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

NOSE = [1, 2, 98, 327]

REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133,
        246, 161, 160, 159, 158, 157, 173]

LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362,
        466, 388, 387, 386, 385, 384, 398]

LHAND = list(range(468, 489))  # 21 landmarks
RHAND = list(range(522, 543))  # 21 landmarks

# Selected landmarks for model (118 total)
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE


def download_model(name: str) -> Path:
    """Download MediaPipe model if not exists."""
    model = MODELS[name]
    if not model["path"].exists():
        model["path"].parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name} model...")
        urllib.request.urlretrieve(model["url"], str(model["path"]))
        print(f"Downloaded: {model['path']}")
    return model["path"]


class LandmarkExtractor:
    """
    Extracts landmarks from video frames using MediaPipe.
    
    Output format: (num_frames, 543, 3)
    - 543 landmarks per frame
    - 3 coordinates (x, y, z) normalized [0, 1]
    """
    
    def __init__(self, use_face=True, use_pose=True):
        """
        Initialize extractors.
        
        Args:
            use_face: Include face landmarks
            use_pose: Include pose landmarks
        """
        self.use_face = use_face
        self.use_pose = use_pose
        
        # Download models
        hand_path = download_model("hand")
        
        # Hand detector (always needed)
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(hand_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        
        # Face detector
        if use_face:
            face_path = download_model("face")
            face_options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(face_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5
            )
            self.face_detector = vision.FaceLandmarker.create_from_options(face_options)
        
        # Pose detector
        if use_pose:
            pose_path = download_model("pose")
            pose_options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(pose_path)),
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.5
            )
            self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
    
    def extract_frame(self, frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """
        Extract landmarks from a single frame.
        
        Args:
            frame: BGR image (H, W, 3)
            timestamp_ms: Frame timestamp in milliseconds
            
        Returns:
            landmarks: (543, 3) array of normalized coordinates
        """
        # Initialize with NaN
        landmarks = np.full((543, 3), np.nan, dtype=np.float32)
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Face landmarks (indices 0-467)
        if self.use_face:
            try:
                result = self.face_detector.detect_for_video(mp_image, timestamp_ms)
                if result.face_landmarks:
                    for lms in result.face_landmarks:
                        for i, lm in enumerate(lms):
                            if i < 468:
                                landmarks[i] = [lm.x, lm.y, lm.z]
            except Exception:
                pass
        
        # Hand landmarks
        try:
            result = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
            if result.hand_landmarks:
                for hand_idx, lms in enumerate(result.hand_landmarks):
                    handedness = result.handedness[hand_idx][0].category_name
                    # Left hand: 468-488, Right hand: 522-542
                    base_idx = 468 if handedness == "Left" else 522
                    for i, lm in enumerate(lms):
                        if i < 21:
                            landmarks[base_idx + i] = [lm.x, lm.y, lm.z]
        except Exception:
            pass
        
        # Pose landmarks (indices 489-521)
        if self.use_pose:
            try:
                result = self.pose_detector.detect_for_video(mp_image, timestamp_ms)
                if result.pose_landmarks:
                    for lms in result.pose_landmarks:
                        for i, lm in enumerate(lms):
                            if i < 33:
                                landmarks[489 + i] = [lm.x, lm.y, lm.z]
            except Exception:
                pass
        
        return landmarks
    
    def extract_video(self, video_path: str, max_frames: int = None) -> np.ndarray:
        """
        Extract landmarks from all frames in a video.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            landmarks: (num_frames, 543, 3) array
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        all_landmarks = []
        frame_idx = 0
        
        pbar = tqdm(total=total_frames, desc="Extracting landmarks")
        
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp_ms = int((frame_idx / fps) * 1000)
            landmarks = self.extract_frame(frame, timestamp_ms)
            all_landmarks.append(landmarks)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        return np.stack(all_landmarks)


def extract_from_video(input_path: str, output_path: str, 
                       use_face: bool = True, use_pose: bool = True):
    """
    Extract landmarks from video and save to .npy file.
    
    Args:
        input_path: Path to input video
        output_path: Path to output .npy file
        use_face: Include face landmarks
        use_pose: Include pose landmarks
    """
    extractor = LandmarkExtractor(use_face=use_face, use_pose=use_pose)
    landmarks = extractor.extract_video(input_path)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), landmarks)
    
    print(f"Saved landmarks: {output_path}")
    print(f"  Shape: {landmarks.shape}")
    print(f"  Valid frames: {np.sum(~np.isnan(landmarks[:, LHAND[0], 0]))}/{len(landmarks)}")


def main():
    parser = argparse.ArgumentParser(description="Extract landmarks from video")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", required=True, help="Output .npy path")
    parser.add_argument("--no-face", action="store_true", help="Skip face landmarks")
    parser.add_argument("--no-pose", action="store_true", help="Skip pose landmarks")
    
    args = parser.parse_args()
    
    extract_from_video(
        args.input, 
        args.output,
        use_face=not args.no_face,
        use_pose=not args.no_pose
    )


if __name__ == "__main__":
    main()
