"""
Hand Tracking Module using MediaPipe Tasks Vision (HandLandmarker).
Provides hand landmark detection, landmark coordinate scaling, distance calculation,
and finger posture classification.
"""

import os
import math
import urllib.request
from typing import List, Tuple, Optional, NamedTuple
import cv2
import numpy as np
import mediapipe as mp

from config import (
    MODEL_PATH,
    MODEL_URL,
    NUM_HANDS,
    MIN_HAND_DETECTION_CONFIDENCE,
    MIN_HAND_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE
)

# Standard MediaPipe Hand Landmark Connections for Skeleton Drawing
HAND_CONNECTIONS = [
    # Palm
    (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index finger
    (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (17, 18), (18, 19), (19, 20)
]


class HandData:
    """Encapsulates detected hand landmarks and posture metrics."""

    def __init__(
        self,
        pixel_landmarks: List[Tuple[int, int]],
        norm_landmarks: List[Tuple[float, float, float]],
        handedness: str,
        frame_width: int,
        frame_height: int
    ):
        self.pixel_landmarks = pixel_landmarks  # 21 tuples: (x_px, y_px)
        self.norm_landmarks = norm_landmarks    # 21 tuples: (x_norm, y_norm, z_norm)
        self.handedness = handedness            # 'Right' or 'Left'
        self.width = frame_width
        self.height = frame_height

    @property
    def wrist(self) -> Tuple[int, int]:
        return self.pixel_landmarks[0]

    @property
    def thumb_tip(self) -> Tuple[int, int]:
        return self.pixel_landmarks[4]

    @property
    def index_tip(self) -> Tuple[int, int]:
        return self.pixel_landmarks[8]

    @property
    def middle_tip(self) -> Tuple[int, int]:
        return self.pixel_landmarks[12]

    @property
    def ring_tip(self) -> Tuple[int, int]:
        return self.pixel_landmarks[16]

    @property
    def pinky_tip(self) -> Tuple[int, int]:
        return self.pixel_landmarks[20]

    def distance_pixels(self, idx1: int, idx2: int) -> float:
        """Euclidean distance in pixels between two landmark indices."""
        x1, y1 = self.pixel_landmarks[idx1]
        x2, y2 = self.pixel_landmarks[idx2]
        return math.hypot(x1 - x2, y1 - y2)

    def midpoint(self, idx1: int, idx2: int) -> Tuple[int, int]:
        """Calculates integer midpoint between two landmarks."""
        x1, y1 = self.pixel_landmarks[idx1]
        x2, y2 = self.pixel_landmarks[idx2]
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def get_finger_states(self) -> List[bool]:
        """
        Determines whether each finger is extended (True) or folded (False).
        Returns list of 5 booleans: [Thumb, Index, Middle, Ring, Pinky].
        """
        states = []
        p = self.pixel_landmarks

        # Thumb: compare tip (4) to IP joint (3) and MCP (2) relative to wrist
        # For mirrored webcam, distance between thumb tip and pinky MCP (17) is greater when extended
        thumb_tip_to_pinky_mcp = self.distance_pixels(4, 17)
        thumb_ip_to_pinky_mcp = self.distance_pixels(3, 17)
        thumb_extended = thumb_tip_to_pinky_mcp > (thumb_ip_to_pinky_mcp * 1.15)
        states.append(thumb_extended)

        # 4 Fingers (Index, Middle, Ring, Pinky):
        # Finger is extended if TIP is higher up (smaller y) than PIP joint (by margin)
        finger_indices = [(8, 6), (12, 10), (16, 14), (20, 18)]
        for tip_idx, pip_idx in finger_indices:
            # Tip y is smaller than PIP joint y (remember y=0 is top of image)
            is_extended = p[tip_idx][1] < p[pip_idx][1]
            states.append(is_extended)

        return states


class HandTracker:
    """Manages MediaPipe HandLandmarker lifecycle and frame detection."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self._ensure_model_exists()

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

        self.landmarker = HandLandmarker.create_from_options(options)

    def _ensure_model_exists(self):
        """Verifies model file exists; downloads if missing."""
        if not os.path.exists(self.model_path) or os.path.getsize(self.model_path) < 1000000:
            print(f"Downloading hand landmarker model to {self.model_path}...")
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            print("Download complete.")

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[HandData]:
        """
        Process an OpenCV BGR frame and return HandData for the primary detected hand.
        """
        h, w, _ = frame_bgr.shape
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.hand_landmarks:
            return None

        # Take primary hand (first detected)
        primary_landmarks = detection_result.hand_landmarks[0]

        # Determine handedness label if available
        handedness_label = "Right"
        if detection_result.handedness and len(detection_result.handedness) > 0:
            categories = detection_result.handedness[0]
            if len(categories) > 0:
                handedness_label = categories[0].category_name

        # Convert to pixel coordinates
        pixel_landmarks = []
        norm_landmarks = []
        for lm in primary_landmarks:
            px_x = int(lm.x * w)
            px_y = int(lm.y * h)
            # Clamp to frame boundaries
            px_x = max(0, min(w - 1, px_x))
            px_y = max(0, min(h - 1, px_y))
            pixel_landmarks.append((px_x, px_y))
            norm_landmarks.append((lm.x, lm.y, lm.z))

        return HandData(
            pixel_landmarks=pixel_landmarks,
            norm_landmarks=norm_landmarks,
            handedness=handedness_label,
            frame_width=w,
            frame_height=h
        )

    def close(self):
        """Releases the landmarker resources."""
        if self.landmarker:
            self.landmarker.close()
