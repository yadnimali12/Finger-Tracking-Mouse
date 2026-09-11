"""
Configuration Module for Finger Tracking Mouse System.
Defines camera, screen, tracking bounds, gesture thresholds, and visual styles.
"""

import os
import ctypes
import pyautogui

# ---------------------------------------------------------------------------
# Screen Resolution (Auto-detected)
# ---------------------------------------------------------------------------
try:
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    SCREEN_WIDTH = user32.GetSystemMetrics(0)
    SCREEN_HEIGHT = user32.GetSystemMetrics(1)
except Exception:
    SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# ---------------------------------------------------------------------------
# Camera Configuration
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ---------------------------------------------------------------------------
# MediaPipe HandLandmarker Model
# ---------------------------------------------------------------------------
MODEL_FILENAME = "hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

NUM_HANDS = 1
MIN_HAND_DETECTION_CONFIDENCE = 0.6
MIN_HAND_PRESENCE_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

# ---------------------------------------------------------------------------
# Active Mapping Boundary (Margins in camera frame pixels)
# Prevents user from having to reach outside webcam view to reach screen edges.
# ---------------------------------------------------------------------------
MARGIN_LEFT = 85
MARGIN_RIGHT = 85
MARGIN_TOP = 70
MARGIN_BOTTOM = 110

# ---------------------------------------------------------------------------
# Dynamic Smoothing & Anti-Jitter Settings
# ---------------------------------------------------------------------------
SMOOTHING_PROFILES = {
    "Balanced": {
        "alpha_min": 0.18,      # Steady cursor at standstill / micro-motions
        "alpha_max": 0.82,      # Responsive cursor at fast speeds
        "velocity_scale": 30.0  # Speed transition factor
    },
    "Ultra Smooth": {
        "alpha_min": 0.10,
        "alpha_max": 0.65,
        "velocity_scale": 40.0
    },
    "Fast / Gaming": {
        "alpha_min": 0.35,
        "alpha_max": 0.95,
        "velocity_scale": 20.0
    }
}
DEFAULT_PROFILE = "Balanced"

# ---------------------------------------------------------------------------
# Gesture Thresholds (Distances in pixels calibrated for 640x480 frame)
# ---------------------------------------------------------------------------
# Left click: Thumb tip (4) to Index tip (8)
PINCH_CLICK_DIST = 32
PINCH_RELEASE_DIST = 46

# Right click: Thumb tip (4) to Middle tip (12)
RIGHT_CLICK_DIST = 32
RIGHT_RELEASE_DIST = 46

# Hold duration for Drag & Drop (in seconds)
DRAG_HOLD_SECONDS = 0.32

# Double click max time gap (seconds)
DOUBLE_CLICK_MAX_GAP = 0.35

# Two-finger scrolling: Index (8) and Middle (12) up together
SCROLL_FINGER_MAX_DIST = 42    # Index tip to Middle tip max separation for scroll
SCROLL_DEADZONE = 4            # Min pixels movement before scroll event fires
SCROLL_MULTIPLIER = 2.4        # Sensitivity of scroll wheel

# Safety Pause gesture (Open flat palm held for duration)
PAUSE_HOLD_SECONDS = 1.0

# ---------------------------------------------------------------------------
# Visual Styles (Cyberpunk / Modern Neon Theme - BGR Colors for OpenCV)
# ---------------------------------------------------------------------------
COLOR_ACCENT = (255, 215, 0)       # Electric Cyan / Gold
COLOR_PRIMARY = (255, 120, 20)     # Neon Azure Blue
COLOR_CLICK = (60, 255, 120)       # Emerald Green
COLOR_RIGHT_CLICK = (255, 70, 200) # Bright Magenta
COLOR_DRAG = (30, 160, 255)        # Vivid Amber
COLOR_SCROLL = (240, 200, 50)      # Sky Blue
COLOR_PAUSED = (100, 100, 120)     # Slate Muted
COLOR_TEXT = (255, 255, 255)       # Crisp White
COLOR_BG_CARD = (30, 30, 35)       # Deep Obsidian
COLOR_BORDER = (70, 70, 80)        # Soft Outline
