"""
Finger Tracking Virtual Mouse System - Main Application.
Integrates real-time webcam feed, MediaPipe hand tracking, gesture interpretation,
zero-latency cursor motion, and a futuristic Heads-Up Display.
"""

import sys
import time
import cv2
import numpy as np

from config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    SMOOTHING_PROFILES,
    DEFAULT_PROFILE
)
from hand_tracker import HandTracker
from filter import AdaptiveSmoothingFilter
from mouse_controller import MouseController
from gesture_controller import GestureController, GestureState
from ui_overlay import UIOverlay


def print_banner():
    banner = f"""
========================================================================
            AI FINGER TRACKING VIRTUAL MOUSE SYSTEM
========================================================================
  [+] Screen Resolution : {SCREEN_WIDTH} x {SCREEN_HEIGHT}
  [+] Camera Resolution : {CAMERA_WIDTH} x {CAMERA_HEIGHT} @ {CAMERA_FPS} FPS
  [+] Engine            : Google MediaPipe Tasks HandLandmarker
  [+] Driver            : Windows Native User32 / Low-latency Input

  GESTURE CONTROLS:
    * Move Cursor       : Point & move Index Fingertip
    * Left Click        : Pinch Index + Thumb quickly
    * Double Click      : Fast double pinch Index + Thumb
    * Drag & Drop       : Pinch Index + Thumb and hold > 0.3s
    * Right Click       : Pinch Middle Finger + Thumb
    * Scroll Up / Down  : Two fingers (Index + Middle) up & move up/down
    * Safety Pause      : Open flat palm for 1s, or press 'P' key

  HOTKEYS:
    [Q] Quit    [P] Pause/Resume    [S] Switch Sensitivity
    [C] Toggle Active Box           [H] Show/Hide Gesture Guide
========================================================================
    """
    print(banner)


def main():
    print_banner()

    # 1. Initialize Camera
    print(f"Connecting to camera index {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"Warning: Could not open camera with DirectShow. Falling back to default backend...")
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"ERROR: Failed to open webcam at index {CAMERA_INDEX}. Please check webcam connection.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    # 2. Initialize Subsystems
    print("Loading Hand Landmarker AI Model...")
    tracker = HandTracker()
    mouse = MouseController()

    profile_keys = list(SMOOTHING_PROFILES.keys())
    profile_idx = profile_keys.index(DEFAULT_PROFILE) if DEFAULT_PROFILE in profile_keys else 0
    current_profile_name = profile_keys[profile_idx]
    prof = SMOOTHING_PROFILES[current_profile_name]

    smoother = AdaptiveSmoothingFilter(
        alpha_min=prof["alpha_min"],
        alpha_max=prof["alpha_max"],
        velocity_scale=prof["velocity_scale"]
    )

    gesture = GestureController(mouse=mouse, filter_smoother=smoother)
    ui = UIOverlay()

    window_name = "AI Finger Tracking Virtual Mouse"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.time()
    fps = 30.0

    print("System active! Position your hand in front of the camera.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Dropped camera frame.")
                continue

            # Mirror frame horizontally for natural mirror interaction
            frame = cv2.flip(frame, 1)

            # Calculate instantaneous FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Process hand detection
            hand = tracker.process_frame(frame)

            # Update gestures and mouse dispatch
            state, action_desc = gesture.update(hand)

            # Trigger ripple visual on click events
            if action_desc in ["Left Click", "Double Click"] and hand:
                ui.trigger_click_ripple(hand.index_tip[0], hand.index_tip[1])
            elif action_desc == "Right Click" and hand:
                ui.trigger_click_ripple(hand.middle_tip[0], hand.middle_tip[1], color=(255, 70, 200))

            # Render Heads-Up Display
            annotated_frame = ui.render(
                frame=frame,
                hand=hand,
                state=state,
                action_text=action_desc,
                pinch_ratio=gesture.pinch_ratio,
                fps=fps,
                profile_name=current_profile_name
            )

            cv2.imshow(window_name, annotated_frame)

            # Handle Keyboard Shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\nExiting application...")
                break
            elif key == ord('p') or key == ord(' '):  # 'p' or Space
                gesture.toggle_pause()
                print(f"[HotKey] Tracking {'Paused' if gesture.is_paused else 'Resumed'}")
            elif key == ord('c'):
                ui.show_boundary = not ui.show_boundary
                print(f"[HotKey] Screen active boundary: {'ON' if ui.show_boundary else 'OFF'}")
            elif key == ord('h'):
                ui.show_help = not ui.show_help
                print(f"[HotKey] Gesture help guide: {'ON' if ui.show_help else 'OFF'}")
            elif key == ord('s'):
                profile_idx = (profile_idx + 1) % len(profile_keys)
                current_profile_name = profile_keys[profile_idx]
                p = SMOOTHING_PROFILES[current_profile_name]
                smoother.update_profile(p["alpha_min"], p["alpha_max"], p["velocity_scale"])
                print(f"[HotKey] Switched smoothing profile to: {current_profile_name}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Cleanup and release
        print("Cleaning up resources...")
        mouse.release_all()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Finger Tracking Mouse System stopped successfully.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
