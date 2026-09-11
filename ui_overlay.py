"""
UI Overlay Module.
Renders a futuristic Cyberpunk Heads-Up Display (HUD) on the OpenCV camera frame:
- Active screen boundary box with corner brackets
- Hand skeleton joints with glowing fingertip crosshair
- Dynamic pinch proximity gauge and click ripple animations
- Modern state pill badges (Moving, Click, Drag, Scroll, Paused)
- FPS counter and interactive keyboard cheat-sheet overlay
"""

import time
import math
from typing import Optional, List, Tuple
import cv2
import numpy as np

from hand_tracker import HandData, HAND_CONNECTIONS
from gesture_controller import GestureState
from config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    MARGIN_LEFT,
    MARGIN_RIGHT,
    MARGIN_TOP,
    MARGIN_BOTTOM,
    PINCH_CLICK_DIST,
    COLOR_ACCENT,
    COLOR_PRIMARY,
    COLOR_CLICK,
    COLOR_RIGHT_CLICK,
    COLOR_DRAG,
    COLOR_SCROLL,
    COLOR_PAUSED,
    COLOR_TEXT,
    COLOR_BG_CARD,
    COLOR_BORDER
)


class UIOverlay:
    """Renders visual feedback, HUD elements, and telemetry."""

    def __init__(self):
        self.show_help = False
        self.show_boundary = True

        # Click ripple animation state: list of (x, y, start_time, color)
        self.ripples: List[Tuple[int, int, float, Tuple[int, int, int]]] = []

    def trigger_click_ripple(self, x: int, y: int, color: Tuple[int, int, int] = COLOR_CLICK):
        """Spawns an expanding ripple at (x, y)."""
        self.ripples.append((x, y, time.time(), color))

    def draw_corner_bracket(self, frame: np.ndarray, pt: Tuple[int, int], dx: int, dy: int, color: Tuple[int, int, int], length: int = 16, thickness: int = 2):
        """Draws an L-shaped sci-fi corner bracket."""
        x, y = pt
        cv2.line(frame, (x, y), (x + dx * length, y), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + dy * length), color, thickness, cv2.LINE_AA)

    def draw_active_boundary(self, frame: np.ndarray):
        """Draws the screen mapping boundary box with corner brackets."""
        x1 = MARGIN_LEFT
        y1 = MARGIN_TOP
        x2 = CAMERA_WIDTH - MARGIN_RIGHT
        y2 = CAMERA_HEIGHT - MARGIN_BOTTOM

        # Subtle boundary rectangle
        sub_overlay = frame.copy()
        cv2.rectangle(sub_overlay, (x1, y1), (x2, y2), COLOR_BORDER, 1, cv2.LINE_AA)
        cv2.addWeighted(sub_overlay, 0.4, frame, 0.6, 0, frame)

        # Tech Corner Brackets
        self.draw_corner_bracket(frame, (x1, y1), 1, 1, COLOR_PRIMARY, length=18, thickness=2)
        self.draw_corner_bracket(frame, (x2, y1), -1, 1, COLOR_PRIMARY, length=18, thickness=2)
        self.draw_corner_bracket(frame, (x1, y2), 1, -1, COLOR_PRIMARY, length=18, thickness=2)
        self.draw_corner_bracket(frame, (x2, y2), -1, -1, COLOR_PRIMARY, length=18, thickness=2)

        # Boundary Label
        cv2.putText(
            frame,
            "SCREEN ACTIVE ZONE",
            (x1 + 6, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (160, 160, 180),
            1,
            cv2.LINE_AA
        )

    def draw_hand_skeleton(self, frame: np.ndarray, hand: HandData):
        """Draws glowing skeleton connections and joints."""
        # Draw connection lines
        for id1, id2 in HAND_CONNECTIONS:
            pt1 = hand.pixel_landmarks[id1]
            pt2 = hand.pixel_landmarks[id2]
            cv2.line(frame, pt1, pt2, (100, 100, 120), 1, cv2.LINE_AA)

        # Draw joints
        for i, pt in enumerate(hand.pixel_landmarks):
            if i in [4, 8, 12, 16, 20]:
                # Fingertips: larger accent circles
                cv2.circle(frame, pt, 5, COLOR_PRIMARY, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 7, COLOR_ACCENT, 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 3, (160, 160, 180), -1, cv2.LINE_AA)

        # Index Fingertip Target Crosshair
        ix, iy = hand.index_tip
        cv2.circle(frame, (ix, iy), 14, COLOR_ACCENT, 1, cv2.LINE_AA)
        cv2.line(frame, (ix - 18, iy), (ix - 10, iy), COLOR_ACCENT, 1, cv2.LINE_AA)
        cv2.line(frame, (ix + 10, iy), (ix + 18, iy), COLOR_ACCENT, 1, cv2.LINE_AA)
        cv2.line(frame, (ix, iy - 18), (ix, iy - 10), COLOR_ACCENT, 1, cv2.LINE_AA)
        cv2.line(frame, (ix, iy + 10), (ix, iy + 18), COLOR_ACCENT, 1, cv2.LINE_AA)

    def draw_pinch_gauge(self, frame: np.ndarray, hand: HandData, pinch_ratio: float, state: GestureState):
        """Draws elastic line between thumb and index with proximity gauge."""
        tx, ty = hand.thumb_tip
        ix, iy = hand.index_tip
        mx, my = int((tx + ix) / 2), int((ty + iy) / 2)

        # Dynamic color interpolation: Cyan (open) -> Emerald Green (pinched)
        r = int(COLOR_PRIMARY[0] * (1.0 - pinch_ratio) + COLOR_CLICK[0] * pinch_ratio)
        g = int(COLOR_PRIMARY[1] * (1.0 - pinch_ratio) + COLOR_CLICK[1] * pinch_ratio)
        b = int(COLOR_PRIMARY[2] * (1.0 - pinch_ratio) + COLOR_CLICK[2] * pinch_ratio)
        line_color = (r, g, b)

        # Elastic connecting line
        cv2.line(frame, (tx, ty), (ix, iy), line_color, 2 if pinch_ratio > 0.5 else 1, cv2.LINE_AA)

        # Pinch gauge circle at midpoint
        gauge_radius = int(6 + 8 * pinch_ratio)
        cv2.circle(frame, (mx, my), gauge_radius, line_color, -1 if pinch_ratio > 0.85 else 1, cv2.LINE_AA)

    def draw_ripples(self, frame: np.ndarray):
        """Updates and renders expanding click ripples."""
        now = time.time()
        active_ripples = []
        max_duration = 0.4  # seconds

        for x, y, start_t, color in self.ripples:
            elapsed = now - start_t
            if elapsed < max_duration:
                progress = elapsed / max_duration
                radius = int(8 + progress * 35)
                alpha = 1.0 - progress
                # Draw fading ripple ring
                overlay = frame.copy()
                cv2.circle(overlay, (x, y), radius, color, 2, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
                active_ripples.append((x, y, start_t, color))

        self.ripples = active_ripples

    def draw_header_hud(
        self,
        frame: np.ndarray,
        state: GestureState,
        action_text: str,
        fps: float,
        profile_name: str,
        handedness: Optional[str]
    ):
        """Draws modern glassmorphic top header bar with status badges."""
        # Top banner background
        header_h = 42
        header_bg = frame[0:header_h, 0:CAMERA_WIDTH].copy()
        cv2.rectangle(header_bg, (0, 0), (CAMERA_WIDTH, header_h), (20, 20, 25), -1)
        cv2.addWeighted(header_bg, 0.85, frame[0:header_h, 0:CAMERA_WIDTH], 0.15, 0, frame[0:header_h, 0:CAMERA_WIDTH])
        cv2.line(frame, (0, header_h), (CAMERA_WIDTH, header_h), COLOR_BORDER, 1)

        # Determine state badge color
        state_colors = {
            GestureState.MOVING: COLOR_PRIMARY,
            GestureState.PINCHING: COLOR_ACCENT,
            GestureState.DRAGGING: COLOR_DRAG,
            GestureState.RIGHT_CLICK: COLOR_RIGHT_CLICK,
            GestureState.SCROLLING: COLOR_SCROLL,
            GestureState.PAUSED: COLOR_PAUSED,
            GestureState.IDLE: (140, 140, 150)
        }
        badge_color = state_colors.get(state, COLOR_PRIMARY)

        # State Pill Badge
        state_text = f" {state.value} "
        if state == GestureState.MOVING:
            state_text = " TRACKING "
        elif state == GestureState.PINCHING:
            state_text = " PINCH "

        cv2.rectangle(frame, (10, 8), (140, 34), badge_color, -1)
        cv2.rectangle(frame, (10, 8), (140, 34), COLOR_TEXT, 1)
        cv2.putText(frame, state_text, (16, 26), cv2.FONT_HERSHEY_DUPLEX, 0.45, (10, 10, 15), 1, cv2.LINE_AA)

        # Action Details
        cv2.putText(frame, action_text, (150, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT, 1, cv2.LINE_AA)

        # Right-side Badges: FPS & Profile
        fps_str = f"{int(fps)} FPS"
        prof_str = f"Mode: {profile_name}"
        hand_str = f"Hand: {handedness or 'None'}"

        info_text = f"{fps_str}  |  {prof_str}  |  {hand_str}"
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
        right_x = CAMERA_WIDTH - text_size[0] - 12
        cv2.putText(frame, info_text, (right_x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 200, 220), 1, cv2.LINE_AA)

    def draw_footer_bar(self, frame: np.ndarray):
        """Draws the bottom hotkey quick reference bar."""
        footer_h = 24
        y1 = CAMERA_HEIGHT - footer_h
        footer_bg = frame[y1:CAMERA_HEIGHT, 0:CAMERA_WIDTH].copy()
        cv2.rectangle(footer_bg, (0, 0), (CAMERA_WIDTH, footer_h), (15, 15, 20), -1)
        cv2.addWeighted(footer_bg, 0.85, frame[y1:CAMERA_HEIGHT, 0:CAMERA_WIDTH], 0.15, 0, frame[y1:CAMERA_HEIGHT, 0:CAMERA_WIDTH])
        cv2.line(frame, (0, y1), (CAMERA_WIDTH, y1), COLOR_BORDER, 1)

        key_legend = "[Q] Quit   [P] Pause   [C] Zone Box   [S] Sensitivity   [H] Help Card"
        cv2.putText(frame, key_legend, (16, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 180, 190), 1, cv2.LINE_AA)

    def draw_help_card(self, frame: np.ndarray):
        """Draws modal overlay detailing all gesture commands."""
        card_w = 460
        card_h = 300
        x1 = int((CAMERA_WIDTH - card_w) / 2)
        y1 = int((CAMERA_HEIGHT - card_h) / 2)
        x2 = x1 + card_w
        y2 = y1 + card_h

        # Semi-transparent dark card background
        card_bg = frame[y1:y2, x1:x2].copy()
        cv2.rectangle(card_bg, (0, 0), (card_w, card_h), (25, 25, 30), -1)
        cv2.addWeighted(card_bg, 0.92, frame[y1:y2, x1:x2], 0.08, 0, frame[y1:y2, x1:x2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PRIMARY, 2)

        # Title
        cv2.putText(frame, "VIRTUAL MOUSE GESTURE GUIDE", (x1 + 18, y1 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_ACCENT, 1, cv2.LINE_AA)
        cv2.line(frame, (x1 + 18, y1 + 38), (x2 - 18, y1 + 38), COLOR_BORDER, 1)

        gestures = [
            ("Move Cursor", "Point and move Index Fingertip"),
            ("Left Click", "Pinch Index + Thumb quickly and release"),
            ("Double Click", "Quick double pinch Index + Thumb"),
            ("Drag & Drop", "Pinch Index + Thumb and hold > 0.3s; release to drop"),
            ("Right Click", "Pinch Middle Finger + Thumb"),
            ("Scroll Up/Down", "Lift Index + Middle together and move up/down"),
            ("Safety Pause", "Open flat palm for 1 sec, or press 'P' key"),
            ("Quit System", "Press 'Q' key in application window")
        ]

        row_y = y1 + 65
        for title, desc in gestures:
            cv2.circle(frame, (x1 + 25, row_y - 4), 3, COLOR_PRIMARY, -1, cv2.LINE_AA)
            cv2.putText(frame, title + ":", (x1 + 36, row_y), cv2.FONT_HERSHEY_DUPLEX, 0.38, COLOR_TEXT, 1, cv2.LINE_AA)
            cv2.putText(frame, desc, (x1 + 155, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 190, 200), 1, cv2.LINE_AA)
            row_y += 26

        cv2.putText(frame, "Press 'H' to close this guide", (x1 + 130, y2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_ACCENT, 1, cv2.LINE_AA)

    def render(
        self,
        frame: np.ndarray,
        hand: Optional[HandData],
        state: GestureState,
        action_text: str,
        pinch_ratio: float,
        fps: float,
        profile_name: str
    ) -> np.ndarray:
        """Assembles all UI layers onto the camera frame."""
        # 1. Active screen mapping boundary
        if self.show_boundary:
            self.draw_active_boundary(frame)

        # 2. Hand skeleton & pinch gauge
        if hand is not None:
            self.draw_hand_skeleton(frame, hand)
            if state != GestureState.SCROLLING and state != GestureState.PAUSED:
                self.draw_pinch_gauge(frame, hand, pinch_ratio, state)

        # 3. Dynamic ripples
        self.draw_ripples(frame)

        # 4. Header HUD banner
        handedness = hand.handedness if hand else None
        self.draw_header_hud(frame, state, action_text, fps, profile_name, handedness)

        # 5. Footer quick shortcuts
        self.draw_footer_bar(frame)

        # 6. Help modal if toggled
        if self.show_help:
            self.draw_help_card(frame)

        return frame
