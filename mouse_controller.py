"""
Mouse Controller Module.
Handles coordinate mapping from camera active zone to full desktop screen resolution
and dispatches ultra-fast, zero-latency mouse events using Windows user32 / PyAutoGUI.
"""

import sys
import ctypes
from typing import Tuple
import pyautogui

from config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    MARGIN_LEFT,
    MARGIN_RIGHT,
    MARGIN_TOP,
    MARGIN_BOTTOM
)

# Disable PyAutoGUI artificial delays
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# Windows mouse_event flags
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
WHEEL_DELTA = 120


class MouseController:
    """Dispatches native Windows cursor and button events."""

    def __init__(self):
        self.is_windows = sys.platform.startswith("win")
        if self.is_windows:
            self.user32 = ctypes.windll.user32
        else:
            self.user32 = None

        self.screen_w = SCREEN_WIDTH
        self.screen_h = SCREEN_HEIGHT

        # Active boundary coordinates inside camera frame
        self.x_min = MARGIN_LEFT
        self.x_max = CAMERA_WIDTH - MARGIN_RIGHT
        self.y_min = MARGIN_TOP
        self.y_max = CAMERA_HEIGHT - MARGIN_BOTTOM

        self.is_left_down = False
        self.is_right_down = False

    def map_to_screen(self, cam_x: float, cam_y: float) -> Tuple[int, int]:
        """
        Maps a camera frame coordinate into the active tracking zone,
        then scales to screen resolution with boundary clamping.
        """
        # Normalize within active zone [0.0, 1.0]
        norm_x = (cam_x - self.x_min) / max(1.0, (self.x_max - self.x_min))
        norm_y = (cam_y - self.y_min) / max(1.0, (self.y_max - self.y_min))

        # Clamp to bounds
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        # Scale to desktop screen resolution
        screen_x = int(norm_x * (self.screen_w - 1))
        screen_y = int(norm_y * (self.screen_h - 1))

        return screen_x, screen_y

    def move_to(self, screen_x: int, screen_y: int):
        """Move cursor instantly to (screen_x, screen_y)."""
        screen_x = max(0, min(self.screen_w - 1, int(screen_x)))
        screen_y = max(0, min(self.screen_h - 1, int(screen_y)))

        if self.user32:
            self.user32.SetCursorPos(screen_x, screen_y)
        else:
            pyautogui.moveTo(screen_x, screen_y)

    def left_down(self):
        """Press down left mouse button (for drag)."""
        if not self.is_left_down:
            if self.user32:
                self.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            else:
                pyautogui.mouseDown(button='left')
            self.is_left_down = True

    def left_up(self):
        """Release left mouse button."""
        if self.is_left_down:
            if self.user32:
                self.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            else:
                pyautogui.mouseUp(button='left')
            self.is_left_down = False

    def left_click(self):
        """Perform a single left click."""
        if self.user32:
            self.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            self.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        else:
            pyautogui.click(button='left')

    def double_click(self):
        """Perform a double click."""
        if self.user32:
            self.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            self.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            self.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            self.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        else:
            pyautogui.doubleClick(button='left')

    def right_click(self):
        """Perform a single right click."""
        if self.user32:
            self.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
            self.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
        else:
            pyautogui.click(button='right')

    def scroll(self, notches: int):
        """
        Scroll mouse wheel. Positive values scroll UP, negative scroll DOWN.
        """
        if notches == 0:
            return
        if self.user32:
            # notches * WHEEL_DELTA
            wheel_amount = int(notches * WHEEL_DELTA)
            self.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, wheel_amount, 0)
        else:
            pyautogui.scroll(int(notches * 60))

    def release_all(self):
        """Emergency release of any held mouse buttons."""
        if self.is_left_down:
            self.left_up()
        if self.is_right_down:
            if self.user32:
                self.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            else:
                pyautogui.mouseUp(button='right')
            self.is_right_down = False
