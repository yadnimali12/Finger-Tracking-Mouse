"""
Gesture Controller Module.
Implements a state machine to recognize natural finger gestures:
- Pointer Navigation
- Left Click & Double Click
- Drag and Drop
- Right Click
- Two-Finger Smooth Scrolling
- Open Palm Safety Pause
"""

import time
from enum import Enum
from typing import Tuple, Optional
from hand_tracker import HandData
from mouse_controller import MouseController
from filter import AdaptiveSmoothingFilter
from config import (
    PINCH_CLICK_DIST,
    PINCH_RELEASE_DIST,
    RIGHT_CLICK_DIST,
    RIGHT_RELEASE_DIST,
    DRAG_HOLD_SECONDS,
    DOUBLE_CLICK_MAX_GAP,
    SCROLL_FINGER_MAX_DIST,
    SCROLL_DEADZONE,
    SCROLL_MULTIPLIER,
    PAUSE_HOLD_SECONDS
)


class GestureState(Enum):
    IDLE = "IDLE"
    MOVING = "MOVING"
    PINCHING = "PINCHING"
    DRAGGING = "DRAGGING"
    RIGHT_CLICK = "RIGHT_CLICK"
    SCROLLING = "SCROLLING"
    PAUSED = "PAUSED"


class GestureController:
    """Interprets hand landmarks and dispatches mouse actions with debouncing."""

    def __init__(self, mouse: MouseController, filter_smoother: AdaptiveSmoothingFilter):
        self.mouse = mouse
        self.smoother = filter_smoother

        self.state = GestureState.IDLE
        self.is_paused = False

        # Left pinch / click timing
        self.pinch_start_time: Optional[float] = None
        self.last_click_time: float = 0.0
        self.is_pinched = False

        # Right click state
        self.is_right_pinched = False
        self.right_click_fired = False

        # Scrolling state
        self.last_scroll_y: Optional[float] = None

        # Pause gesture state (flat open palm)
        self.palm_open_start_time: Optional[float] = None
        self.last_pause_toggle_time: float = 0.0

        # UI metrics
        self.pinch_ratio: float = 0.0  # 0.0 (open) to 1.0 (fully pinched)
        self.last_action_desc: str = "Ready"

    def toggle_pause(self):
        """Toggles tracking pause on/off."""
        self.is_paused = not self.is_paused
        self.mouse.release_all()
        self.smoother.reset()
        if self.is_paused:
            self.state = GestureState.PAUSED
            self.last_action_desc = "Paused"
        else:
            self.state = GestureState.IDLE
            self.last_action_desc = "Resumed"

    def update(self, hand: Optional[HandData]) -> Tuple[GestureState, str]:
        """
        Process a new hand frame and execute appropriate mouse actions.
        Returns (current_state, action_description).
        """
        now = time.time()

        if hand is None:
            if self.is_pinched and self.state == GestureState.DRAGGING:
                self.mouse.left_up()
                self.state = GestureState.IDLE
                self.is_pinched = False
            self.smoother.reset()
            self.last_scroll_y = None
            self.palm_open_start_time = None
            if not self.is_paused:
                self.state = GestureState.IDLE
            return self.state, self.last_action_desc

        finger_states = hand.get_finger_states()
        thumb_up, index_up, middle_up, ring_up, pinky_up = finger_states

        # -------------------------------------------------------------------
        # 1. Open Palm Safety Pause Detection
        # -------------------------------------------------------------------
        all_fingers_up = all(finger_states)
        if all_fingers_up:
            if self.palm_open_start_time is None:
                self.palm_open_start_time = now
            elif (now - self.palm_open_start_time) >= PAUSE_HOLD_SECONDS:
                if (now - self.last_pause_toggle_time) > 1.5:  # Debounce toggle
                    self.toggle_pause()
                    self.last_pause_toggle_time = now
                    self.palm_open_start_time = None
                    return self.state, self.last_action_desc
        else:
            self.palm_open_start_time = None

        if self.is_paused:
            self.state = GestureState.PAUSED
            return self.state, "Tracking Paused (Press 'P' or show Open Palm)"

        # -------------------------------------------------------------------
        # Distance Calculations
        # -------------------------------------------------------------------
        # Index tip (8) to Thumb tip (4)
        dist_thumb_index = hand.distance_pixels(4, 8)
        # Middle tip (12) to Thumb tip (4)
        dist_thumb_middle = hand.distance_pixels(4, 12)
        # Index tip (8) to Middle tip (12)
        dist_index_middle = hand.distance_pixels(8, 12)

        # Calculate visual pinch ratio for HUD gauge
        denom = max(1.0, float(PINCH_RELEASE_DIST - PINCH_CLICK_DIST))
        raw_ratio = 1.0 - ((dist_thumb_index - PINCH_CLICK_DIST) / denom)
        self.pinch_ratio = max(0.0, min(1.0, raw_ratio))

        # Raw index fingertip coordinate
        raw_ix, raw_iy = hand.index_tip

        # -------------------------------------------------------------------
        # 2. Two-Finger Scrolling Mode
        # Both Index & Middle fingers up, Ring & Pinky curled
        # -------------------------------------------------------------------
        if index_up and middle_up and not ring_up and not pinky_up and (dist_index_middle < SCROLL_FINGER_MAX_DIST):
            self.state = GestureState.SCROLLING
            # Release any active drag
            if self.is_pinched:
                self.mouse.left_up()
                self.is_pinched = False

            mid_x, mid_y = hand.midpoint(8, 12)

            if self.last_scroll_y is not None:
                delta_y = mid_y - self.last_scroll_y
                if abs(delta_y) > SCROLL_DEADZONE:
                    # Inverted for intuitive wheel scrolling:
                    # Moving fingers UP (smaller y) -> scroll UP (positive clicks)
                    # Moving fingers DOWN (larger y) -> scroll DOWN (negative clicks)
                    scroll_clicks = -int(delta_y * SCROLL_MULTIPLIER)
                    if scroll_clicks != 0:
                        self.mouse.scroll(scroll_clicks)
                        direction = "UP" if scroll_clicks > 0 else "DOWN"
                        self.last_action_desc = f"Scrolling {direction}"

            self.last_scroll_y = mid_y
            return self.state, self.last_action_desc

        self.last_scroll_y = None

        # -------------------------------------------------------------------
        # 3. Right Click (Thumb + Middle Finger Pinch)
        # -------------------------------------------------------------------
        if dist_thumb_middle < RIGHT_CLICK_DIST:
            if not self.is_right_pinched and not self.right_click_fired:
                self.is_right_pinched = True
                self.right_click_fired = True
                self.mouse.right_click()
                self.state = GestureState.RIGHT_CLICK
                self.last_action_desc = "Right Click"
                return self.state, self.last_action_desc
        elif dist_thumb_middle > RIGHT_RELEASE_DIST:
            self.is_right_pinched = False
            self.right_click_fired = False

        # -------------------------------------------------------------------
        # 4. Cursor Navigation & Smoothing
        # -------------------------------------------------------------------
        smooth_x, smooth_y = self.smoother.step(raw_ix, raw_iy)
        screen_x, screen_y = self.mouse.map_to_screen(smooth_x, smooth_y)
        self.mouse.move_to(screen_x, screen_y)

        # -------------------------------------------------------------------
        # 5. Left Click / Drag & Drop / Double Click
        # -------------------------------------------------------------------
        if dist_thumb_index < PINCH_CLICK_DIST:
            if not self.is_pinched:
                self.is_pinched = True
                self.pinch_start_time = now

            pinch_duration = now - self.pinch_start_time if self.pinch_start_time else 0.0

            # If held longer than DRAG_HOLD_SECONDS, initiate drag
            if pinch_duration >= DRAG_HOLD_SECONDS:
                if self.state != GestureState.DRAGGING:
                    self.state = GestureState.DRAGGING
                    self.mouse.left_down()
                    self.last_action_desc = "Dragging (Hold)"
            else:
                self.state = GestureState.PINCHING
                self.last_action_desc = "Pinch"

        elif dist_thumb_index > PINCH_RELEASE_DIST:
            if self.is_pinched:
                pinch_duration = now - self.pinch_start_time if self.pinch_start_time else 0.0

                if self.state == GestureState.DRAGGING:
                    # End drag
                    self.mouse.left_up()
                    self.last_action_desc = "Dropped"
                else:
                    # Quick tap -> Click or Double Click
                    if (now - self.last_click_time) < DOUBLE_CLICK_MAX_GAP:
                        self.mouse.double_click()
                        self.last_action_desc = "Double Click"
                        self.last_click_time = 0.0
                    else:
                        self.mouse.left_click()
                        self.last_action_desc = "Left Click"
                        self.last_click_time = now

                self.is_pinched = False
                self.pinch_start_time = None

            self.state = GestureState.MOVING
            if self.last_action_desc not in ["Left Click", "Double Click", "Dropped", "Right Click"]:
                self.last_action_desc = "Moving Pointer"

        return self.state, self.last_action_desc
