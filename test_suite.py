"""
Automated Test Suite for Finger Tracking Mouse System.
Validates filter mathematics, coordinate mapping, gesture calculation, and hand tracker pipeline.
"""

import unittest
import numpy as np

from filter import AdaptiveSmoothingFilter
from mouse_controller import MouseController
from hand_tracker import HandData, HandTracker
from gesture_controller import GestureState


class TestAdaptiveSmoothingFilter(unittest.TestCase):

    def test_initial_point(self):
        f = AdaptiveSmoothingFilter(alpha_min=0.2, alpha_max=0.8, velocity_scale=30.0)
        x, y = f.step(100.0, 200.0)
        self.assertEqual(x, 100.0)
        self.assertEqual(y, 200.0)

    def test_micro_tremor_deadband(self):
        f = AdaptiveSmoothingFilter(alpha_min=0.2, alpha_max=0.8, deadband=1.0)
        f.step(100.0, 100.0)
        # Move by 0.5 pixels (below deadband)
        x2, y2 = f.step(100.4, 100.3)
        self.assertEqual(x2, 100.0)
        self.assertEqual(y2, 100.0)

    def test_dynamic_velocity_scaling(self):
        f = AdaptiveSmoothingFilter(alpha_min=0.1, alpha_max=0.9, velocity_scale=50.0)
        f.step(10.0, 10.0)
        # Fast sweep: jump 100 pixels away
        f.step(110.0, 10.0)
        self.assertAlmostEqual(f.current_alpha, 0.9, places=2)

    def test_reset(self):
        f = AdaptiveSmoothingFilter()
        f.step(50.0, 50.0)
        f.reset()
        self.assertIsNone(f.filtered_x)
        self.assertIsNone(f.filtered_y)


class TestMouseCoordinateMapping(unittest.TestCase):

    def setUp(self):
        self.mouse = MouseController()

    def test_corners_within_active_bounds(self):
        # Center of active zone
        mid_cam_x = (self.mouse.x_min + self.mouse.x_max) / 2
        mid_cam_y = (self.mouse.y_min + self.mouse.y_max) / 2
        sx, sy = self.mouse.map_to_screen(mid_cam_x, mid_cam_y)
        self.assertAlmostEqual(sx, self.mouse.screen_w / 2, delta=5)
        self.assertAlmostEqual(sy, self.mouse.screen_h / 2, delta=5)

    def test_boundary_clamping(self):
        # Extreme negative coordinate
        sx, sy = self.mouse.map_to_screen(-100, -100)
        self.assertEqual(sx, 0)
        self.assertEqual(sy, 0)

        # Extreme coordinate beyond camera
        sx, sy = self.mouse.map_to_screen(2000, 2000)
        self.assertEqual(sx, self.mouse.screen_w - 1)
        self.assertEqual(sy, self.mouse.screen_h - 1)


class TestHandDataCalculations(unittest.TestCase):

    def test_distance_and_midpoint(self):
        pixel_landmarks = [(0, 0)] * 21
        pixel_landmarks[4] = (10, 20)
        pixel_landmarks[8] = (40, 60)
        norm_landmarks = [(0.0, 0.0, 0.0)] * 21

        hand = HandData(
            pixel_landmarks=pixel_landmarks,
            norm_landmarks=norm_landmarks,
            handedness="Right",
            frame_width=640,
            frame_height=480
        )

        dist = hand.distance_pixels(4, 8)
        self.assertEqual(dist, 50.0)  # 30-40-50 triangle

        mid = hand.midpoint(4, 8)
        self.assertEqual(mid, (25, 40))


class TestHandTrackerPipeline(unittest.TestCase):

    def test_blank_frame_processing(self):
        tracker = HandTracker()
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = tracker.process_frame(blank_frame)
        self.assertIsNone(result)
        tracker.close()


if __name__ == "__main__":
    unittest.main()
