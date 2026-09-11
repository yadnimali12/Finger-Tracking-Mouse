"""
Adaptive Smoothing Filter Module.
Provides dynamic low-pass filtering and velocity-dependent smoothing to eliminate
hand tremor/camera jitter while maintaining zero-lag responsiveness during fast motions.
"""

import math
from typing import Tuple, Optional


class AdaptiveSmoothingFilter:
    """
    Velocity-aware dynamic low-pass filter (2D coordinate smoother).
    Adjusts alpha smoothing weight in real-time based on finger movement speed.
    """

    def __init__(
        self,
        alpha_min: float = 0.18,
        alpha_max: float = 0.82,
        velocity_scale: float = 30.0,
        deadband: float = 0.8
    ):
        """
        :param alpha_min: Minimum smoothing factor (applied when finger is still / micro-motion).
        :param alpha_max: Maximum smoothing factor (applied during fast sweeps).
        :param velocity_scale: Speed in pixels/frame at which filter reaches max responsiveness.
        :param deadband: Displacement in pixels below which coordinate updates are dampened.
        """
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.velocity_scale = velocity_scale
        self.deadband = deadband

        self.filtered_x: Optional[float] = None
        self.filtered_y: Optional[float] = None
        self.prev_raw_x: Optional[float] = None
        self.prev_raw_y: Optional[float] = None
        self.current_alpha: float = alpha_min

    def reset(self):
        """Reset internal filter state."""
        self.filtered_x = None
        self.filtered_y = None
        self.prev_raw_x = None
        self.prev_raw_y = None
        self.current_alpha = self.alpha_min

    def update_profile(self, alpha_min: float, alpha_max: float, velocity_scale: float):
        """Dynamically update smoothing parameters from preset."""
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.velocity_scale = velocity_scale

    def step(self, raw_x: float, raw_y: float) -> Tuple[float, float]:
        """
        Process a new raw coordinate pair and return the smoothed coordinates.

        :param raw_x: Raw input X coordinate.
        :param raw_y: Raw input Y coordinate.
        :return: (smoothed_x, smoothed_y) as floats.
        """
        if self.filtered_x is None or self.filtered_y is None:
            self.filtered_x = float(raw_x)
            self.filtered_y = float(raw_y)
            self.prev_raw_x = float(raw_x)
            self.prev_raw_y = float(raw_y)
            return self.filtered_x, self.filtered_y

        # Calculate raw displacement from previous filtered point
        dx = raw_x - self.filtered_x
        dy = raw_y - self.filtered_y
        dist = math.hypot(dx, dy)

        # Apply deadband for microscopic tremors
        if dist < self.deadband:
            return self.filtered_x, self.filtered_y

        # Compute dynamic alpha using non-linear velocity curve
        normalized_speed = min(1.0, dist / max(1.0, self.velocity_scale))
        speed_factor = normalized_speed * normalized_speed  # Quadratic curve for smooth ramp
        self.current_alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * speed_factor

        # Exponential Moving Average update
        self.filtered_x += self.current_alpha * dx
        self.filtered_y += self.current_alpha * dy

        self.prev_raw_x = raw_x
        self.prev_raw_y = raw_y

        return self.filtered_x, self.filtered_y
