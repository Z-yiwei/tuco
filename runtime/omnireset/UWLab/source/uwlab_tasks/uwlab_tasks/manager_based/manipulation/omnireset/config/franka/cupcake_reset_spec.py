"""Pure-Python pose contract for the CupCake side-lying task-0 reset."""

from __future__ import annotations

import math


RESET_TYPE = "CupCakeSideLyingFront3cm"

# Both objects stay on the robot centerline within a 3 cm x/y jitter box.
# Their x centers are separated enough to keep the 30 cm plate and the
# side-lying 10 cm CupCake from intersecting at the range boundaries.
CUPCAKE_POSE_RANGE = {
    "x": (0.29, 0.35),
    "y": (-0.03, 0.03),
    "z": (0.047, 0.047),
    "roll": (math.pi / 2.0, math.pi / 2.0),
    "pitch": (0.0, 0.0),
    "yaw": (-math.pi / 4.0, math.pi / 4.0),
}

PLATE_POSE_RANGE = {
    "x": (0.57, 0.63),
    "y": (-0.03, 0.03),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> tuple[float, ...]:
    """Return the IsaacLab-compatible ``(w, x, y, z)`` Euler quaternion."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )
