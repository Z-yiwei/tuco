"""Pose contract for the CupCake side-lying reset used by evaluation."""

from __future__ import annotations

import math


RESET_TYPE = "CupCakeSideLyingFront3cm"

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
    """Return the IsaacLab-compatible quaternion in ``(w, x, y, z)`` order."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )
