"""Numpy replicas of the exact IsaacLab math helpers used by the obs terms.

Faithful to isaaclab/utils/math.py so obs reconstruction in the MuJoCo env
(no isaaclab dependency) is bit-comparable. All quats are (w, x, y, z).
"""
import numpy as np


def quat_mul(q1, q2):
    """Hamilton product, (w,x,y,z). q1,q2: (...,4)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def quat_conjugate(q):
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_inv(q, eps=1e-9):
    return quat_conjugate(q) / np.clip((q ** 2).sum(-1, keepdims=True), eps, None)


def quat_apply(q, v):
    """Rotate vector v (...,3) by quaternion q (...,4), (w,x,y,z)."""
    qv = q[..., 1:4]
    w = q[..., 0:1]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def axis_angle_from_quat(quat, eps=1.0e-6):
    """(w,x,y,z) -> axis-angle 3-vector (axis * angle). Matches IsaacLab exactly."""
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))  # canonicalize w >= 0
    mag = np.linalg.norm(quat[..., 1:], axis=-1)
    half_angle = np.arctan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    small = np.abs(angle) <= eps
    sin_half_over_angle = np.where(small, 0.5 - angle * angle / 48.0,
                                   np.sin(half_angle) / np.where(small, 1.0, angle))
    return quat[..., 1:4] / sin_half_over_angle[..., None]


def subtract_frame_transforms(t01, q01, t02, q02):
    """Pose of frame 2 w.r.t. frame 1, given both w.r.t. frame 0.

    Returns (t12, q12). Matches isaaclab.utils.math.subtract_frame_transforms.
    """
    q10 = quat_inv(q01)
    q12 = quat_mul(q10, q02)
    t12 = quat_apply(q10, t02 - t01)
    return t12, q12


def combine_frame_transforms(t01, q01, t12, q12):
    """Pose of frame 2 w.r.t. frame 0, given 1-in-0 and 2-in-1. Returns (t02, q02)."""
    q02 = quat_mul(q01, q12)
    t02 = t01 + quat_apply(q01, t12)
    return t02, q02


def pose_in_root(root_pos, root_quat, tgt_pos, tgt_quat):
    """[pos(3), axis_angle(3)] of target in root frame — i.e. one obs pose term."""
    t12, q12 = subtract_frame_transforms(root_pos, root_quat, tgt_pos, tgt_quat)
    aa = axis_angle_from_quat(q12)
    return np.concatenate([t12, aa], axis=-1)
