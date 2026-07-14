"""SO(3) helpers using NED world and FRD body frames."""
from __future__ import annotations

import math

import numpy as np


def clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, float(value)))


def clamp_norm(vector, limit: float) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm <= float(limit) or norm <= 1e-15:
        return value.copy()
    return value * (float(limit) / norm)


def unit(vector, fallback=None) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm > 1e-12:
        return value / norm
    if fallback is None:
        raise ValueError("cannot normalise a zero vector")
    return unit(fallback)


def hat(vector) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee(matrix) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    return np.array([value[2, 1], value[0, 2], value[1, 0]])


def exp_so3(rotation_vector) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=float)
    theta = float(np.linalg.norm(vector))
    K = hat(vector)
    if theta < 1e-7:
        return np.eye(3) + K + 0.5 * (K @ K)
    a = math.sin(theta) / theta
    b = (1.0 - math.cos(theta)) / (theta * theta)
    return np.eye(3) + a * K + b * (K @ K)


def log_so3(rotation) -> np.ndarray:
    """Return the shortest rotation vector represented by ``rotation``.

    The half-turn branch avoids the zero-error singularity of the common
    skew-only attitude error at exactly 180 degrees.
    """
    R = np.asarray(rotation, dtype=float)
    cosine = clamp(0.5 * (float(np.trace(R)) - 1.0), -1.0, 1.0)
    angle = math.acos(cosine)
    if angle < 1e-7:
        return 0.5 * vee(R - R.T)
    if math.pi - angle < 1e-5:
        A = 0.5 * (R + np.eye(3))
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        dominant = int(np.argmax(axis))
        if dominant == 0:
            axis[1] = math.copysign(axis[1], R[0, 1] + R[1, 0])
            axis[2] = math.copysign(axis[2], R[0, 2] + R[2, 0])
        elif dominant == 1:
            axis[2] = math.copysign(axis[2], R[1, 2] + R[2, 1])
        return angle * unit(axis, [1.0, 0.0, 0.0])
    return angle * vee(R - R.T) / (2.0 * math.sin(angle))


def project_so3(rotation) -> np.ndarray:
    """Project numerical integration drift onto the nearest proper rotation."""
    U, _, Vt = np.linalg.svd(np.asarray(rotation, dtype=float))
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def wrap_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def attitude_from_down_and_heading(body_down_ned, yaw: float) -> np.ndarray:
    """Construct ``R_nb`` from desired body-down thrust axis and heading."""
    b3 = unit(body_down_ned, [0.0, 0.0, 1.0])
    heading = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    b2 = np.cross(b3, heading)
    if float(np.linalg.norm(b2)) < 1e-7:
    # Keep a horizontal basis when the requested heading is vertical.
        seed = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(seed, b3))) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        b2 = np.cross(b3, seed)
    b2 = unit(b2)
    b1 = unit(np.cross(b2, b3))
    return np.column_stack((b1, b2, b3))


def body_rate_between(R0, R1, dt: float) -> np.ndarray:
    return log_so3(np.asarray(R0).T @ np.asarray(R1)) / max(float(dt), 1e-9)


def yaw_of(R_nb) -> float:
    R = np.asarray(R_nb, dtype=float)
    return math.atan2(float(R[1, 0]), float(R[0, 0]))
