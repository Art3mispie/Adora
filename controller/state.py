"""Shared flight data."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _vector3(value) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(3).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite 3-vector")
    return result


def _rotation(value) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(3, 3).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite attitude")
    return result


@dataclass(frozen=True)
class StateEstimate:
    time_s: float
    position_ned: np.ndarray
    velocity_ned: np.ndarray
    R_nb: np.ndarray
    body_rate_frd: np.ndarray
    imu_age_s: float = 0.0
    position_sigma_m: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "position_ned", _vector3(self.position_ned))
        object.__setattr__(self, "velocity_ned", _vector3(self.velocity_ned))
        object.__setattr__(self, "R_nb", _rotation(self.R_nb))
        object.__setattr__(self, "body_rate_frd", _vector3(self.body_rate_frd))


@dataclass(frozen=True)
class FlatReference:
    time_s: float
    segment: int
    position_ned: np.ndarray
    velocity_ned: np.ndarray
    acceleration_ned: np.ndarray
    jerk_ned: np.ndarray
    snap_ned: np.ndarray
    yaw: float

    def __post_init__(self):
        for name in ("position_ned", "velocity_ned", "acceleration_ned",
                     "jerk_ned", "snap_ned"):
            object.__setattr__(self, name, _vector3(getattr(self, name)))


@dataclass(frozen=True)
class NominalControl:
    reference: FlatReference
    R_nb: np.ndarray
    body_rate_frd: np.ndarray
    body_acceleration_frd: np.ndarray
    specific_thrust_mps2: float
    rotor_fraction: float
    throttle: float
    feasible: bool = True
    limiting: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "R_nb", _rotation(self.R_nb))
        object.__setattr__(self, "body_rate_frd", _vector3(self.body_rate_frd))
        object.__setattr__(self, "body_acceleration_frd",
                           _vector3(self.body_acceleration_frd))


@dataclass(frozen=True)
class ActuatorCommand:
    body_fields: np.ndarray
    throttle: float
    desired_body_rate_frd: np.ndarray
    achieved_body_rate_frd: np.ndarray
    saturated: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "body_fields", _vector3(self.body_fields))
        object.__setattr__(self, "desired_body_rate_frd",
                           _vector3(self.desired_body_rate_frd))
        object.__setattr__(self, "achieved_body_rate_frd",
                           _vector3(self.achieved_body_rate_frd))


@dataclass(frozen=True)
class FeasibilityReport:
    feasible: bool
    iterations: int
    duration_s: float
    max_speed_mps: float
    max_acceleration_mps2: float
    max_jerk_mps3: float
    max_tilt_rad: float
    max_body_rate_rps: np.ndarray
    max_body_acceleration_rps2: np.ndarray
    max_thrust_ratio: float
    violations: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(self, "max_body_rate_rps", _vector3(self.max_body_rate_rps))
        object.__setattr__(self, "max_body_acceleration_rps2",
                           _vector3(self.max_body_acceleration_rps2))
