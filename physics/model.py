from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np


SETTINGS_PATH = Path(__file__).resolve().parents[1] / "data" / "physics.json"


@lru_cache(maxsize=1)
def load_settings() -> dict:
    data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported physics settings")
    return data


def _vector(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(3).copy()
    if not np.isfinite(result).all():
        raise ValueError(f"non-finite {name}")
    return result


@dataclass(frozen=True)
class VehicleModel:
    profile_name: str
    native_enum_value: int
    mass_kg: float
    gravity_mps2: float
    velocity_drag_coefficients: np.ndarray
    drag_reference_area_m2: float
    thrust_z0_factor: float
    motor_reactivity: float
    motor_time_scale_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "velocity_drag_coefficients",
            _vector(self.velocity_drag_coefficients, "velocity drag"),
        )
        if min(
            self.mass_kg,
            self.gravity_mps2,
            self.drag_reference_area_m2,
            self.thrust_z0_factor,
            self.motor_reactivity,
            self.motor_time_scale_s,
        ) <= 0.0:
            raise ValueError("invalid vehicle settings")

    @staticmethod
    def _thrust_curve(rotor_fraction) -> np.ndarray:
        rotor = np.asarray(rotor_fraction, dtype=float)
        return 16.06 * np.sin(0.6593 * rotor - 0.01853) + np.sin(
            5.436 * rotor + 2.962
        )

    def collective_thrust_n(self, rotor_fraction) -> np.ndarray:
        return 4.0 * self.thrust_z0_factor * self._thrust_curve(rotor_fraction)

    def rotor_fraction_for_collective_thrust(
        self, thrust_n: float
    ) -> tuple[float, bool]:
        requested = float(thrust_n)
        low = float(self.collective_thrust_n(0.0))
        high = float(self.collective_thrust_n(1.0))
        if requested <= low:
            return 0.0, requested < low
        if requested >= high:
            return 1.0, requested > high
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if float(self.collective_thrust_n(mid)) < requested:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0, False

    @property
    def hover_rotor_fraction(self) -> float:
        value, _ = self.rotor_fraction_for_collective_thrust(
            self.mass_kg * self.gravity_mps2
        )
        return value

    def motor_lag_alpha(self, dt: float) -> float:
        return self.motor_reactivity * float(dt) / self.motor_time_scale_s

    def motor_input_for_next_fraction(
        self, current_fraction: float, next_fraction: float, dt: float
    ) -> float:
        alpha = self.motor_lag_alpha(dt)
        if alpha <= 1e-9:
            raise ValueError("motor-lag inverse requires a positive dt")
        return float(current_fraction) + (
            float(next_fraction) - float(current_fraction)
        ) / alpha

    def drag_acceleration_ned(self, velocity_ned, rotation_ned_from_body) -> np.ndarray:
        rotation = np.asarray(rotation_ned_from_body, dtype=float).reshape(3, 3)
        body_velocity = rotation.T @ _vector(velocity_ned, "NED velocity")
        temperature_k = 288.15
        scale = 176.48871 / temperature_k * self.drag_reference_area_m2
        force = -np.sign(body_velocity) * (
            scale * self.velocity_drag_coefficients * np.abs(body_velocity) ** 2
        )
        return rotation @ force / self.mass_kg


def load_vehicle() -> VehicleModel:
    return VehicleModel(**load_settings()["vehicle"])
