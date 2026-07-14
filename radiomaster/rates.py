from dataclasses import dataclass
import math
from pathlib import Path
import struct

import numpy as np

from controller.state import ActuatorCommand
from physics.model import VehicleModel, load_settings


@dataclass(frozen=True)
class RateSettings:
    rc_rate: float
    super_rate: float
    rc_expo: float


def betaflight_rate_rps(stick: float, settings: RateSettings) -> float:
    value = float(stick)
    magnitude = abs(value)
    if settings.rc_expo:
        value = value * (1.0 - settings.rc_expo) + value * magnitude**3 * settings.rc_expo
    rate = settings.rc_rate
    if rate > 2.0:
        rate += 14.54 * (rate - 2.0)
    degrees_per_second = 200.0 * rate * value
    if settings.super_rate:
        degrees_per_second /= min(
            1.0, max(0.01, 1.0 - magnitude * settings.super_rate)
        )
    return math.radians(degrees_per_second)


def inverse_betaflight_rate(
    rate_rps: float, settings: RateSettings, maximum_stick: float = 1.0
) -> tuple[float, float, bool]:
    requested = float(rate_rps)
    sign = -1.0 if requested < 0.0 else 1.0
    target = abs(requested)
    if target == 0.0:
        return 0.0, 0.0, False
    maximum = abs(betaflight_rate_rps(maximum_stick, settings))
    if target >= maximum:
        return sign * maximum_stick, sign * maximum, target > maximum
    lo, hi = 0.0, float(maximum_stick)
    for _ in range(55):
        mid = (lo + hi) / 2.0
        if abs(betaflight_rate_rps(mid, settings)) < target:
            lo = mid
        else:
            hi = mid
    stick = sign * (lo + hi) / 2.0
    return stick, betaflight_rate_rps(stick, settings), False


@dataclass(frozen=True)
class RateInterface:
    roll: RateSettings
    pitch: RateSettings
    yaw: RateSettings
    profile_index: int
    maximum_stick: float = 1.0

    @property
    def maximum_body_rate_rps(self) -> np.ndarray:
        return np.asarray(
            [
                abs(betaflight_rate_rps(self.maximum_stick, axis))
                for axis in (self.roll, self.pitch, self.yaw)
            ]
        )

    def encode(self, desired_body_rate_frd, throttle: float) -> ActuatorCommand:
        requested = np.asarray(desired_body_rate_frd, dtype=float).reshape(3)
        fields = np.zeros(3)
        achieved = np.zeros(3)
        reasons = []
        for index, (value, settings, axis) in enumerate(
            zip(requested, (self.roll, self.pitch, self.yaw), ("roll", "pitch", "yaw"))
        ):
            fields[index], achieved[index], saturated = inverse_betaflight_rate(
                value, settings, self.maximum_stick
            )
            if saturated:
                reasons.append(axis)
        bounded_throttle = float(np.clip(throttle, 0.0, 1.0))
        if bounded_throttle != throttle:
            reasons.append("throttle")
        return ActuatorCommand(
            fields,
            bounded_throttle,
            requested,
            achieved,
            bool(reasons),
            tuple(reasons),
        )


def load_rates() -> RateInterface:
    data = load_settings()["actuator"]
    return RateInterface(
        RateSettings(*data["roll"]),
        RateSettings(*data["pitch"]),
        RateSettings(*data["yaw"]),
        int(data["acro_profile"]),
        float(data["maximum_stick"]),
    )


def validate_simulator_save(
    vehicle: VehicleModel, rates: RateInterface, save_path: str | Path | None = None
) -> tuple[bool, str]:
    if save_path is None:
        save_path = (
            Path.home()
            / "AppData/Local/FlightSim/Saved/SaveGames/DCLSave-LocalPlayer.sav"
        )
    path = Path(save_path)
    if not path.is_file():
        return False, f"simulator save not found: {path}"
    raw = path.read_bytes()
    offsets = load_settings()["save_validation"]
    body_offset = int(offsets["body_selection_offset"], 16)
    controller_offset = int(offsets["controller_settings_offset"], 16)
    if raw[:4] != b"GVAS" or len(raw) < controller_offset + 8 + 0x54:
        return False, "simulator save layout does not match this controller"
    if raw[body_offset] != vehicle.native_enum_value:
        return False, f"select the {vehicle.profile_name} vehicle in the simulator"
    if raw[controller_offset] != 0:
        return False, "select ACRO flight mode in the simulator"
    selected = struct.unpack_from("<i", raw, controller_offset + 4)[0]
    if selected != rates.profile_index:
        return False, f"select ACRO profile {rates.profile_index} in the simulator"
    block = controller_offset + 8
    for offset, actual_settings in zip(
        (0x30, 0x3C, 0x48), (rates.roll, rates.pitch, rates.yaw)
    ):
        actual = struct.unpack_from("<fff", raw, block + offset)
        wanted = (
            actual_settings.rc_rate,
            actual_settings.super_rate,
            actual_settings.rc_expo,
        )
        if not np.allclose(actual, wanted, rtol=0.0, atol=2e-6):
            return False, "simulator ACRO rates differ from controller settings"
    return True, f"{vehicle.profile_name} / ACRO {rates.profile_index} verified"
