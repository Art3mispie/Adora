"""Turn a smooth race trajectory into a dynamically feasible control queue."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .math3d import attitude_from_down_and_heading, body_rate_between, exp_so3, log_so3
from .trajectory import (SampledTrajectory, TrajectoryLimits,
                         build_kinematic_trajectory)
from controller.state import FeasibilityReport, FlatReference, NominalControl
from .model import VehicleModel


@dataclass(frozen=True)
class DynamicLimits:
    maximum_tilt_rad: float = math.radians(55.0)
    maximum_body_rate_rps: tuple[float, float, float] = (3.5, 3.5, 2.8)
    maximum_body_acceleration_rps2: tuple[float, float, float] = (22.0, 22.0, 16.0)
    minimum_motor_input: float = 0.06
    maximum_motor_input: float = 0.85
    maximum_iterations: int = 12


@dataclass(frozen=True)
class RacePlan:
    trajectory: SampledTrajectory
    queue: "PrecomputedQueue"
    feasibility: FeasibilityReport


def attitude_and_thrust(reference: FlatReference,
                        vehicle: VehicleModel,
                        acceleration_override=None,
                        velocity_override=None) -> tuple[np.ndarray, float]:
    """Solve attitude/thrust with body-axis drag in the correct frame."""
    gravity = np.array([0.0, 0.0, vehicle.gravity_mps2])
    acceleration = (reference.acceleration_ned if acceleration_override is None
                    else np.asarray(acceleration_override, dtype=float))
    velocity = (reference.velocity_ned if velocity_override is None
                else np.asarray(velocity_override, dtype=float))
    thrust_axis = gravity - acceleration
    rotation = attitude_from_down_and_heading(thrust_axis, reference.yaw)
    # Drag is body-axis, so attitude and drag are solved together.
    for _ in range(2):
        drag = vehicle.drag_acceleration_ned(velocity, rotation)
        thrust_axis = gravity + drag - acceleration
        rotation = attitude_from_down_and_heading(thrust_axis, reference.yaw)
    return rotation, float(np.linalg.norm(thrust_axis))


def _differentiate_attitudes(rotations: list[np.ndarray],
                             times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = len(rotations)
    rates = np.zeros((count, 3))
    for index in range(count - 1):
        rates[index] = body_rate_between(
            rotations[index], rotations[index + 1], times[index + 1] - times[index])
    if count > 1:
        rates[-1] = rates[-2]

    accelerations = np.zeros_like(rates)
    for index in range(count - 1):
        dt = max(float(times[index + 1] - times[index]), 1e-9)
        # Compare both rates in the same body frame.
        next_in_current = rotations[index].T @ rotations[index + 1] @ rates[index + 1]
        accelerations[index] = (next_in_current - rates[index]) / dt
    if count > 1:
        accelerations[-1] = accelerations[-2]
    return rates, accelerations


def compile_controls(references: list[FlatReference], vehicle: VehicleModel,
                     limits: DynamicLimits,
                     initial_rotor_fraction: float | None = None
                     ) -> tuple[list[NominalControl], dict[str, np.ndarray]]:
    if len(references) < 2:
        raise ValueError("at least two reference samples are required")
    times = np.asarray([item.time_s for item in references], dtype=float)
    rotations: list[np.ndarray] = []
    thrust = np.zeros(len(references))
    rotor = np.zeros(len(references))
    thrust_saturated = np.zeros(len(references), dtype=bool)
    for index, reference in enumerate(references):
        rotation, specific = attitude_and_thrust(reference, vehicle)
        rotations.append(rotation)
        thrust[index] = specific
        rotor[index], thrust_saturated[index] = (
            vehicle.rotor_fraction_for_collective_thrust(specific * vehicle.mass_kg))
    rates, accelerations = _differentiate_attitudes(rotations, times)

    # Pre-emphasize the motor input by one physics step.
    motor_input = np.empty(len(references))
    current = (rotor[0] if initial_rotor_fraction is None
               else float(initial_rotor_fraction))
    for index in range(len(references) - 1):
        dt = float(times[index + 1] - times[index])
        motor_input[index] = vehicle.motor_input_for_next_fraction(
            current, rotor[index + 1], dt)
        current = rotor[index + 1]
    motor_input[-1] = rotor[-1]

    max_rate = np.asarray(limits.maximum_body_rate_rps, dtype=float)
    max_accel = np.asarray(limits.maximum_body_acceleration_rps2, dtype=float)
    controls: list[NominalControl] = []
    tilt = np.asarray([math.acos(min(1.0, max(-1.0, float(R[2, 2]))))
                       for R in rotations])
    for index, reference in enumerate(references):
        reasons: list[str] = []
        if tilt[index] > limits.maximum_tilt_rad:
            reasons.append("tilt")
        if np.any(np.abs(rates[index]) > max_rate):
            reasons.append("body_rate")
        if np.any(np.abs(accelerations[index]) > max_accel):
            reasons.append("body_acceleration")
        if motor_input[index] < limits.minimum_motor_input:
            reasons.append("motor_input_low")
        if motor_input[index] > limits.maximum_motor_input:
            reasons.append("motor_input_high")
        if thrust_saturated[index]:
            reasons.append("native_thrust_curve")
        controls.append(NominalControl(
            reference=reference,
            R_nb=rotations[index],
            body_rate_frd=rates[index],
            body_acceleration_frd=accelerations[index],
            specific_thrust_mps2=float(thrust[index]),
            rotor_fraction=float(rotor[index]),
            throttle=float(motor_input[index]),
            feasible=not reasons,
            limiting=tuple(reasons),
        ))
    diagnostics = {
        "tilt": tilt,
        "rates": rates,
        "accelerations": accelerations,
        "motor_input": motor_input,
        "rotor_fraction": rotor,
        "thrust": thrust,
    }
    return controls, diagnostics


def _dynamic_retime_factors(controls: list[NominalControl], diagnostics: dict,
                            segment_count: int, limits: DynamicLimits) -> np.ndarray:
    factors = np.ones(segment_count)
    max_rate = np.asarray(limits.maximum_body_rate_rps, dtype=float)
    max_accel = np.asarray(limits.maximum_body_acceleration_rps2, dtype=float)
    for index, control in enumerate(controls):
        factor = 1.0
        tilt = float(diagnostics["tilt"][index])
        if tilt > limits.maximum_tilt_rad:
            ratio = math.tan(min(tilt, math.radians(85.0))) / max(
                math.tan(limits.maximum_tilt_rad), 1e-6)
            factor = max(factor, math.sqrt(max(1.0, ratio)))
        factor = max(factor, float(np.max(
            np.abs(control.body_rate_frd) / np.maximum(max_rate, 1e-6))))
        accel_ratio = float(np.max(
            np.abs(control.body_acceleration_frd) / np.maximum(max_accel, 1e-6)))
        factor = max(factor, math.sqrt(max(1.0, accel_ratio)))
        if control.throttle > limits.maximum_motor_input:
            # Motor pre-emphasis is driven primarily by jerk (time^-3).
            factor = max(factor, (control.throttle / limits.maximum_motor_input) ** (1 / 3))
        if control.throttle < limits.minimum_motor_input:
            factor = max(factor, 1.08)
        if factor <= 1.0005:
            continue
        segment = control.reference.segment
        factors[segment] = max(factors[segment], factor)
        if segment:
            factors[segment - 1] = max(factors[segment - 1], math.sqrt(factor))
        if segment + 1 < segment_count:
            factors[segment + 1] = max(factors[segment + 1], math.sqrt(factor))
    return factors


def build_race_plan(waypoints_ned, initial_velocity_ned, initial_yaw: float,
                    vehicle: VehicleModel, control_dt: float,
                    trajectory_limits: TrajectoryLimits,
                    dynamic_limits: DynamicLimits | None = None,
                    initial_rotor_fraction: float | None = None,
                    initial_acceleration_ned=None) -> RacePlan:
    """Precompute and locally retime until kinematic and actuator limits hold."""
    limits = dynamic_limits or DynamicLimits()
    points = np.asarray(waypoints_ned, dtype=float)
    trajectory, references, kinematic_iterations = build_kinematic_trajectory(
        points, initial_velocity_ned, control_dt, trajectory_limits, initial_yaw,
        initial_acceleration=initial_acceleration_ned)
    dynamic_iterations = 0
    controls: list[NominalControl] = []
    diagnostics: dict[str, np.ndarray] = {}
    for dynamic_iterations in range(1, limits.maximum_iterations + 1):
        controls, diagnostics = compile_controls(
            references, vehicle, limits, initial_rotor_fraction)
        factors = _dynamic_retime_factors(
            controls, diagnostics, len(trajectory.durations_s), limits)
        if float(np.max(factors)) <= 1.0005:
            break
        # Slow the clock globally so the race line cannot move between iterations.
        trajectory = trajectory.retime(
            min(1.8, 1.01 * float(np.max(factors))))
        references = trajectory.sample(control_dt, initial_yaw)
    else:
        raise RuntimeError("trajectory could not satisfy the native dynamic envelope")

    speed = np.asarray([np.linalg.norm(item.velocity_ned) for item in references])
    accel = np.asarray([np.linalg.norm(item.acceleration_ned) for item in references])
    jerk = np.asarray([np.linalg.norm(item.jerk_ned) for item in references])
    reasons = sorted({reason for item in controls for reason in item.limiting})
    report = FeasibilityReport(
        feasible=not reasons,
        iterations=kinematic_iterations + dynamic_iterations,
        duration_s=trajectory.duration_s,
        max_speed_mps=float(np.max(speed)),
        max_acceleration_mps2=float(np.max(accel)),
        max_jerk_mps3=float(np.max(jerk)),
        max_tilt_rad=float(np.max(diagnostics["tilt"])),
        max_body_rate_rps=np.max(np.abs(diagnostics["rates"]), axis=0),
        max_body_acceleration_rps2=np.max(
            np.abs(diagnostics["accelerations"]), axis=0),
        max_thrust_ratio=float(np.max(diagnostics["motor_input"])
                               / limits.maximum_motor_input),
        violations=tuple(reasons),
    )
    return RacePlan(trajectory, PrecomputedQueue(controls), report)


class PrecomputedQueue:
    """Time-interpolated immutable control queue."""

    def __init__(self, controls: list[NominalControl]):
        if len(controls) < 2:
            raise ValueError("control queue requires at least two samples")
        self.controls = tuple(controls)
        self.times = np.asarray([item.reference.time_s for item in controls], dtype=float)
        if np.any(np.diff(self.times) <= 0.0):
            raise ValueError("queue times must be strictly increasing")

    @property
    def duration_s(self) -> float:
        return float(self.times[-1])

    def __len__(self) -> int:
        return len(self.controls)

    @staticmethod
    def _vector(a, b, u: float) -> np.ndarray:
        return (1.0 - u) * np.asarray(a) + u * np.asarray(b)

    def sample(self, phase_s: float) -> NominalControl:
        t = min(self.duration_s, max(0.0, float(phase_s)))
        right = int(np.searchsorted(self.times, t, side="right"))
        if right == 0:
            return self.controls[0]
        if right >= len(self.controls):
            return self.controls[-1]
        left = right - 1
        a, b = self.controls[left], self.controls[right]
        u = (t - self.times[left]) / (self.times[right] - self.times[left])
        yaw_delta = (b.reference.yaw - a.reference.yaw + math.pi) % (2 * math.pi) - math.pi
        reference = FlatReference(
            time_s=t,
            segment=a.reference.segment,
            position_ned=self._vector(a.reference.position_ned, b.reference.position_ned, u),
            velocity_ned=self._vector(a.reference.velocity_ned, b.reference.velocity_ned, u),
            acceleration_ned=self._vector(
                a.reference.acceleration_ned, b.reference.acceleration_ned, u),
            jerk_ned=self._vector(a.reference.jerk_ned, b.reference.jerk_ned, u),
            snap_ned=self._vector(a.reference.snap_ned, b.reference.snap_ned, u),
            yaw=a.reference.yaw + u * yaw_delta,
        )
        rotation = a.R_nb @ exp_so3(u * log_so3(a.R_nb.T @ b.R_nb))
        limiting = tuple(sorted(set(a.limiting + b.limiting)))
        return NominalControl(
            reference=reference,
            R_nb=rotation,
            body_rate_frd=self._vector(a.body_rate_frd, b.body_rate_frd, u),
            body_acceleration_frd=self._vector(
                a.body_acceleration_frd, b.body_acceleration_frd, u),
            specific_thrust_mps2=float((1.0 - u) * a.specific_thrust_mps2
                                       + u * b.specific_thrust_mps2),
            rotor_fraction=float((1.0 - u) * a.rotor_fraction + u * b.rotor_fraction),
            throttle=float((1.0 - u) * a.throttle + u * b.throttle),
            feasible=a.feasible and b.feasible,
            limiting=limiting,
        )
