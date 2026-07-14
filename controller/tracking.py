"""Physics feed-forward with bounded state feedback."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from physics.math3d import clamp_norm, log_so3
from physics.model import VehicleModel
from physics.precompute import attitude_and_thrust
from radiomaster.rates import RateInterface
from .state import ActuatorCommand, NominalControl, StateEstimate


@dataclass(frozen=True)
class ControllerSettings:
    natural_frequency_xy: float = 1.45
    natural_frequency_z: float = 1.60
    damping_ratio_xy: float = 0.95
    damping_ratio_z: float = 1.00
    integral_gain_ned: tuple[float, float, float] = (0.06, 0.06, 0.30)
    integral_limit_ned: tuple[float, float, float] = (2.0, 2.0, 4.0)
    maximum_feedback_xy_mps2: float = 6.0
    maximum_feedback_z_mps2: float = 5.0
    attitude_gain_frd: tuple[float, float, float] = (5.0, 5.0, 3.0)
    maximum_attitude_correction_rps: tuple[float, float, float] = (1.5, 1.5, 1.0)
    maximum_feedback_rate_slew_rps2: tuple[float, float, float] = (6.0, 6.0, 4.0)
    maximum_feedback_throttle_slew_per_s: float = 1.5
    minimum_motor_input: float = 0.06
    maximum_motor_input: float = 0.85
    integral_leak_per_s: float = 0.02


@dataclass(frozen=True)
class ControlDiagnostics:
    position_error_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_error_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    feedback_acceleration_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude_error_frd: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tracking_error_m: float = 0.0
    desired_tilt_rad: float = 0.0
    desired_rotor_fraction: float = 0.0
    nominal_throttle: float = 0.0
    applied_throttle: float = 0.0


class PhysicsTrackingController:
    def __init__(self, vehicle: VehicleModel,
                 actuator: RateInterface,
                 settings: ControllerSettings | None = None):
        self.vehicle = vehicle
        self.actuator = actuator
        self.settings = settings or ControllerSettings()
        self.reset()

    def reset(self) -> None:
        self.integral_error_ned = np.zeros(3)
        self.rate_correction_frd = np.zeros(3)
        self.throttle_correction = 0.0
        self.last_command: ActuatorCommand | None = None
        self.diagnostics = ControlDiagnostics()

    def _feedback_acceleration(self, position_error: np.ndarray,
                               velocity_error: np.ndarray, dt: float) -> np.ndarray:
        cfg = self.settings
        previous = self.integral_error_ned.copy()
        step = max(0.0, float(dt))
        self.integral_error_ned *= max(0.0, 1.0 - cfg.integral_leak_per_s * step)
        self.integral_error_ned += position_error * step
        limit = np.asarray(cfg.integral_limit_ned, dtype=float)
        self.integral_error_ned = np.clip(self.integral_error_ned, -limit, limit)

        wn = np.array([cfg.natural_frequency_xy, cfg.natural_frequency_xy,
                       cfg.natural_frequency_z])
        zeta = np.array([cfg.damping_ratio_xy, cfg.damping_ratio_xy,
                         cfg.damping_ratio_z])
        feedback = (wn * wn * position_error
                    + 2.0 * zeta * wn * velocity_error
                    + np.asarray(cfg.integral_gain_ned)
                    * self.integral_error_ned)
        feedback[:2] = clamp_norm(feedback[:2], cfg.maximum_feedback_xy_mps2)
        feedback[2] = np.clip(
            feedback[2], -cfg.maximum_feedback_z_mps2,
            cfg.maximum_feedback_z_mps2)
        if self.last_command is not None and self.last_command.saturated:
            self.integral_error_ned = previous
        return feedback

    def update(self, state: StateEstimate, nominal: NominalControl,
               dt: float) -> ActuatorCommand:
        cfg = self.settings
        position_error = nominal.reference.position_ned - state.position_ned
        velocity_error = nominal.reference.velocity_ned - state.velocity_ned
        feedback = self._feedback_acceleration(position_error, velocity_error, dt)
        requested_acceleration = nominal.reference.acceleration_ned + feedback
        desired_R, specific_thrust = attitude_and_thrust(
            nominal.reference, self.vehicle,
            acceleration_override=requested_acceleration,
            velocity_override=state.velocity_ned)

        feedforward_world = nominal.R_nb @ nominal.body_rate_frd
        feedforward_current = state.R_nb.T @ feedforward_world
        attitude_error = log_so3(state.R_nb.T @ desired_R)
        raw_correction = np.asarray(cfg.attitude_gain_frd) * attitude_error
        raw_correction = np.clip(
            raw_correction,
            -np.asarray(cfg.maximum_attitude_correction_rps),
            np.asarray(cfg.maximum_attitude_correction_rps))
        step = max(float(dt), 1e-5)
        rate_step = np.asarray(cfg.maximum_feedback_rate_slew_rps2) * step
        self.rate_correction_frd += np.clip(
            raw_correction - self.rate_correction_frd, -rate_step, rate_step)
        requested_rate = feedforward_current + self.rate_correction_frd

        desired_rotor, thrust_curve_saturated = (
            self.vehicle.rotor_fraction_for_collective_thrust(
                specific_thrust * self.vehicle.mass_kg))
        # Apply feedback through the same motor-lag inverse as the plan.
        alpha = max(self.vehicle.motor_lag_alpha(step), 0.05)
        raw_throttle_correction = (
            desired_rotor - nominal.rotor_fraction) / alpha
        throttle_step = cfg.maximum_feedback_throttle_slew_per_s * step
        self.throttle_correction += float(np.clip(
            raw_throttle_correction - self.throttle_correction,
            -throttle_step, throttle_step))
        requested_throttle = nominal.throttle + self.throttle_correction
        applied_throttle = float(np.clip(
            requested_throttle, cfg.minimum_motor_input, cfg.maximum_motor_input))
        command = self.actuator.encode(requested_rate, applied_throttle)
        reasons = list(command.reasons)
        if applied_throttle != requested_throttle:
            reasons.append("racing_collective_envelope")
        if thrust_curve_saturated:
            reasons.append("native_thrust_curve")
        if reasons != list(command.reasons):
            command = ActuatorCommand(
                body_fields=command.body_fields,
                throttle=command.throttle,
                desired_body_rate_frd=command.desired_body_rate_frd,
                achieved_body_rate_frd=command.achieved_body_rate_frd,
                saturated=True,
                reasons=tuple(reasons),
            )
        self.last_command = command
        self.diagnostics = ControlDiagnostics(
            position_error_ned=position_error.copy(),
            velocity_error_ned=velocity_error.copy(),
            feedback_acceleration_ned=feedback.copy(),
            attitude_error_frd=attitude_error.copy(),
            tracking_error_m=float(np.linalg.norm(position_error)),
            desired_tilt_rad=math.acos(
                min(1.0, max(-1.0, float(desired_R[2, 2])))),
            desired_rotor_fraction=desired_rotor,
            nominal_throttle=nominal.throttle,
            applied_throttle=command.throttle,
        )
        return command
