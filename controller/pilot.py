from dataclasses import dataclass
import math
import time

import numpy as np

from physics.model import load_settings, load_vehicle
from radiomaster.rates import load_rates, validate_simulator_save
from .estimator import Estimator
from .mission import MissionOutput, MissionSettings, RaceMission
from .state import ActuatorCommand, StateEstimate
from .tracking import ControllerSettings


LAUNCH_RAMP_PER_S = 0.60
LAUNCH_MAX_WAIT_S = 2.0
LIFTOFF_BODY_X_MPS2 = 1.5
LIVE_IMU_BODY_Z_MIN_MPS2 = 0.5 * 9.81


def say(message: str) -> None:
    print(f"[auto] {message}", flush=True)


@dataclass(frozen=True)
class SafetyLimits:
    maximum_tilt_deg: float = 80.0
    maximum_imu_age_s: float = 0.10
    saturation_warning_dwell_s: float = 0.75


class SafetyMonitor:
    def __init__(self, limits: SafetyLimits) -> None:
        self.limits = limits
        self.reset()

    def reset(self) -> None:
        self.saturation_dwell_s = 0.0

    def check(
        self, state: StateEstimate, command: ActuatorCommand, dt: float
    ) -> tuple[bool, str | None, tuple[str, ...]]:
        values = np.r_[
            state.position_ned,
            state.velocity_ned,
            state.R_nb.ravel(),
            state.body_rate_frd,
        ]
        if not np.isfinite(values).all():
            return False, "non-finite state estimate", ()
        if state.imu_age_s > self.limits.maximum_imu_age_s:
            return False, "IMU stream stale", ()
        alignment = float(np.clip(state.R_nb[2, 2], -1.0, 1.0))
        tilt = math.degrees(math.acos(alignment))
        if tilt > self.limits.maximum_tilt_deg:
            return False, f"vehicle flipped ({tilt:.0f} deg)", ()
        step = max(0.0, float(dt))
        if command.saturated:
            self.saturation_dwell_s += step
        else:
            self.saturation_dwell_s = max(0.0, self.saturation_dwell_s - step)
        warnings = (
            ("actuator saturation persistent",)
            if self.saturation_dwell_s > self.limits.saturation_warning_dwell_s
            else ()
        )
        return True, None, warnings


class RacePilot:
    def __init__(self) -> None:
        settings = load_settings()
        self.vehicle = load_vehicle()
        self.actuator = load_rates()
        self.estimator = Estimator()
        self.mission = RaceMission(
            self.vehicle,
            self.actuator,
            MissionSettings(**settings["mission"]),
            ControllerSettings(**settings["controller"]),
        )
        self.safety = SafetyMonitor(SafetyLimits(**settings["safety"]))
        self.reset()

    @property
    def control_hz(self) -> float:
        return self.mission.settings.control_hz

    @property
    def launch_motor_input(self) -> float:
        dynamic, _ = self.vehicle.rotor_fraction_for_collective_thrust(
            1.20 * self.vehicle.mass_kg * self.vehicle.gravity_mps2
        )
        return min(0.45, max(0.315, dynamic))

    def reset(self) -> None:
        self.estimator.reset()
        self.mission.reset()
        self.safety.reset()
        self.mode = "GROUND"
        self.armed = False
        self.throttle = 0.0
        self.launch_wall_s = None
        self.last_command = self.actuator.encode(np.zeros(3), 0.0)
        self.kill_reason = None
        self.warnings = ()
        self.validation = validate_simulator_save(self.vehicle, self.actuator)
        self.status_message = "" if self.validation[0] else self.validation[1]
        self.output = None

    def state(
        self, time_s: float, body_rate_frd, imu_age_s: float = 0.0
    ) -> StateEstimate:
        position_sigma, _ = self.estimator.uncertainty()
        return StateEstimate(
            time_s,
            self.estimator.p,
            self.estimator.v,
            self.estimator.R,
            body_rate_frd,
            max(0.0, float(imu_age_s)),
            position_sigma,
        )

    def try_arm(self, state: StateEstimate, imu_sample, gates_ned) -> tuple[bool, str]:
        if not self.validation[0]:
            return False, self.validation[1]
        if state.imu_age_s > self.safety.limits.maximum_imu_age_s:
            self.status_message = "waiting for IMU"
            return False, "waiting for IMU"
        if not self.estimator.calibrate(imu_sample):
            self.status_message = "spawn IMU is not stationary"
            return False, "spawn IMU is not stationary"
        aligned = self.state(state.time_s, state.body_rate_frd, state.imu_age_s)
        self.mission.prepare(aligned, gates_ned)
        self.armed = True
        self.mode = "LAUNCH"
        self.launch_wall_s = time.monotonic()
        self.status_message = self.validation[1]
        report = self.mission.plan.feasibility
        say(
            f"planned {report.duration_s:.2f} s course at {self.control_hz:.0f} Hz; "
            f"vmax {report.max_speed_mps:.1f} m/s, "
            f"tilt {math.degrees(report.max_tilt_rad):.1f} deg"
        )
        say("armed; launch ramp")
        return True, self.validation[1]

    def zero(self, throttle: float = 0.0) -> ActuatorCommand:
        return self.actuator.encode(np.zeros(3), throttle)

    def kill(self, reason: str = "operator kill") -> ActuatorCommand:
        first = self.mode != "KILL"
        self.mode = "KILL"
        self.armed = False
        self.kill_reason = reason
        self.throttle = 0.0
        self.last_command = self.zero()
        if first:
            say(f"KILL - {reason}")
        return self.last_command

    def update(
        self, state: StateEstimate, imu_acceleration_frd, dt: float
    ) -> ActuatorCommand:
        if self.mode in {"KILL", "FINISHED", "GROUND"}:
            return self.zero()
        if self.mode == "LAUNCH":
            self.throttle = min(
                self.launch_motor_input,
                self.throttle + LAUNCH_RAMP_PER_S * max(0.0, dt),
            )
            acceleration = np.asarray(imu_acceleration_frd, dtype=float)
            airborne = (
                abs(float(acceleration[0])) < LIFTOFF_BODY_X_MPS2
                and abs(float(acceleration[2])) > LIVE_IMU_BODY_Z_MIN_MPS2
            )
            if airborne:
                self.estimator.release()
                self.mode = "RACE"
                self.mission.phase.reset()
                self.mission.controller.reset()
                say(f"airborne at input {self.throttle:.3f}")
            elif (
                self.throttle >= self.launch_motor_input - 1e-6
                and time.monotonic() - float(self.launch_wall_s) > LAUNCH_MAX_WAIT_S
            ):
                return self.kill("stand did not release")
            self.last_command = self.zero(self.throttle)
            return self.last_command

        output: MissionOutput = self.mission.update(state, dt)
        safe, reason, self.warnings = self.safety.check(state, output.command, dt)
        if not safe:
            return self.kill(reason or "safety monitor")
        self.throttle = output.command.throttle
        self.last_command = output.command
        self.output = output
        if self.mission.exhausted:
            return self.kill("trajectory exhausted before gate confirmation")
        return self.last_command

    def mark_crossed(self, gate_ned) -> None:
        if gate_ned is not None:
            self.estimator.anchor_position(gate_ned)
        self.mission.mark_crossed()
        if self.mission.complete:
            self.finish()

    def finish(self) -> None:
        self.mode = "FINISHED"
        self.armed = False
        self.throttle = 0.0
        self.last_command = self.zero()

    def snapshot(self) -> dict:
        output = getattr(self, "output", None)
        plan = self.mission.plan
        diagnostics = self.mission.controller.diagnostics
        if plan is None:
            path = np.empty((0, 3))
        else:
            samples = plan.trajectory.positions_ned
            stride = max(1, int(math.ceil(len(samples) / 500)))
            path = samples[::stride].copy()
            if len(path) and not np.array_equal(path[-1], samples[-1]):
                path = np.vstack((path, samples[-1]))
        return {
            "mode": self.mode,
            "armed": self.armed,
            "position": self.estimator.p.copy(),
            "velocity": self.estimator.v.copy(),
            "speed": self.estimator.speed,
            "attitude": self.estimator.R.copy(),
            "anchor": self.estimator.anchor,
            "throttle": self.throttle,
            "track_error": diagnostics.tracking_error_m,
            "tilt_deg": math.degrees(diagnostics.desired_tilt_rad),
            "rotor_fraction": diagnostics.desired_rotor_fraction,
            "feedback": diagnostics.feedback_acceleration_ned.copy(),
            "phase_s": 0.0 if output is None else output.phase_s,
            "duration_s": 0.0 if output is None else output.duration_s,
            "phase_rate": 0.0 if output is None else output.phase_rate,
            "path": path,
            "target": None if output is None else output.reference_position_ned.copy(),
            "route": (
                np.empty((0, 3))
                if self.mission.route is None
                else self.mission.route.gates_ned.copy()
            ),
            "warnings": self.warnings,
            "message": self.kill_reason or self.status_message,
            "profile": self.vehicle.profile_name,
            "acro_profile": self.actuator.profile_index,
            "control_hz": self.control_hz,
            "plan": None if plan is None else plan.feasibility,
            "planning_gate": None,
        }
