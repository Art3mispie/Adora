from dataclasses import dataclass
import math

import numpy as np

from physics.model import VehicleModel
from physics.precompute import DynamicLimits, RacePlan, build_race_plan
from physics.trajectory import TrajectoryLimits
from radiomaster.rates import RateInterface
from .state import ActuatorCommand, FlatReference, StateEstimate
from .tracking import ControllerSettings, PhysicsTrackingController


@dataclass(frozen=True)
class RaceRoute:
    waypoints_ned: np.ndarray
    gates_ned: np.ndarray

    @property
    def gate_count(self) -> int:
        return len(self.gates_ned)


def build_route(position_ned, gates_ned) -> RaceRoute:
    position = np.asarray(position_ned, dtype=float).reshape(3)
    gates = np.asarray(gates_ned, dtype=float)
    if (
        gates.ndim != 2
        or gates.shape[1] != 3
        or len(gates) == 0
        or not np.isfinite(np.r_[position, gates.ravel()]).all()
    ):
        raise ValueError("route points must be finite")
    remaining = [gate.copy() for gate in gates]
    ordered = []
    cursor = position
    while remaining:
        index = min(
            range(len(remaining)),
            key=lambda item: float(np.linalg.norm(remaining[item] - cursor)),
        )
        cursor = remaining.pop(index)
        ordered.append(cursor.copy())
    gates = np.asarray(ordered)
    direction = gates[-1] - (gates[-2] if len(gates) > 1 else position)
    norm = float(np.linalg.norm(direction))
    if norm < 1.0:
        raise ValueError("final course segment is too short")
    exit_point = gates[-1] + 10.0 * direction / norm
    return RaceRoute(np.vstack((position, gates, exit_point)), gates.copy())


@dataclass(frozen=True)
class PhaseSettings:
    minimum_rate: float = 0.10
    lag_start_m: float = 0.75
    lag_stop_m: float = 6.0
    cross_start_m: float = 1.0
    cross_stop_m: float = 5.0


class PhaseTracker:
    def __init__(self, settings: PhaseSettings) -> None:
        self.settings = settings
        self.reset()

    def reset(self) -> None:
        self.phase_s = 0.0
        self.rate = 0.0
        self.lag_m = 0.0
        self.cross_track_m = 0.0

    @staticmethod
    def _ramp(value: float, start: float, stop: float, minimum: float) -> float:
        if value <= start:
            return 1.0
        if value >= stop:
            return minimum
        u = (value - start) / max(stop - start, 1e-9)
        return 1.0 + u * (minimum - 1.0)

    def advance(
        self,
        dt: float,
        state: StateEstimate,
        reference: FlatReference,
        duration_s: float,
    ) -> float:
        error = reference.position_ned - state.position_ned
        speed = float(np.linalg.norm(reference.velocity_ned))
        if speed > 0.25:
            tangent = reference.velocity_ned / speed
            along = float(error @ tangent)
            self.lag_m = max(0.0, along)
            self.cross_track_m = float(np.linalg.norm(error - along * tangent))
        else:
            self.lag_m = 0.0
            self.cross_track_m = float(np.linalg.norm(error))
        cfg = self.settings
        self.rate = min(
            self._ramp(
                self.lag_m, cfg.lag_start_m, cfg.lag_stop_m, cfg.minimum_rate
            ),
            self._ramp(
                self.cross_track_m,
                cfg.cross_start_m,
                cfg.cross_stop_m,
                cfg.minimum_rate,
            ),
        )
        self.phase_s = min(
            float(duration_s),
            self.phase_s + max(0.0, float(dt)) * self.rate,
        )
        return self.phase_s

    def anchor(self, phase_s: float, duration_s: float) -> None:
        self.phase_s = min(float(duration_s), max(self.phase_s, float(phase_s)))


@dataclass(frozen=True)
class MissionSettings:
    control_hz: float = 120.0
    cruise_speed_mps: float = 10.0
    maximum_acceleration_mps2: float = 7.0
    maximum_jerk_mps3: float = 28.0
    maximum_snap_mps4: float = 130.0
    maximum_tilt_deg: float = 55.0
    motor_input_reserve: float = 0.85
    rate_reserve: float = 0.90
    gate_capture_lookahead_s: float = 0.05
    gate_capture_timeout_s: float = 2.0


@dataclass(frozen=True)
class MissionOutput:
    command: ActuatorCommand
    reference_position_ned: np.ndarray
    phase_s: float
    duration_s: float
    phase_rate: float


class RaceMission:
    def __init__(
        self,
        vehicle: VehicleModel,
        actuator: RateInterface,
        settings: MissionSettings,
        controller_settings: ControllerSettings,
    ) -> None:
        self.vehicle = vehicle
        self.actuator = actuator
        self.settings = settings
        self.controller = PhysicsTrackingController(
            vehicle, actuator, controller_settings
        )
        self.phase = PhaseTracker(PhaseSettings())
        self.reset()

    def reset(self) -> None:
        self.controller.reset()
        self.phase.reset()
        self.plan: RacePlan | None = None
        self.route: RaceRoute | None = None
        self.crossed_count = 0
        self.gate_capture_dwell_s = 0.0

    def build_plan(
        self,
        state: StateEstimate,
        gates_ned,
        initial_rotor_fraction: float | None = None,
    ) -> tuple[RaceRoute, RacePlan]:
        route = build_route(state.position_ned, gates_ned)
        body_down = state.R_nb[:, 2]
        if float(body_down[2]) <= 0.2:
            raise RuntimeError("attitude is outside the planning envelope")
        specific_thrust = self.vehicle.gravity_mps2 / float(body_down[2])
        acceleration = (
            np.array([0.0, 0.0, self.vehicle.gravity_mps2])
            - specific_thrust * body_down
        )
        cfg = self.settings
        trajectory_limits = TrajectoryLimits(
            cruise_speed_mps=cfg.cruise_speed_mps,
            acceleration_mps2=cfg.maximum_acceleration_mps2,
            jerk_mps3=cfg.maximum_jerk_mps3,
            snap_mps4=cfg.maximum_snap_mps4,
            min_segment_s=0.25,
            max_iterations=24,
        )
        dynamic_limits = DynamicLimits(
            maximum_tilt_rad=math.radians(cfg.maximum_tilt_deg),
            maximum_body_rate_rps=tuple(
                cfg.rate_reserve * self.actuator.maximum_body_rate_rps
            ),
            maximum_body_acceleration_rps2=(18.0, 18.0, 12.0),
            minimum_motor_input=0.06,
            maximum_motor_input=cfg.motor_input_reserve,
            maximum_iterations=14,
        )
        plan = build_race_plan(
            route.waypoints_ned,
            np.zeros(3),
            math.atan2(float(state.R_nb[1, 0]), float(state.R_nb[0, 0])),
            self.vehicle,
            1.0 / cfg.control_hz,
            trajectory_limits,
            dynamic_limits,
            initial_rotor_fraction=(
                self.vehicle.hover_rotor_fraction
                if initial_rotor_fraction is None
                else initial_rotor_fraction
            ),
            initial_acceleration_ned=acceleration,
        )
        if not plan.feasibility.feasible:
            raise RuntimeError(
                "race plan violates " + ", ".join(plan.feasibility.violations)
            )
        return route, plan

    def install(self, route: RaceRoute, plan: RacePlan) -> None:
        self.route = route
        self.plan = plan
        self.phase.reset()
        self.crossed_count = 0
        self.gate_capture_dwell_s = 0.0
        self.controller.reset()

    def prepare(self, state: StateEstimate, gates_ned) -> None:
        self.install(*self.build_plan(state, gates_ned))

    def mark_crossed(self) -> None:
        if self.plan is None or self.complete:
            return
        self.crossed_count += 1
        gate_time = float(
            np.sum(self.plan.trajectory.durations_s[: self.crossed_count])
        )
        self.phase.anchor(gate_time, self.plan.queue.duration_s)
        self.gate_capture_dwell_s = 0.0

    def update(self, state: StateEstimate, dt: float) -> MissionOutput:
        if self.plan is None:
            raise RuntimeError("race mission is not prepared")
        sample_time = self.phase.phase_s
        nominal = self.plan.queue.sample(sample_time)
        command = self.controller.update(state, nominal, dt)
        gate_time = float(
            np.sum(self.plan.trajectory.durations_s[: self.crossed_count + 1])
        )
        phase_limit = (
            self.plan.queue.duration_s
            if self.complete
            else min(
                self.plan.queue.duration_s,
                gate_time + self.settings.gate_capture_lookahead_s,
            )
        )
        self.phase.advance(dt, state, nominal.reference, phase_limit)
        if self.phase.phase_s >= phase_limit - 1e-6:
            self.gate_capture_dwell_s += max(0.0, float(dt))
        else:
            self.gate_capture_dwell_s = 0.0
        return MissionOutput(
            command,
            nominal.reference.position_ned.copy(),
            sample_time,
            self.plan.queue.duration_s,
            self.phase.rate,
        )

    @property
    def exhausted(self) -> bool:
        return self.plan is not None and not self.complete and (
            self.gate_capture_dwell_s >= self.settings.gate_capture_timeout_s
        )

    @property
    def complete(self) -> bool:
        return self.route is not None and self.crossed_count >= self.route.gate_count
