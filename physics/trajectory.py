"""Minimum-snap race line with an independent timing law."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from controller.state import FlatReference


POLY_DEGREE = 7
COEFFICIENTS = POLY_DEGREE + 1
CONTINUITY_ORDER = 3
SNAP_ORDER = 4
YAW_TRACK_NATURAL_FREQUENCY = 3.0
YAW_TRACK_DAMPING = 1.0
YAW_RATE_LIMIT_RPS = 1.5
YAW_ACCELERATION_LIMIT_RPS2 = 4.0
YAW_DIRECTION_SPEED_MPS = 0.10


def _falling_factorial(power: int, order: int) -> float:
    if power < order:
        return 0.0
    value = 1
    for item in range(power - order + 1, power + 1):
        value *= item
    return float(value)


def _basis(u: float, derivative: int, duration: float) -> np.ndarray:
    values = np.zeros(COEFFICIENTS)
    scale = float(duration) ** derivative
    for power in range(derivative, COEFFICIENTS):
        values[power] = (_falling_factorial(power, derivative)
                         * float(u) ** (power - derivative) / scale)
    return values


def _snap_cost(duration: float) -> np.ndarray:
    result = np.zeros((COEFFICIENTS, COEFFICIENTS))
    scale = float(duration) ** (2 * SNAP_ORDER - 1)
    for row in range(SNAP_ORDER, COEFFICIENTS):
        for col in range(SNAP_ORDER, COEFFICIENTS):
            result[row, col] = (_falling_factorial(row, SNAP_ORDER)
                                * _falling_factorial(col, SNAP_ORDER)
                                / (row + col - 2 * SNAP_ORDER + 1) / scale)
    return result


def _constraint_matrix(durations: np.ndarray) -> np.ndarray:
    segments = len(durations)
    rows = []

    # Both sides of an internal waypoint meet exactly at the gate.
    for segment, duration in enumerate(durations):
        for u in (0.0, 1.0):
            row = np.zeros(segments * COEFFICIENTS)
            start = segment * COEFFICIENTS
            row[start:start + COEFFICIENTS] = _basis(u, 0, duration)
            rows.append(row)

    for segment in range(segments - 1):
        for derivative in range(1, CONTINUITY_ORDER + 1):
            row = np.zeros(segments * COEFFICIENTS)
            left = segment * COEFFICIENTS
            right = (segment + 1) * COEFFICIENTS
            row[left:left + COEFFICIENTS] = _basis(
                1.0, derivative, durations[segment])
            row[right:right + COEFFICIENTS] = -_basis(
                0.0, derivative, durations[segment + 1])
            rows.append(row)

    for derivative in range(1, CONTINUITY_ORDER + 1):
        first = np.zeros(segments * COEFFICIENTS)
        first[:COEFFICIENTS] = _basis(0.0, derivative, durations[0])
        rows.append(first)
        last = np.zeros(segments * COEFFICIENTS)
        last[-COEFFICIENTS:] = _basis(1.0, derivative, durations[-1])
        rows.append(last)
    return np.asarray(rows)


def _constraint_values(waypoints: np.ndarray, initial_derivatives: np.ndarray,
                       final_derivatives: np.ndarray) -> np.ndarray:
    values = []
    for segment in range(len(waypoints) - 1):
        values.extend((waypoints[segment], waypoints[segment + 1]))
    zero = np.zeros(3)
    for _ in range((len(waypoints) - 2) * CONTINUITY_ORDER):
        values.append(zero)
    for derivative in range(CONTINUITY_ORDER):
        values.extend((initial_derivatives[derivative], final_derivatives[derivative]))
    return np.asarray(values, dtype=float)


def solve_minimum_snap(waypoints, durations, initial_velocity=None,
                       initial_acceleration=None, initial_jerk=None,
                       final_velocity=None, final_acceleration=None,
                       final_jerk=None) -> np.ndarray:
    """Return coefficients shaped ``(segment, xyz, power)``."""
    points = np.asarray(waypoints, dtype=float)
    times = np.asarray(durations, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("waypoints must be an (N,3) array with N >= 2")
    if len(times) != len(points) - 1 or np.any(times <= 0.0):
        raise ValueError("durations must contain one positive value per segment")

    zero = np.zeros(3)
    initial = np.asarray([
        zero if initial_velocity is None else initial_velocity,
        zero if initial_acceleration is None else initial_acceleration,
        zero if initial_jerk is None else initial_jerk,
    ], dtype=float)
    final = np.asarray([
        zero if final_velocity is None else final_velocity,
        zero if final_acceleration is None else final_acceleration,
        zero if final_jerk is None else final_jerk,
    ], dtype=float)

    variables = len(times) * COEFFICIENTS
    Q = np.zeros((variables, variables))
    for segment, duration in enumerate(times):
        start = segment * COEFFICIENTS
        Q[start:start + COEFFICIENTS, start:start + COEFFICIENTS] = _snap_cost(duration)
    # A scale-relative diagonal keeps the constrained solve well-conditioned.
    q_scale = max(float(np.max(np.abs(Q))), 1.0)
    Q += np.eye(variables) * q_scale * 1e-12
    A = _constraint_matrix(times)
    b = _constraint_values(points, initial, final)
    KKT = np.block([[Q, A.T], [A, np.zeros((len(A), len(A)))]])
    rhs = np.vstack((np.zeros((variables, 3)), b))
    try:
        solution = np.linalg.solve(KKT, rhs)[:variables]
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(KKT, rhs, rcond=1e-12)[0][:variables]
    residual = float(np.max(np.abs(A @ solution - b)))
    if not np.isfinite(residual) or residual > 2e-5:
        raise RuntimeError(f"minimum-snap constraints did not converge: {residual:g}")
    return solution.reshape(len(times), COEFFICIENTS, 3).transpose(0, 2, 1)


@dataclass(frozen=True)
class TrajectoryLimits:
    cruise_speed_mps: float
    acceleration_mps2: float
    jerk_mps3: float
    snap_mps4: float
    min_segment_s: float = 0.20
    max_iterations: int = 20
    launch_duration_s: float = 3.0


@dataclass(frozen=True)
class PolynomialTrajectory:
    waypoints_ned: np.ndarray
    durations_s: np.ndarray
    coefficients: np.ndarray

    @property
    def duration_s(self) -> float:
        return float(np.sum(self.durations_s))

    def evaluate(self, time_s: float, derivative: int = 0) -> tuple[int, np.ndarray]:
        t = min(max(float(time_s), 0.0), self.duration_s)
        edges = np.r_[0.0, np.cumsum(self.durations_s)]
        segment = min(int(np.searchsorted(edges, t, side="right") - 1),
                      len(self.durations_s) - 1)
        local = t - float(edges[segment])
        duration = float(self.durations_s[segment])
        u = min(max(local / duration, 0.0), 1.0)
        value = self.coefficients[segment] @ _basis(u, derivative, duration)
        return segment, value

    def sample(self, dt: float, initial_yaw: float = 0.0) -> list[FlatReference]:
        count = int(math.ceil(self.duration_s / float(dt)))
        times = np.minimum(np.arange(count + 1, dtype=float) * float(dt), self.duration_s)
        # Avoid a duplicate terminal sample on a fractional final step.
        times = np.unique(times)
        values = [[self.evaluate(t, derivative) for derivative in range(5)] for t in times]
        positions = np.asarray([row[0][1] for row in values])
        velocities = np.asarray([row[1][1] for row in values])
        # Track the tangent gradually; atan2(v) is unstable as speed leaves zero.
        yaw = np.empty(len(times))
        heading = float(initial_yaw)
        heading_rate = 0.0
        target = heading
        yaw[0] = heading
        for index in range(1, len(times)):
            direction = velocities[index]
            if float(np.hypot(direction[0], direction[1])) > YAW_DIRECTION_SPEED_MPS:
                candidate = math.atan2(float(direction[1]), float(direction[0]))
                target += (candidate - target + math.pi) % (2.0 * math.pi) - math.pi
            dt = max(float(times[index] - times[index - 1]), 1e-9)
            error = (target - heading + math.pi) % (2.0 * math.pi) - math.pi
            acceleration = (YAW_TRACK_NATURAL_FREQUENCY ** 2 * error
                            - 2.0 * YAW_TRACK_DAMPING
                            * YAW_TRACK_NATURAL_FREQUENCY * heading_rate)
            acceleration = min(YAW_ACCELERATION_LIMIT_RPS2,
                               max(-YAW_ACCELERATION_LIMIT_RPS2, acceleration))
            heading_rate = min(YAW_RATE_LIMIT_RPS,
                               max(-YAW_RATE_LIMIT_RPS, heading_rate + acceleration * dt))
            heading += heading_rate * dt
            yaw[index] = heading
        return [FlatReference(
            float(t), int(values[index][0][0]), positions[index], velocities[index],
            values[index][2][1], values[index][3][1], values[index][4][1],
            float(yaw[index])) for index, t in enumerate(times)]


def _yaw_profile(times: np.ndarray, velocities: np.ndarray,
                 initial_yaw: float) -> np.ndarray:
    """Track the horizontal path tangent with bounded yaw rate/acceleration."""
    yaw = np.empty(len(times))
    heading = float(initial_yaw)
    heading_rate = 0.0
    target = heading
    yaw[0] = heading
    for index in range(1, len(times)):
        direction = velocities[index]
        if float(np.hypot(direction[0], direction[1])) > YAW_DIRECTION_SPEED_MPS:
            candidate = math.atan2(float(direction[1]), float(direction[0]))
            target += (candidate - target + math.pi) % (2.0 * math.pi) - math.pi
        dt = max(float(times[index] - times[index - 1]), 1e-9)
        error = (target - heading + math.pi) % (2.0 * math.pi) - math.pi
        acceleration = (YAW_TRACK_NATURAL_FREQUENCY ** 2 * error
                        - 2.0 * YAW_TRACK_DAMPING
                        * YAW_TRACK_NATURAL_FREQUENCY * heading_rate)
        acceleration = min(YAW_ACCELERATION_LIMIT_RPS2,
                           max(-YAW_ACCELERATION_LIMIT_RPS2, acceleration))
        heading_rate = min(YAW_RATE_LIMIT_RPS,
                           max(-YAW_RATE_LIMIT_RPS,
                               heading_rate + acceleration * dt))
        heading += heading_rate * dt
        yaw[index] = heading
    return yaw


@dataclass(frozen=True)
class SampledTrajectory:
    """Dense trajectory with waypoint-to-waypoint durations."""

    waypoints_ned: np.ndarray
    durations_s: np.ndarray
    times_s: np.ndarray
    positions_ned: np.ndarray
    velocities_ned: np.ndarray
    accelerations_ned: np.ndarray
    jerks_ned: np.ndarray
    snaps_ned: np.ndarray
    initial_yaw: float = 0.0

    @property
    def duration_s(self) -> float:
        return float(self.times_s[-1])

    @property
    def waypoint_times_s(self) -> np.ndarray:
        return np.r_[0.0, np.cumsum(self.durations_s)]

    def _interpolate(self, values: np.ndarray, time_s: float) -> np.ndarray:
        t = min(max(float(time_s), 0.0), self.duration_s)
        return np.asarray([
            np.interp(t, self.times_s, values[:, axis]) for axis in range(3)
        ], dtype=float)

    def evaluate(self, time_s: float, derivative: int = 0) -> tuple[int, np.ndarray]:
        tables = (
            self.positions_ned, self.velocities_ned, self.accelerations_ned,
            self.jerks_ned, self.snaps_ned,
        )
        if derivative < 0 or derivative >= len(tables):
            raise ValueError("sampled trajectory supports derivatives 0 through 4")
        t = min(max(float(time_s), 0.0), self.duration_s)
        segment = min(int(np.searchsorted(self.waypoint_times_s, t,
                                          side="right") - 1),
                      len(self.durations_s) - 1)
        return max(segment, 0), self._interpolate(tables[derivative], t)

    def sample(self, dt: float, initial_yaw: float | None = None) -> list[FlatReference]:
        count = int(math.ceil(self.duration_s / float(dt)))
        times = np.unique(np.minimum(
            np.arange(count + 1, dtype=float) * float(dt), self.duration_s))
        positions = np.stack([self._interpolate(self.positions_ned, t) for t in times])
        velocities = np.stack([self._interpolate(self.velocities_ned, t) for t in times])
        accelerations = np.stack([
            self._interpolate(self.accelerations_ned, t) for t in times])
        jerks = np.stack([self._interpolate(self.jerks_ned, t) for t in times])
        snaps = np.stack([self._interpolate(self.snaps_ned, t) for t in times])
        yaw = _yaw_profile(times, velocities,
                           self.initial_yaw if initial_yaw is None else initial_yaw)
        edges = self.waypoint_times_s
        segments = np.clip(np.searchsorted(edges, times, side="right") - 1,
                           0, len(self.durations_s) - 1)
        return [FlatReference(
            float(t), int(segments[index]), positions[index], velocities[index],
            accelerations[index], jerks[index], snaps[index], float(yaw[index]))
            for index, t in enumerate(times)]

    def retime(self, factor: float) -> "SampledTrajectory":
        """Uniformly slow the clock without changing the spatial race line."""
        scale = max(float(factor), 1.0)
        return SampledTrajectory(
            self.waypoints_ned.copy(), self.durations_s * scale,
            self.times_s * scale, self.positions_ned.copy(),
            self.velocities_ned / scale,
            self.accelerations_ned / scale ** 2,
            self.jerks_ned / scale ** 3,
            self.snaps_ned / scale ** 4,
            self.initial_yaw,
        )


def _launch_polynomial(cruise_speed: float, initial_acceleration: float,
                       duration: float) -> np.ndarray:
    """Coefficients of v(t) with fixed endpoint acceleration and jerk."""
    T = float(duration)
    v0 = 0.0
    a0 = float(initial_acceleration)
    matrix = np.asarray([
        [T ** 3, T ** 4, T ** 5],
        [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
        [6 * T, 12 * T ** 2, 20 * T ** 3],
    ])
    high = np.linalg.solve(matrix, np.asarray([
        float(cruise_speed) - a0 * T, -a0, 0.0]))
    return np.asarray([v0, a0, 0.0, *high], dtype=float)


def _poly_derivative(coefficients: np.ndarray, times: np.ndarray,
                     order: int) -> np.ndarray:
    coeff = np.asarray(coefficients, dtype=float).copy()
    for _ in range(order):
        coeff = np.asarray([power * coeff[power]
                            for power in range(1, len(coeff))], dtype=float)
    return np.polynomial.polynomial.polyval(times, coeff)


def _integral_coefficients(coefficients: np.ndarray) -> np.ndarray:
    result = np.zeros(len(coefficients) + 1)
    result[1:] = np.asarray(coefficients) / np.arange(1, len(result))
    return result


def _spatial_race_line(points: np.ndarray, initial_acceleration) -> tuple[
        PolynomialTrajectory, np.ndarray]:
    """Build C3 corridors that pass straight through each gate."""
    chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(chords < 1e-6):
        raise ValueError("race waypoints must not contain consecutive duplicates")
    tangents = np.zeros_like(points)
    first_direction = points[1] - points[0]
    if initial_acceleration is not None:
        launch = np.asarray(initial_acceleration, dtype=float).reshape(3)
        if float(np.linalg.norm(launch)) > 0.25:
            first_direction = launch
    first_direction = first_direction / np.linalg.norm(first_direction)
    tangents[0] = first_direction
    for index in range(1, len(points) - 1):
        incoming = (points[index] - points[index - 1]) / chords[index - 1]
        outgoing = (points[index + 1] - points[index]) / chords[index]
        bisector = incoming + outgoing
        if float(np.linalg.norm(bisector)) < 1e-6:
            bisector = incoming
        tangents[index] = bisector / np.linalg.norm(bisector)
    final_direction = points[-1] - points[-2]
    final_direction = final_direction / np.linalg.norm(final_direction)
    tangents[-1] = final_direction

    coefficients = np.empty((len(chords), 3, COEFFICIENTS))
    zero = np.zeros(3)
    for segment, duration in enumerate(chords):
        matrix = np.stack(
            [_basis(0.0, derivative, duration) for derivative in range(4)]
            + [_basis(1.0, derivative, duration) for derivative in range(4)])
        values = np.stack((
            points[segment], tangents[segment], zero, zero,
            points[segment + 1], tangents[segment + 1], zero, zero,
        ))
        coefficients[segment] = np.linalg.solve(matrix, values).T
    return PolynomialTrajectory(points.copy(), chords, coefficients), first_direction


def _build_arc_length_trajectory(points: np.ndarray, control_dt: float,
                                 limits: TrajectoryLimits, initial_yaw: float,
                                 initial_acceleration) -> SampledTrajectory:
    spatial, launch_direction = _spatial_race_line(points, initial_acceleration)
    # A monotone lookup maps arc length back to the polynomial coordinate.
    dense_count = max(4000, int(math.ceil(spatial.duration_s / 0.01)) + 1)
    u_dense = np.linspace(0.0, spatial.duration_s, dense_count)
    p_dense = np.stack([spatial.evaluate(u, 0)[1] for u in u_dense])
    s_dense = np.r_[0.0, np.cumsum(np.linalg.norm(
        np.diff(p_dense, axis=0), axis=1))]
    length = float(s_dense[-1])

    cruise = float(limits.cruise_speed_mps)
    launch_duration = float(limits.launch_duration_s)
    if cruise <= 0.0 or launch_duration <= 0.0:
        raise ValueError("cruise speed and launch duration must be positive")
    launch_vector = (np.zeros(3) if initial_acceleration is None
                     else np.asarray(initial_acceleration, dtype=float).reshape(3))
    launch_acceleration = max(0.0, float(launch_direction @ launch_vector))
    launch_coeff = _launch_polynomial(
        cruise, launch_acceleration, launch_duration)
    launch_integral = _integral_coefficients(launch_coeff)
    launch_distance = float(_poly_derivative(
        launch_integral, np.asarray([launch_duration]), 0)[0])
    launch_grid = np.linspace(0.0, launch_duration, 1001)
    launch_speeds = _poly_derivative(launch_coeff, launch_grid, 0)
    if (float(np.min(launch_speeds)) < -1e-6
            or np.any(np.diff(launch_speeds) < -1e-5)):
        raise RuntimeError("launch time law is not monotone; increase launch duration")

    if length >= launch_distance:
        duration = launch_duration + (length - launch_distance) / cruise
    else:
        lo, hi = 0.0, launch_duration
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            travelled = float(_poly_derivative(
                launch_integral, np.asarray([mid]), 0)[0])
            if travelled < length:
                lo = mid
            else:
                hi = mid
        duration = 0.5 * (lo + hi)

    count = int(math.ceil(duration / float(control_dt)))
    times = np.unique(np.minimum(
        np.arange(count + 1, dtype=float) * float(control_dt), duration))
    in_launch = times <= launch_duration
    speeds = np.full(len(times), cruise)
    tangential_acceleration = np.zeros(len(times))
    distances = np.empty(len(times))
    launch_times = times[in_launch]
    speeds[in_launch] = _poly_derivative(launch_coeff, launch_times, 0)
    tangential_acceleration[in_launch] = _poly_derivative(
        launch_coeff, launch_times, 1)
    distances[in_launch] = _poly_derivative(
        launch_integral, launch_times, 0)
    distances[~in_launch] = (launch_distance
                             + cruise * (times[~in_launch] - launch_duration))
    distances = np.minimum(distances, length)

    u = np.interp(distances, s_dense, u_dense)
    positions = np.stack([spatial.evaluate(value, 0)[1] for value in u])
    first = np.stack([spatial.evaluate(value, 1)[1] for value in u])
    second = np.stack([spatial.evaluate(value, 2)[1] for value in u])
    first_norm = np.linalg.norm(first, axis=1)
    tangents = first / first_norm[:, None]
    normal_second = second - tangents * np.sum(second * tangents, axis=1)[:, None]
    curvature = normal_second / first_norm[:, None] ** 2
    velocities = speeds[:, None] * tangents
    accelerations = (tangential_acceleration[:, None] * tangents
                     + speeds[:, None] ** 2 * curvature)
    edge_order = 2 if len(times) >= 3 else 1
    jerks = np.gradient(accelerations, times, axis=0, edge_order=edge_order)
    snaps = np.gradient(jerks, times, axis=0, edge_order=edge_order)

    waypoint_u = np.r_[0.0, np.cumsum(spatial.durations_s)]
    waypoint_s = np.interp(waypoint_u, u_dense, s_dense)
    waypoint_times = np.interp(waypoint_s, distances, times)
    waypoint_times[0] = 0.0
    waypoint_times[-1] = duration
    durations = np.diff(waypoint_times)
    return SampledTrajectory(
        points.copy(), durations, times, positions, velocities,
        accelerations, jerks, snaps, float(initial_yaw))


def _sample_violation(reference: FlatReference, limits: TrajectoryLimits) -> float:
    speed = float(np.linalg.norm(reference.velocity_ned)) / limits.cruise_speed_mps
    accel = math.sqrt(float(np.linalg.norm(reference.acceleration_ned))
                      / limits.acceleration_mps2)
    jerk = (float(np.linalg.norm(reference.jerk_ned)) / limits.jerk_mps3) ** (1.0 / 3.0)
    snap = (float(np.linalg.norm(reference.snap_ned)) / limits.snap_mps4) ** 0.25
    return max(1.0, speed, accel, jerk, snap)


def build_kinematic_trajectory(waypoints, initial_velocity, control_dt: float,
                               limits: TrajectoryLimits,
                               initial_yaw: float = 0.0,
                               initial_acceleration=None) -> tuple[SampledTrajectory,
                                                                  list[FlatReference],
                                                                  int]:
    """Build a fixed race line, then uniformly retime its independent clock."""
    points = np.asarray(waypoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("waypoints must be an (N,3) array with N >= 2")
    trajectory = _build_arc_length_trajectory(
        points, control_dt, limits, initial_yaw, initial_acceleration)
    for iteration in range(1, limits.max_iterations + 1):
        samples = trajectory.sample(control_dt, initial_yaw)
        violation = max(_sample_violation(reference, limits)
                        for reference in samples)
        if violation <= 1.0005:
            return trajectory, samples, iteration
        # Uniform retiming preserves the spatial curve.
        trajectory = trajectory.retime(min(2.0, 1.01 * violation))
    raise RuntimeError("trajectory could not satisfy its kinematic envelope")
