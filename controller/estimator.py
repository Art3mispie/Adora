import numpy as np


G = 9.81
PITCH_AT_SPAWN = -np.radians(20.0)
TILT_TOLERANCE = np.radians(5.0)
ACCEL_NOISE = 0.5
GYRO_NOISE = 0.03
BIAS_NOISE = 0.01
POSITION_STEP_LIMIT = 0.25
CAMERA_POSITION_STEP_LIMIT = 0.08
CAMERA_ATTITUDE_STEP_LIMIT = np.radians(0.25)
NIS_MAX = 16.0
NIS_REJECT = 100.0


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def skew(vector) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_so3(vector) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-9:
        return np.eye(3)
    cross = skew(np.asarray(vector, dtype=float) / angle)
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


class Estimator:
    def __init__(self) -> None:
        self.gyro_bias = np.zeros(3)
        self.gravity = G
        self.reset()

    def reset(self) -> None:
        self.gravity_ned = np.array([0.0, 0.0, self.gravity])
        self.R = rotation_from_rpy(0.0, PITCH_AT_SPAWN, 0.0)
        self.v = np.zeros(3)
        self.p = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.P = np.diag(
            [0.2**2] * 3
            + [0.1**2] * 3
            + [np.radians(2.0) ** 2] * 3
            + [0.3**2] * 3
        )
        self.grounded = True
        self.anchor = "spawn"

    def uncertainty(self) -> tuple[float, float]:
        return (
            float(np.sqrt(max(np.trace(self.P[0:3, 0:3]) / 3.0, 0.0))),
            float(np.sqrt(max(np.trace(self.P[6:9, 6:9]) / 3.0, 0.0))),
        )

    def calibrate(self, imu_sample) -> bool:
        values = np.asarray(imu_sample, dtype=float)
        if values.size < 6:
            return False
        mean = values.reshape(-1, 6).mean(axis=0)
        acceleration, gyro = mean[:3], mean[3:]
        gravity = float(np.linalg.norm(acceleration))
        if not 0.7 * G < gravity < 1.3 * G or np.linalg.norm(gyro) > 0.2:
            return False
        pitch = float(np.arcsin(np.clip(acceleration[0] / gravity, -1.0, 1.0)))
        roll = float(np.arctan2(-acceleration[1], -acceleration[2]))
        if abs(roll) > TILT_TOLERANCE or abs(pitch - PITCH_AT_SPAWN) > TILT_TOLERANCE:
            return False
        self.reset()
        self.gyro_bias = gyro.copy()
        self.anchor = "IMU"
        return True

    def step_attitude(self, gyro, dt: float) -> None:
        if not self.grounded:
            self.R = self.R @ exp_so3(
                (np.asarray(gyro, dtype=float) - self.gyro_bias) * dt
            )

    def step(self, acceleration, dt: float) -> None:
        body_acceleration = np.asarray(acceleration, dtype=float) - self.accel_bias
        world_acceleration = self.R @ body_acceleration + self.gravity_ned
        if self.grounded:
            return
        self.p += self.v * dt + 0.5 * world_acceleration * dt * dt
        self.v += world_acceleration * dt
        transition = np.eye(12)
        transition[0:3, 3:6] = dt * np.eye(3)
        transition[3:6, 6:9] = -(self.R @ skew(body_acceleration)) * dt
        transition[3:6, 9:12] = -self.R * dt
        covariance = transition @ self.P @ transition.T
        identity = np.eye(3)
        accel_variance = ACCEL_NOISE**2
        covariance[0:3, 0:3] += accel_variance * dt**3 / 3.0 * identity
        covariance[0:3, 3:6] += accel_variance * dt**2 / 2.0 * identity
        covariance[3:6, 0:3] += accel_variance * dt**2 / 2.0 * identity
        covariance[3:6, 3:6] += accel_variance * dt * identity
        covariance[6:9, 6:9] += GYRO_NOISE**2 * dt * identity
        covariance[9:12, 9:12] += BIAS_NOISE**2 * dt * identity
        self.P = covariance

    def release(self) -> None:
        self.grounded = False

    def anchor_position(self, position_ned, sigma: float = 0.75) -> bool:
        position = np.asarray(position_ned, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            return False
        measurement = np.zeros((3, 12))
        measurement[:, 0:3] = np.eye(3)
        noise = max(float(sigma), 0.05) ** 2 * np.eye(3)
        innovation = position - self.p
        residual_covariance = measurement @ self.P @ measurement.T + noise
        self._update(measurement, residual_covariance, innovation)
        self.anchor = "gate"
        return True

    def fuse_known_gate(
        self,
        gate_ned,
        innovation,
        world_ray,
        sigma_along: float,
        sigma_across: float,
    ) -> bool:
        gate = np.asarray(gate_ned, dtype=float)
        innovation = np.asarray(innovation, dtype=float)
        ray = np.asarray(world_ray, dtype=float)
        if (
            gate.shape != (3,)
            or innovation.shape != (3,)
            or ray.shape != (3,)
            or not np.isfinite(np.r_[gate, innovation, ray]).all()
        ):
            return False
        ray /= max(float(np.linalg.norm(ray)), 1e-12)
        noise = (
            (float(sigma_along) ** 2 - float(sigma_across) ** 2)
            * np.outer(ray, ray)
            + float(sigma_across) ** 2 * np.eye(3)
        )
        measurement = np.zeros((3, 12))
        measurement[:, 0:3] = np.eye(3)
        measurement[:, 6:9] = -skew(gate - self.p) @ self.R
        residual = measurement @ self.P @ measurement.T + noise
        nis = float(innovation @ np.linalg.solve(residual, innovation))
        if nis > NIS_REJECT:
            return False
        if nis > NIS_MAX:
            residual *= nis / NIS_MAX
        self._update(
            measurement,
            residual,
            innovation,
            CAMERA_POSITION_STEP_LIMIT,
            CAMERA_ATTITUDE_STEP_LIMIT,
        )
        self.anchor = "camera gate"
        return True

    def _update(
        self,
        measurement,
        residual_covariance,
        innovation,
        maximum_position_step: float = POSITION_STEP_LIMIT,
        maximum_attitude_step: float | None = None,
    ) -> None:
        gain = self.P @ measurement.T @ np.linalg.inv(residual_covariance)
        correction = gain @ innovation
        scale = min(
            1.0,
            maximum_position_step
            / max(float(np.linalg.norm(correction[0:3])), maximum_position_step),
        )
        if maximum_attitude_step is not None:
            scale = min(
                scale,
                maximum_attitude_step
                / max(float(np.linalg.norm(correction[6:9])), maximum_attitude_step),
            )
        gain *= scale
        correction *= scale
        self.p += correction[0:3]
        self.v += correction[3:6]
        self.R = self.R @ exp_so3(correction[6:9])
        self.accel_bias += correction[9:12]
        identity_minus_gain = np.eye(12) - gain @ measurement
        assumed_noise = residual_covariance - measurement @ self.P @ measurement.T
        covariance = (
            identity_minus_gain @ self.P @ identity_minus_gain.T
            + gain @ assumed_noise @ gain.T
        )
        self.P = 0.5 * (covariance + covariance.T)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.v))
