from collections import deque
import threading

import numpy as np

from . import camera
from .camera import Camera
from .detector import Detector


ASSOCIATION_DEG = 15.0
RANGE_TRUST_M = 45.0
POSE_KEEP = 256
CLOCK_KEEP = 64
PAIR_MAX_US = 100_000
MAP_HITS = 8
MAP_KEEP = 60
MAP_SETTLE_FRAMES = 12
MAX_MAP_RANGE_M = 250.0


class _ClockAlignment:
    def __init__(self) -> None:
        self.samples = deque(maxlen=CLOCK_KEEP)

    def reset(self) -> None:
        self.samples.clear()

    def observe(self, camera_usec: int, imu_usec: int) -> None:
        if self.samples and imu_usec < self.samples[-1][1]:
            self.samples.clear()
        if not self.samples or camera_usec != self.samples[-1][0]:
            self.samples.append((int(camera_usec), int(imu_usec)))

    def to_imu(self, camera_usec: int) -> int | None:
        if len(self.samples) < 4:
            return None
        values = np.asarray(self.samples, dtype=np.float64)
        camera_ref, imu_ref = values[-1]
        camera_delta = (values[:, 0] - camera_ref) * 1e-6
        imu_delta = (values[:, 1] - imu_ref) * 1e-6
        denominator = float(camera_delta @ camera_delta)
        if denominator < 1e-9:
            return None
        rate = float(camera_delta @ imu_delta) / denominator
        if not 0.05 <= rate <= 20.0:
            return None
        return int(round(imu_ref + rate * (float(camera_usec) - camera_ref)))


class Vision:
    def __init__(self, active_detector: str = "orange", camera_port: int = 5600) -> None:
        import cv2

        self._cv2 = cv2
        if active_detector == "orange":
            self.detector = Detector()
        elif active_detector == "yolo":
            from .yolo import YoloDetector

            self.detector = YoloDetector()
        else:
            raise ValueError("detector must be 'orange' or 'yolo'")
        self.camera = Camera(camera_port)
        self.lock = threading.Lock()
        self._preview = None
        self._known_gates = np.empty((0, 3))
        self._active_gate = 0
        self._poses = deque(maxlen=POSE_KEEP)
        self._corrections = deque(maxlen=POSE_KEEP)
        self._map_tracks = []
        self._map_stable_frames = 0
        self._observed_gates = []
        self._clock = _ClockAlignment()
        self._last_sync_usec = None
        self.matches = 0
        self.corrections = 0
        self.aim_error_deg = 0.0
        self.gate_offset_m = 0.0
        self.detector.attach(self.camera, self._on_detection)

    def set_known_gates(self, gates) -> None:
        values = np.asarray(gates, dtype=float).reshape(-1, 3)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("known gates must be finite")
        with self.lock:
            self._known_gates = values.copy()
            self._observed_gates = [deque(maxlen=30) for _ in values]

    def map_ready(self) -> bool:
        with self.lock:
            return self._map_stable_frames >= MAP_SETTLE_FRAMES and any(
                len(track) >= MAP_HITS for track in self._map_tracks
            )

    def mapped_gates(self, expected: int | None = None) -> np.ndarray:
        with self.lock:
            stable = [
                (index, len(track), np.median(np.asarray(track), axis=0))
                for index, track in enumerate(self._map_tracks)
                if len(track) >= MAP_HITS
            ]
        if expected is not None and expected > 0:
            stable = sorted(stable, key=lambda item: item[1], reverse=True)[
                : int(expected)
            ]
        stable.sort(key=lambda item: item[0])
        if not stable:
            return np.empty((0, 3))
        positions = np.asarray([position for _, _, position in stable], dtype=float)
        if len(positions) > 2:
            spacing = float(np.linalg.norm(positions[1] - positions[0]))
            for index in range(2, len(positions)):
                previous = positions[index - 1]
                point = positions[index]
                if spacing <= 5.0 or float(np.linalg.norm(point - previous)) >= spacing:
                    continue
                direction = point / max(float(np.linalg.norm(point)), 1e-9)
                projection = float(direction @ previous)
                discriminant = projection**2 + spacing**2 - float(previous @ previous)
                distance = projection + np.sqrt(max(0.0, discriminant))
                positions[index] = direction * max(float(np.linalg.norm(point)), distance)
        return positions

    def perceived_gates(self) -> np.ndarray:
        with self.lock:
            gates = self._known_gates.copy()
            for index, observations in enumerate(self._observed_gates):
                if observations:
                    gates[index] = np.median(np.asarray(observations), axis=0)
            return gates

    def sync(self, estimator, imu_usec: int, active_gate: int, correct: bool) -> None:
        imu_usec = int(imu_usec)
        with self.lock:
            if self._last_sync_usec is not None and imu_usec < self._last_sync_usec:
                self._poses.clear()
                self._corrections.clear()
                self._clock.reset()
            if imu_usec == self._last_sync_usec:
                return
            self._last_sync_usec = imu_usec
            self._active_gate = int(active_gate)
            self._poses.append(
                (
                    imu_usec,
                    estimator.R.copy(),
                    estimator.p.copy(),
                    estimator.uncertainty(),
                )
            )
            latest = self.camera.latest()
            if latest is not None:
                self._clock.observe(latest[0], imu_usec)
            corrections = list(self._corrections)
            self._corrections.clear()
        if not correct:
            return
        for gate_index, innovation, ray, along, across in corrections:
            if gate_index == int(active_gate) and estimator.fuse_known_gate(
                self._known_gates[gate_index], innovation, ray, along, across
            ):
                self.corrections += 1

    def reset(self) -> None:
        with self.lock:
            self._preview = None
            self._known_gates = np.empty((0, 3))
            self._poses.clear()
            self._corrections.clear()
            self._map_tracks.clear()
            self._map_stable_frames = 0
            self._observed_gates = []
            self._clock.reset()
            self._last_sync_usec = None
            self.matches = 0
            self.corrections = 0
            self.aim_error_deg = 0.0
            self.gate_offset_m = 0.0

    def close(self) -> None:
        self.detector.close()
        self.camera.close()

    def preview(self):
        with self.lock:
            return self._preview

    def health(self) -> dict:
        return {
            "detector": self.detector.name,
            "camera_frames": self.camera.n_frames,
            "detector_runs": self.detector.n_runs,
            "detector_ms": self.detector.last_latency_ms,
            "detections": self.detector.last_count,
            "matches": self.matches,
            "corrections": self.corrections,
            "aim_error_deg": self.aim_error_deg,
            "gate_offset_m": self.gate_offset_m,
            "mapped_gates": len(self.mapped_gates()),
            "map_ready": self.map_ready(),
            "camera_error": self.camera.last_error,
        }

    def _paired_pose(self, camera_usec: int):
        target = self._clock.to_imu(camera_usec)
        if target is None or not self._poses:
            return None
        best = min(self._poses, key=lambda item: abs(item[0] - target))
        return best if abs(best[0] - target) <= PAIR_MAX_US else None

    @staticmethod
    def _match(gate, pose, detections):
        _, rotation, position, uncertainty = pose
        expected = rotation.T @ (gate - position)
        expected_range = float(np.linalg.norm(expected))
        if expected_range < 1e-6:
            return None
        cone = np.sqrt(
            np.radians(ASSOCIATION_DEG) ** 2
            + (3.0 * uncertainty[1]) ** 2
            + (3.0 * uncertainty[0] / expected_range) ** 2
        )
        matches = []
        for direction, measured_range, confidence in detections:
            direction = np.asarray(direction, dtype=float)
            norm = float(np.linalg.norm(direction))
            measured_range = float(measured_range)
            if (
                confidence <= 0.7
                or norm < 1e-9
                or not np.isfinite(measured_range)
                or measured_range <= 0.0
            ):
                continue
            direction /= norm
            if direction[0] <= 0.15:
                continue
            if measured_range > RANGE_TRUST_M:
                if expected_range < 0.6 * RANGE_TRUST_M:
                    continue
            elif not 0.6 < measured_range / expected_range < 1.6:
                continue
            angle = float(
                np.arccos(
                    np.clip(float(direction @ expected) / expected_range, -1.0, 1.0)
                )
            )
            if angle < cone:
                matches.append((angle, direction, measured_range))
        return min(matches, default=None, key=lambda item: item[0])

    def _map_frame(self, pose, detections) -> None:
        _, rotation, position, _ = pose
        measurements = []
        for direction, measured_range, confidence in detections:
            direction = np.asarray(direction, dtype=float)
            norm = float(np.linalg.norm(direction))
            measured_range = float(measured_range)
            if (
                confidence <= 0.7
                or norm < 1e-9
                or not np.isfinite(direction).all()
                or direction[0] / norm <= 0.15
                or not 2.0 <= measured_range <= MAX_MAP_RANGE_M
            ):
                continue
            measurements.append(
                position + rotation @ (direction / norm * measured_range)
            )
        available = set(range(len(self._map_tracks)))
        created = False
        for measured in sorted(
            measurements, key=lambda point: float(np.linalg.norm(point - position))
        ):
            candidates = [
                (
                    float(
                        np.linalg.norm(
                            measured - np.median(np.asarray(self._map_tracks[index]), axis=0)
                        )
                    ),
                    index,
                )
                for index in available
            ]
            distance, index = min(candidates, default=(float("inf"), -1))
            merge_radius = max(4.0, 0.12 * float(np.linalg.norm(measured - position)))
            if distance <= merge_radius:
                self._map_tracks[index].append(measured)
                available.remove(index)
            else:
                self._map_tracks.append(deque([measured], maxlen=MAP_KEEP))
                created = True
        self._map_stable_frames = (
            0 if created or not measurements else self._map_stable_frames + 1
        )

    def _on_detection(self, camera_usec, crop, detections, polygons) -> None:
        raw = np.asarray(crop).copy()
        if self.detector.last_mask is None:
            evidence = raw.copy()
            evidence_name = self.detector.name
        else:
            evidence = self._cv2.cvtColor(
                self.detector.last_mask, self._cv2.COLOR_GRAY2BGR
            )
            evidence_name = "RED MASK"
        image = self.detector.annotate(evidence, polygons)
        with self.lock:
            pose = self._paired_pose(int(camera_usec))
            gate_index = self._active_gate
            if pose is not None and not len(self._known_gates):
                self._map_frame(pose, detections)
            elif pose is not None and 0 <= gate_index < len(self._known_gates):
                _, rotation, position, _ = pose
                gate = self._known_gates[gate_index]
                expected = camera.pixel_of(
                    rotation.T @ (gate - position)
                )
                if expected is not None:
                    x, y = (int(round(value)) for value in expected)
                    self._cv2.drawMarker(
                        image,
                        (x, y),
                        (255, 210, 40),
                        self._cv2.MARKER_CROSS,
                        16,
                        2,
                        self._cv2.LINE_AA,
                    )
                    self._cv2.putText(
                        image,
                        f"PLAN G{gate_index + 1}",
                        (min(x + 8, image.shape[1] - 80), max(40, y - 8)),
                        self._cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (255, 210, 40),
                        1,
                        self._cv2.LINE_AA,
                    )
                match = self._match(gate, pose, detections)
                if match is not None:
                    angle, direction, measured_range = match
                    ray = rotation @ direction
                    measured = position + ray * measured_range
                    self._observed_gates[gate_index].append(measured)
                    self.aim_error_deg = float(np.degrees(angle))
                    self.gate_offset_m = float(
                        np.linalg.norm(gate - measured)
                    )
                    observed = camera.pixel_of(direction)
                    if observed is not None:
                        seen = tuple(int(round(value)) for value in observed)
                        self._cv2.drawMarker(
                            image,
                            seen,
                            (40, 150, 240),
                            self._cv2.MARKER_TILTED_CROSS,
                            12,
                            2,
                            self._cv2.LINE_AA,
                        )
                        if expected is not None:
                            self._cv2.line(
                                image,
                                (x, y),
                                seen,
                                (125, 125, 125),
                                1,
                                self._cv2.LINE_AA,
                            )
                    range_sigma = (
                        0.5 * measured_range
                        if measured_range > RANGE_TRUST_M
                        else np.hypot(
                            2.0
                            * measured_range**2
                            / (camera.FX * camera.GATE_SIZE_M),
                            0.05 * measured_range,
                        )
                    )
                    bearing_sigma = 2.0 * measured_range / camera.FX
                    self._corrections.append(
                        (
                            gate_index,
                            gate - measured,
                            ray,
                            range_sigma,
                            bearing_sigma,
                        )
                    )
                    self.matches += 1

            labels = (
                (raw, "RAW INPUT"),
                (
                    image,
                    f"{evidence_name}  {len(detections)} gates  "
                    f"{self.detector.last_latency_ms:.1f} ms",
                ),
            )
            for panel, label in labels:
                self._cv2.rectangle(
                    panel, (0, 0), (panel.shape[1], 26), (245, 245, 245), -1
                )
                self._cv2.putText(
                    panel,
                    label,
                    (8, 18),
                    self._cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (35, 35, 35),
                    1,
                    self._cv2.LINE_AA,
                )
            self._preview = (int(camera_usec), np.vstack((raw, image)))
