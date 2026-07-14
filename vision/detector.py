"""Fast orange-gate detector for the simulator."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import numpy as np

from . import camera

RED_THRESHOLD = 150
MIN_AREA_PX = 12
MAX_AREA_FRACTION = 0.20
MAX_ASPECT_RATIO = 2.0
MAX_COMPONENTS = 12
CONFIDENCE = 0.95
APERTURE_TO_OUTER = 1.72
APERTURE_RECTANGULARITY = 0.45


@dataclass(frozen=True, slots=True)
class OrangeComponent:
    quad: np.ndarray
    area_px: int
    aspect_ratio: float


def _split_overlapping_gates(
    component_mask: np.ndarray,
    *,
    min_area_px: int,
    max_aspect_ratio: float,
) -> tuple[OrangeComponent, ...]:
    """Split touching gate frames using their separate dark apertures."""
    import cv2

    contours, hierarchy = cv2.findContours(
        component_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return ()

    candidates: list[tuple[float, tuple, float]] = []
    for index, contour in enumerate(contours):
        # A gate aperture is a direct hole in the orange component.
        parent = int(hierarchy[0, index, 3])
        if parent < 0 or int(hierarchy[0, parent, 3]) >= 0:
            continue
        area = float(abs(cv2.contourArea(contour)))
        if area < max(12.0, float(min_area_px)):
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            continue
        corners = len(cv2.approxPolyDP(contour, 0.025 * perimeter, True))
        if not 4 <= corners <= 10:
            continue

        rect = cv2.minAreaRect(contour)
        width, height = (float(value) for value in rect[1])
        if min(width, height) < 2.0:
            continue
        aspect = max(width, height) / max(min(width, height), 1e-6)
        rectangularity = area / max(width * height, 1e-6)
        if (aspect > max_aspect_ratio
                or rectangularity < APERTURE_RECTANGULARITY):
            continue
        candidates.append((area, rect, aspect))

    if len(candidates) < 2:
        return ()

    largest = max(item[0] for item in candidates)
    candidates = [item for item in candidates if item[0] >= 0.02 * largest]
    if len(candidates) < 2:
        return ()

    gates: list[OrangeComponent] = []
    for aperture_area, rect, aspect in candidates:
        centre, (width, height), angle = rect
        outer = (centre, (
            width * APERTURE_TO_OUTER,
            height * APERTURE_TO_OUTER,
        ), angle)
        quad = cv2.boxPoints(outer).astype(float)
        red_area = int(max(
            1.0,
            width * height * APERTURE_TO_OUTER ** 2 - aperture_area,
        ))
        gates.append(OrangeComponent(quad, red_area, aspect))
    return tuple(gates)


def square_crop(image: np.ndarray) -> tuple[int, int, np.ndarray]:
    """Return ``(x_offset, y_offset, centred_square)`` for any image shape."""
    height, width = image.shape[:2]
    size = min(height, width)
    x0, y0 = (width - size) // 2, (height - size) // 2
    return x0, y0, image[y0:y0 + size, x0:x0 + size]


def orange_components(
    bgr: np.ndarray,
    *,
    threshold: int | None = None,
    min_area_px: int | None = None,
    max_area_fraction: float | None = None,
    max_aspect_ratio: float | None = None,
    max_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[OrangeComponent, ...]]:
    """Find orange components and split touching gate frames."""
    import cv2

    image = np.asarray(bgr, dtype=np.uint8)
    threshold = int(RED_THRESHOLD if threshold is None else threshold)
    min_area_px = int(MIN_AREA_PX if min_area_px is None else min_area_px)
    max_area_fraction = float(
        MAX_AREA_FRACTION
        if max_area_fraction is None else max_area_fraction
    )
    max_aspect_ratio = float(
        MAX_ASPECT_RATIO
        if max_aspect_ratio is None else max_aspect_ratio
    )
    max_components = int(
        MAX_COMPONENTS if max_components is None else max_components
    )

    red = image[:, :, 2]
    selected = red >= np.uint8(np.clip(threshold, 0, 255))
    mask = selected.astype(np.uint8) * 255
    isolated = np.zeros_like(image)
    isolated[:, :, 2] = np.where(selected, red, 0)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    frame_area = float(image.shape[0] * image.shape[1])
    found: list[OrangeComponent] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < min_area_px or area > max_area_fraction * frame_area:
            continue
        if min(width, height) < 3:
            continue

        component_mask = (labels == label).astype(np.uint8) * 255
        split = _split_overlapping_gates(
            component_mask,
            min_area_px=min_area_px,
            max_aspect_ratio=max_aspect_ratio,
        )
        if split:
            found.extend(split)
            continue

        ys, xs = np.nonzero(labels == label)
        points = np.column_stack((xs, ys)).astype(np.float32)
        rect = cv2.minAreaRect(points)
        rw, rh = (float(value) for value in rect[1])
        if min(rw, rh) < 2.0:
            continue
        aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
        if aspect > max_aspect_ratio:
            continue
        found.append(OrangeComponent(
            cv2.boxPoints(rect).astype(float), area, aspect
        ))

    found.sort(key=lambda item: item.area_px, reverse=True)
    return isolated, mask, tuple(found[:max_components])


class Detector:
    name = "orange threshold"

    def __init__(self, *_unused, **_unused_kwargs) -> None:
        self.n_runs = 0
        self.last_latency_ms = 0.0
        self.last_count = 0
        self.last_mask: np.ndarray | None = None
        self.last_isolated: np.ndarray | None = None
        self._cam = None
        self._emit = None
        self._stop = False

    def attach(self, cam, emit) -> None:
        self._cam, self._emit = cam, emit
        threading.Thread(
            target=self._loop, daemon=True, name="gate-detector"
        ).start()

    def close(self) -> None:
        self._stop = True

    def infer(self, crop: np.ndarray):
        """Return pose measurements and display polygons for one crop."""
        started = time.perf_counter()
        isolated, mask, components = orange_components(crop)
        measured = []
        for component in components:
            quad = component.quad
            pose = camera.gate_pose(quad)
            if pose is None:
                centre = quad.mean(axis=0)
                extent = np.ptp(quad, axis=0)
                pose = camera.ray(
                    float(centre[0]), float(centre[1]),
                    float(extent[0]), float(extent[1]),
                )
            direction, distance = pose
            measured.append((direction, float(distance), component))

        calibration = [
            distance * np.sqrt(component.area_px)
            for _, distance, component in measured
            if distance <= 80.0 and component.area_px > 0
        ]
        area_scale = float(np.median(calibration)) if calibration else None
        detections = []
        polygons = []
        for direction, distance, component in measured:
            if area_scale is not None and component.area_px > 0:
                distance = area_scale / np.sqrt(component.area_px)
            detections.append((direction, distance, CONFIDENCE))
            polygons.append((component.quad, distance))

        self.last_isolated = isolated
        self.last_mask = mask
        self.last_count = len(detections)
        self.n_runs += 1
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        return detections, polygons

    @staticmethod
    def annotate(crop: np.ndarray, polygons, mask: np.ndarray | None = None):
        """Draw high-contrast gate boxes without hiding the source image."""
        import cv2

        output = np.asarray(crop).copy()
        if mask is not None:
            selected = mask > 0
            if np.any(selected):
                warm = np.zeros_like(output)
                warm[:, :, 1] = 70
                warm[:, :, 2] = 255
                output[selected] = cv2.addWeighted(
                    output, 0.45, warm, 0.55, 0.0
                )[selected]

        for index, (quad, distance) in enumerate(polygons, start=1):
            points = np.round(quad).astype(np.int32)
            cv2.polylines(output, [points], True, (255, 235, 60), 2, cv2.LINE_AA)
            for point in points:
                cv2.circle(output, tuple(point), 3, (40, 255, 255), -1, cv2.LINE_AA)
            x, y = points[np.argmin(points[:, 0] + points[:, 1])]
            label = f"G{index}"
            origin = (max(4, int(x) - 2), max(45, int(y) - 5))
            cv2.putText(output, label, origin, cv2.FONT_HERSHEY_DUPLEX,
                        0.40, (8, 8, 8), 2, cv2.LINE_AA)
            cv2.putText(output, label, origin, cv2.FONT_HERSHEY_DUPLEX,
                        0.40, (255, 245, 120), 1, cv2.LINE_AA)
        return output

    def _loop(self) -> None:
        last_stamp = None
        while not self._stop:
            got = self._cam.latest() if self._cam is not None else None
            if got is None or got[0] == last_stamp:
                time.sleep(0.002)
                continue
            stamp, full = got
            last_stamp = stamp
            _, _, crop = square_crop(full)
            detections, polygons = self.infer(crop)
            if self._emit is not None:
                self._emit(stamp, crop, detections, polygons)
