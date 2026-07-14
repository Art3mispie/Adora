from pathlib import Path
import time

import numpy as np

from . import camera
from .detector import CONFIDENCE, Detector


WEIGHTS = Path(__file__).with_name("gates_v2_320n_oob.pt")


class YoloDetector(Detector):
    name = "YOLO OBB"

    def __init__(self, weights: str | Path = WEIGHTS, device: str = "cpu") -> None:
        super().__init__()
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = device

    def infer(self, crop: np.ndarray):
        started = time.perf_counter()
        result = self.model.predict(
            source=np.asarray(crop), conf=0.25, device=self.device, verbose=False
        )[0]
        boxes = () if result.obb is None else result.obb.xyxyxyxy
        polygons = []
        detections = []
        if boxes is not None:
            for value in boxes:
                quad = value.detach().cpu().numpy().astype(float)
                measured = camera.gate_pose(quad)
                if measured is None:
                    centre = quad.mean(axis=0)
                    extent = np.ptp(quad, axis=0)
                    measured = camera.ray(
                        float(centre[0]),
                        float(centre[1]),
                        float(extent[0]),
                        float(extent[1]),
                    )
                direction, distance = measured
                detections.append((direction, distance, CONFIDENCE))
                polygons.append((quad, distance))
        self.last_mask = None
        self.last_isolated = None
        self.last_count = len(detections)
        self.n_runs += 1
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        return detections, polygons
