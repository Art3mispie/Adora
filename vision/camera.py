import socket
import struct
import threading
import time

import numpy as np


WIDTH, HEIGHT = 640, 360
SIZE = HEIGHT
CX = CY = SIZE / 2.0
FX = FY = 320.0
CAMERA_TILT = np.radians(20.0)
GATE_SIZE_M = 2.7
PACKET_HEADER = "<IHHIIQ"

_COS_TILT = float(np.cos(CAMERA_TILT))
_SIN_TILT = float(np.sin(CAMERA_TILT))


def ray(u: float, v: float, width: float, height: float):
    x_camera = (u - CX) / FX
    y_camera = (v - CY) / FY
    norm = float(np.sqrt(1.0 + x_camera**2 + y_camera**2))
    direction = np.array(
        [
            _COS_TILT + _SIN_TILT * y_camera,
            x_camera,
            -_SIN_TILT + _COS_TILT * y_camera,
        ]
    ) / norm
    depth = FX * GATE_SIZE_M / max(width, height, 1.0)
    return direction, depth * norm


def gate_pose(quad: np.ndarray):
    quad = np.asarray(quad, dtype=float)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return None
    x = (quad[:, 0] - CX) / FX
    y = (quad[:, 1] - CY) / FY
    half = GATE_SIZE_M / 2.0
    world_x = np.array([-half, half, half, -half])
    world_y = np.array([-half, -half, half, half])
    matrix = np.zeros((8, 9))
    matrix[0::2, 0], matrix[0::2, 1], matrix[0::2, 2] = world_x, world_y, 1.0
    matrix[0::2, 6], matrix[0::2, 7], matrix[0::2, 8] = (
        -x * world_x,
        -x * world_y,
        -x,
    )
    matrix[1::2, 3], matrix[1::2, 4], matrix[1::2, 5] = world_x, world_y, 1.0
    matrix[1::2, 6], matrix[1::2, 7], matrix[1::2, 8] = (
        -y * world_x,
        -y * world_y,
        -y,
    )
    _, _, vectors = np.linalg.svd(matrix)
    homography = vectors[-1].reshape(3, 3)
    norm_x = float(np.linalg.norm(homography[:, 0]))
    norm_y = float(np.linalg.norm(homography[:, 1]))
    scale = 2.0 / (norm_x + norm_y + 1e-12)
    if (
        not 0.5 < scale * norm_x < 2.0
        or not 0.5 < scale * norm_y < 2.0
        or abs(float(homography[:, 0] @ homography[:, 1]))
        / (norm_x * norm_y + 1e-12)
        > 0.5
    ):
        return None
    translation = scale * homography[:, 2]
    if translation[2] < 0.0:
        translation = -translation
    body = np.array(
        [
            _COS_TILT * translation[2] + _SIN_TILT * translation[1],
            translation[0],
            -_SIN_TILT * translation[2] + _COS_TILT * translation[1],
        ]
    )
    distance = float(np.linalg.norm(translation))
    return body / max(float(np.linalg.norm(body)), 1e-12), distance


def pixel_of(direction: np.ndarray, bounded: bool = True):
    direction = np.asarray(direction, dtype=float)
    forward = _COS_TILT * direction[0] - _SIN_TILT * direction[2]
    down = _SIN_TILT * direction[0] + _COS_TILT * direction[2]
    if forward <= 0.0:
        return None
    u = CX + FX * direction[1] / forward
    v = CY + FY * down / forward
    if not bounded or 0.0 <= u < SIZE and 0.0 <= v < SIZE:
        return u, v
    return None


class Reassembler:
    def __init__(self) -> None:
        self.frames = {}
        self.header_size = struct.calcsize(PACKET_HEADER)

    def feed(self, packet: bytes):
        frame_id, chunk_id, total, _, _, time_ns = struct.unpack_from(
            PACKET_HEADER, packet
        )
        chunks = self.frames.setdefault(frame_id, {})
        chunks[chunk_id] = packet[self.header_size :]
        for stale in [key for key in self.frames if key < frame_id - 3]:
            del self.frames[stale]
        if len(chunks) >= total and all(index in chunks for index in range(total)):
            jpeg = b"".join(chunks[index] for index in range(total))
            del self.frames[frame_id]
            return time_ns // 1000, jpeg
        return None


class Camera:
    def __init__(self, port: int = 5600) -> None:
        import cv2

        self._cv2 = cv2
        self._latest = None
        self._lock = threading.Lock()
        self._stop = False
        self.n_frames = 0
        self.last_error = None
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(0.25)
        self._socket.bind(("0.0.0.0", int(port)))
        threading.Thread(
            target=self._receive, daemon=True, name="camera-receive"
        ).start()

    def latest(self):
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._stop = True
        try:
            self._socket.close()
        except OSError:
            pass

    def _receive(self) -> None:
        reassembler = Reassembler()
        while not self._stop:
            try:
                packet, _ = self._socket.recvfrom(65536)
                complete = reassembler.feed(packet)
                if complete is None:
                    continue
                stamp, jpeg = complete
                image = self._cv2.imdecode(
                    np.frombuffer(jpeg, np.uint8), self._cv2.IMREAD_COLOR
                )
                if image is None:
                    raise ValueError("could not decode camera frame")
                with self._lock:
                    self._latest = (int(stamp), image)
                self.n_frames += 1
            except socket.timeout:
                continue
            except OSError as error:
                if not self._stop:
                    self.last_error = str(error)
                    time.sleep(0.01)
            except Exception as error:
                self.last_error = str(error)
                time.sleep(0.01)
