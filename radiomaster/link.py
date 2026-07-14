import collections
import struct
import threading
import time


RACE_STATUS = 1
TRACK_DATA = 2


class SimulatorLink:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 14550,
        gyro_sign: float = -1.0,
        throttle_limit: float = 1.0,
    ) -> None:
        from pymavlink import mavutil

        self.mavutil = mavutil
        self.gyro_sign = float(gyro_sign)
        self.throttle_limit = float(throttle_limit)
        self.conn = mavutil.mavlink_connection(f"udpin:{host}:{port}")
        print("[link] waiting for simulator heartbeat", flush=True)
        self.conn.wait_heartbeat()
        print(f"[link] connected to system {self.conn.target_system}", flush=True)
        self.rate_mask = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
        self.armed = False
        self.imu = (0.0,) * 6
        self.imu_queue = collections.deque(maxlen=1024)
        self._imu_lock = threading.Lock()
        self.message_counts = collections.Counter()
        self.active_gate = 0
        self.gate_count = 0
        self.race_finished = False
        self.race_status_updates = 0
        self.collisions = []
        self._track_chunks = {}
        self._expected_track_chunks = {}
        threading.Thread(target=self._receive, daemon=True, name="mavlink-receive").start()
        threading.Thread(
            target=self._housekeeping, daemon=True, name="mavlink-heartbeat"
        ).start()

    def arm(self, armed: bool = True) -> None:
        mavlink = self.mavutil.mavlink
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            int(armed),
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def disarm(self) -> None:
        self.arm(False)

    def send_rates(self, roll: float, pitch: float, yaw: float, throttle: float) -> None:
        self.conn.mav.set_attitude_target_send(
            0,
            self.conn.target_system,
            self.conn.target_component,
            self.rate_mask,
            [1.0, 0.0, 0.0, 0.0],
            float(roll),
            float(pitch),
            float(yaw),
            min(self.throttle_limit, max(0.0, float(throttle))),
        )

    def reset(self) -> None:
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            31000,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def drain_imu(self) -> list:
        with self._imu_lock:
            samples = list(self.imu_queue)
            self.imu_queue.clear()
        return samples

    def _housekeeping(self) -> None:
        mavlink = self.mavutil.mavlink
        tick = 0
        while True:
            self.conn.mav.timesync_send(int(time.time_ns()), 0)
            if tick % 5 == 0:
                self.conn.mav.heartbeat_send(
                    mavlink.MAV_TYPE_GCS, mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
                )
            tick += 1
            time.sleep(0.1)

    def _receive(self) -> None:
        while True:
            try:
                message = self.conn.recv_match(blocking=False)
            except ConnectionResetError:
                time.sleep(0.5)
                continue
            if message is None:
                time.sleep(0.001)
                continue
            message_type = message.get_type()
            self.message_counts[message_type] += 1
            if message_type == "HEARTBEAT":
                self.armed = bool(
                    message.base_mode
                    & self.mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
            elif message_type == "HIGHRES_IMU":
                sample = (
                    message.xacc,
                    message.yacc,
                    message.zacc,
                    self.gyro_sign * message.xgyro,
                    self.gyro_sign * message.ygyro,
                    self.gyro_sign * message.zgyro,
                )
                self.imu = sample
                with self._imu_lock:
                    self.imu_queue.append((message.time_usec, sample))
            elif message_type == "ENCAPSULATED_DATA":
                raw = bytes(message.data)
                if raw and raw[0] == RACE_STATUS and len(raw) >= 37:
                    self._receive_race_status(raw)
                elif raw and raw[0] == TRACK_DATA:
                    self._receive_track_chunk(message, raw)
            elif message_type == "DATA_TRANSMISSION_HANDSHAKE":
                transfer_id = int(message.width)
                self._track_chunks[transfer_id] = {}
                self._expected_track_chunks[transfer_id] = int(message.packets)
            elif message_type == "COLLISION":
                self.collisions.append(
                    (message.id, message.threat_level, message.horizontal_minimum_delta)
                )

    def _receive_race_status(self, raw: bytes) -> None:
        _, _, _, finish_time_ns, active_gate, _ = struct.unpack_from("<BQqqIq", raw)
        self.active_gate = int(active_gate)
        self.race_finished = finish_time_ns >= 0
        self.race_status_updates += 1

    def _receive_track_chunk(self, message, raw: bytes) -> None:
        if len(raw) < 3:
            return
        _, transfer_id = struct.unpack_from("<BH", raw)
        expected = self._expected_track_chunks.get(transfer_id)
        if not expected:
            return
        chunks = self._track_chunks[transfer_id]
        chunks[int(message.seqnr)] = raw[3:]
        if len(chunks) < expected or any(index not in chunks for index in range(expected)):
            return
        payload = b"".join(chunks[index] for index in range(expected))
        del self._track_chunks[transfer_id]
        del self._expected_track_chunks[transfer_id]
        if len(payload) >= 2:
            self.gate_count = struct.unpack_from("<H", payload)[0]
