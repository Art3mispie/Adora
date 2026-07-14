from collections import deque
import queue
import threading
import time

import numpy as np

from gui import Dashboard
from vision.pipeline import Vision
from .mission import build_route
from .pilot import RacePilot, say


ALIGNMENT_SAMPLES = 180


def fly(
    simulator,
    *,
    camera_port: int = 5600,
    detector: str = "orange",
) -> None:
    pilot = RacePilot()
    say(
        f"{pilot.vehicle.profile_name}, hover rotor "
        f"{pilot.vehicle.hover_rotor_fraction:.4f}; mapping gates from vision"
    )
    try:
        vision = Vision(detector, camera_port)
        say(f"vision up ({vision.detector.name}); no saved course")
    except Exception as error:
        vision = None
        say(f"vision off ({error})")

    commands = queue.SimpleQueue()
    stopped = threading.Event()
    snapshot_lock = threading.Lock()
    snapshot = [{}]
    shared = {
        "active_gate": 0,
        "crossed_gate": 0,
        "collisions": 0,
        "imu_rate": np.zeros(3),
        "imu_acceleration": np.zeros(3),
        "imu_usec": None,
        "gates": np.empty((0, 3)),
        "gate_count": 0,
    }

    def publish() -> None:
        value = pilot.snapshot()
        gates = np.asarray(shared["gates"])
        if len(gates) and vision is not None:
            perceived = vision.perceived_gates()
            visible_gates = perceived if len(perceived) == len(gates) else gates
        elif vision is not None:
            expected = int(shared["gate_count"])
            visible_gates = vision.mapped_gates(expected or None)
        else:
            visible_gates = gates
        value.update(
            active_gate=int(shared["active_gate"]),
            gate_count=int(shared["gate_count"]),
            collisions=int(shared["collisions"]),
            imu_rate=np.asarray(shared["imu_rate"]).copy(),
            imu_acceleration=np.asarray(shared["imu_acceleration"]).copy(),
            course=visible_gates.copy(),
            vision={} if vision is None else vision.health(),
        )
        with snapshot_lock:
            snapshot[0] = value

    def read_snapshot() -> dict:
        with snapshot_lock:
            return snapshot[0]

    def request_reset() -> None:
        commands.put("reset")

    def request_kill() -> None:
        commands.put("kill")

    def reset_run() -> None:
        simulator.reset()
        simulator.disarm()
        simulator.collisions.clear()
        pilot.reset()
        if vision:
            vision.reset()
        shared.update(
            active_gate=0,
            crossed_gate=0,
            collisions=0,
            imu_rate=np.zeros(3),
            imu_acceleration=np.zeros(3),
            imu_usec=None,
            gates=np.empty((0, 3)),
            gate_count=int(simulator.gate_count),
        )

    def flight_loop() -> None:
        last_imu_usec = None
        last_imu_wall_s = None
        next_control_usec = None
        finished_disarmed = False
        killed_reset = False
        race_status_ready = False
        reset_ready_wall_s = time.monotonic() + 0.25
        alignment_samples = deque(maxlen=ALIGNMENT_SAMPLES)
        period_usec = 1e6 / pilot.control_hz

        while not stopped.is_set():
            try:
                command_name = commands.get_nowait()
            except queue.Empty:
                command_name = None
            if command_name == "reset":
                reset_run()
                last_imu_usec = None
                last_imu_wall_s = None
                next_control_usec = None
                finished_disarmed = False
                killed_reset = False
                alignment_samples.clear()
                race_status_ready = False
                reset_ready_wall_s = time.monotonic() + 0.25
                say("reset")
                publish()
            elif command_name == "kill":
                pilot.kill()

            received_fresh = False
            for usec, sample in simulator.drain_imu():
                if last_imu_usec is not None and usec < last_imu_usec:
                    last_imu_usec = None
                    next_control_usec = None
                if last_imu_usec is not None and usec == last_imu_usec:
                    continue
                dt_imu = (
                    (usec - last_imu_usec) * 1e-6 if last_imu_usec is not None else 0.0
                )
                received_fresh = True
                last_imu_usec = usec
                shared["imu_usec"] = usec
                shared["imu_acceleration"] = np.asarray(sample[:3], dtype=float)
                shared["imu_rate"] = np.asarray(sample[3:], dtype=float)
                if 0.0 < dt_imu < 0.1:
                    pilot.estimator.step_attitude(sample[3:], dt_imu)
                    if pilot.armed:
                        pilot.estimator.step(sample[:3], dt_imu)
                if pilot.mode == "GROUND":
                    alignment_samples.append(sample)
            if received_fresh:
                last_imu_wall_s = time.monotonic()

            active_gate = int(simulator.active_gate)
            shared["active_gate"] = active_gate
            reported_gate_count = int(simulator.gate_count)
            if len(shared["gates"]):
                shared["gate_count"] = len(shared["gates"])
            elif reported_gate_count:
                shared["gate_count"] = reported_gate_count
            elif vision is not None:
                shared["gate_count"] = len(vision.mapped_gates())
            shared["collisions"] = len(simulator.collisions)
            gates = np.asarray(shared["gates"])
            target_gate = min(active_gate, len(gates))
            while int(shared["crossed_gate"]) < target_gate:
                index = int(shared["crossed_gate"])
                pilot.mark_crossed(gates[index])
                shared["crossed_gate"] = index + 1
                say(f"gate {index + 1}/{len(gates)}")

            if simulator.race_finished and pilot.mode in {"LAUNCH", "RACE"}:
                pilot.finish()

            if pilot.mode == "FINISHED":
                if not finished_disarmed:
                    simulator.send_rates(0.0, 0.0, 0.0, 0.0)
                    simulator.disarm()
                    finished_disarmed = True
                    say(f"complete; {shared['collisions']} collisions")
                publish()
                time.sleep(0.001)
                continue

            if not received_fresh or shared["imu_usec"] is None:
                if (
                    pilot.armed
                    and last_imu_wall_s is not None
                    and time.monotonic() - last_imu_wall_s
                    > pilot.safety.limits.maximum_imu_age_s
                ):
                    command = pilot.kill("IMU stream stale")
                    simulator.send_rates(*command.body_fields, command.throttle)
                    simulator.disarm()
                    publish()
                time.sleep(0.001)
                continue

            if vision:
                vision.sync(
                    pilot.estimator,
                    int(shared["imu_usec"]),
                    active_gate,
                    pilot.mode == "RACE",
                )

            if not race_status_ready:
                if (
                    time.monotonic() >= reset_ready_wall_s
                    and active_gate == 0
                    and not simulator.race_finished
                ):
                    race_status_ready = True
                else:
                    simulator.send_rates(0.0, 0.0, 0.0, 0.0)
                    time.sleep(0.001)
                    continue

            usec = int(shared["imu_usec"])
            if next_control_usec is None:
                next_control_usec = float(usec)
            if float(usec) + 1e-6 < next_control_usec:
                continue
            ticks = max(1, int((float(usec) - next_control_usec) // period_usec) + 1)
            next_control_usec += ticks * period_usec
            dt = min(0.05, ticks / pilot.control_hz)
            imu_age_s = max(0.0, time.monotonic() - float(last_imu_wall_s))
            state = pilot.state(usec * 1e-6, shared["imu_rate"], imu_age_s)

            if pilot.mode == "GROUND":
                expected = reported_gate_count or None
                mapped = (
                    np.empty((0, 3))
                    if vision is None
                    else vision.mapped_gates(expected)
                )
                if vision is None:
                    pilot.status_message = "vision is required to map the course"
                elif reported_gate_count and len(mapped) < reported_gate_count:
                    pilot.status_message = f"mapping gates {len(mapped)}/{expected}"
                elif not reported_gate_count and not vision.map_ready():
                    pilot.status_message = f"mapping gates {len(mapped)} (settling)"
                elif len(alignment_samples) < ALIGNMENT_SAMPLES:
                    pilot.status_message = (
                        f"aligning IMU {len(alignment_samples)}/{ALIGNMENT_SAMPLES}"
                    )
                else:
                    gates = build_route(state.position_ned, mapped).gates_ned
                    armed, _ = pilot.try_arm(state, alignment_samples, gates)
                    if armed:
                        shared["gates"] = gates
                        shared["gate_count"] = len(gates)
                        vision.set_known_gates(gates)
                        say(f"mapped {len(gates)} gates from camera")
                        simulator.arm()
                command = pilot.zero()
            else:
                command = pilot.update(state, shared["imu_acceleration"], dt)

            simulator.send_rates(*command.body_fields, command.throttle)
            if pilot.mode == "KILL" and not killed_reset:
                simulator.disarm()
                simulator.reset()
                killed_reset = True
                say("simulator reset after kill")
            publish()

    def guarded_flight_loop() -> None:
        try:
            flight_loop()
        except Exception as error:
            pilot.kill(f"runtime error: {error}")
            simulator.send_rates(0.0, 0.0, 0.0, 0.0)
            simulator.disarm()
            publish()

    worker = threading.Thread(
        target=guarded_flight_loop, daemon=True, name="flight-control"
    )

    def close() -> None:
        stopped.set()
        worker.join(timeout=2.0)
        simulator.disarm()
        if vision:
            vision.close()

    reset_run()
    publish()
    worker.start()
    dashboard = Dashboard(
        read_snapshot,
        None if vision is None else vision.preview,
        request_reset,
        request_kill,
        close,
    )
    dashboard.run()
