from collections import deque
import struct
import threading
import unittest

import numpy as np

from controller.estimator import Estimator
from controller.mission import (
    MissionSettings,
    PhaseSettings,
    PhaseTracker,
    RaceMission,
    build_route,
)
from controller.pilot import RacePilot
from controller.state import FlatReference, StateEstimate
from controller.tracking import ControllerSettings
from gui import project
from physics.model import load_settings, load_vehicle
from radiomaster.link import SimulatorLink, TRACK_DATA
from radiomaster.rates import (
    betaflight_rate_rps,
    inverse_betaflight_rate,
    load_rates,
)
from vision.detector import Detector
from vision import camera
from vision.pipeline import Vision


COURSE = np.array(
    [
        [23.297967910766602, 0.39990234375, -0.031958006322383881],
        [46.893749237060547, 2.4999902248382568, 5.0680418014526367],
        [74.59375, -1.2000097036361694, 13.668041229248047],
        [111.49374389648438, 5.099989891052246, 24.56804084777832],
        [135.49374389648438, 0.7999902367591858, 25.355653762817383],
        [159.19374084472656, 4.399990081787109, 25.968040466308594],
    ]
)


class CoreTest(unittest.TestCase):
    def test_tk_plot_projection(self):
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_allclose(project(points, (0, 2), True), [[1, -3], [4, -6]])

    def test_rate_mapping_round_trip(self):
        rates = load_rates()
        for settings in (rates.roll, rates.pitch, rates.yaw):
            requested = betaflight_rate_rps(0.63, settings)
            stick, achieved, saturated = inverse_betaflight_rate(requested, settings)
            self.assertAlmostEqual(stick, 0.63, places=8)
            self.assertAlmostEqual(achieved, requested, places=8)
            self.assertFalse(saturated)

    def test_perceived_course_is_ordered_and_extended(self):
        position = np.array([4.0, -2.0, 1.0])
        gates = COURSE
        route = build_route(position, gates)
        self.assertEqual(route.gate_count, 6)
        np.testing.assert_allclose(route.waypoints_ned[0], position)
        np.testing.assert_allclose(route.waypoints_ned[1:7], gates)
        self.assertAlmostEqual(
            np.linalg.norm(route.waypoints_ned[-1] - gates[-1]), 10.0
        )

    def test_route_uses_the_older_nearest_gate_order(self):
        gates = np.array([[20.0, 0.0, 0.0], [5.0, 0.0, 0.0], [12.0, 0.0, 0.0]])
        route = build_route(np.zeros(3), gates)
        np.testing.assert_allclose(route.gates_ned[:, 0], [5.0, 12.0, 20.0])

    def test_orange_detector_finds_a_gate_frame(self):
        import cv2

        image = np.zeros((360, 360, 3), dtype=np.uint8)
        cv2.rectangle(image, (90, 70), (270, 250), (0, 120, 255), 20)
        detections, polygons = Detector().infer(image)
        self.assertEqual(len(detections), 1)
        self.assertEqual(len(polygons), 1)

    def test_perceived_course_plan_is_feasible(self):
        settings = load_settings()
        vehicle = load_vehicle()
        estimator = Estimator()
        state = StateEstimate(
            0.0,
            estimator.p,
            estimator.v,
            estimator.R,
            np.zeros(3),
        )
        mission = RaceMission(
            vehicle,
            load_rates(),
            MissionSettings(**settings["mission"]),
            ControllerSettings(**settings["controller"]),
        )
        mission.prepare(state, COURSE)
        self.assertTrue(mission.plan.feasibility.feasible)
        self.assertGreater(len(mission.plan.queue), 100)
        mission.mark_crossed()
        self.assertEqual(mission.crossed_count, 1)
        self.assertGreaterEqual(mission.phase.phase_s, mission.plan.trajectory.durations_s[0])

    def test_course_fixture_matches_the_older_reference_queue(self):
        settings = load_settings()
        vehicle = load_vehicle()
        estimator = Estimator()
        state = StateEstimate(
            0.0,
            estimator.p,
            np.array([6.0, -2.0, 1.0]),
            estimator.R,
            np.zeros(3),
        )
        mission = RaceMission(
            vehicle,
            load_rates(),
            MissionSettings(**settings["mission"]),
            ControllerSettings(**settings["controller"]),
        )
        mission.prepare(state, COURSE)
        plan = mission.plan
        self.assertAlmostEqual(plan.queue.duration_s, 18.697157842800106, places=9)
        first = plan.queue.sample(0.0)
        np.testing.assert_allclose(first.reference.velocity_ned, np.zeros(3), atol=1e-12)
        quarter = plan.queue.sample(plan.queue.duration_s * 0.25)
        np.testing.assert_allclose(
            quarter.reference.position_ned,
            [34.67556400047655, 1.657207941883355, 1.7650300591414028],
            atol=1e-9,
        )

    def test_phase_rate_uses_the_older_direct_governor(self):
        phase = PhaseTracker(PhaseSettings())
        reference = FlatReference(
            0.0,
            0,
            np.array([10.0, 0.0, 0.0]),
            np.array([5.0, 0.0, 0.0]),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            0.0,
        )
        state = StateEstimate(0.0, reference.position_ned, np.zeros(3), np.eye(3), np.zeros(3))
        phase.advance(0.1, state, reference, 20.0)
        self.assertEqual(phase.rate, 1.0)
        behind = StateEstimate(0.0, np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3))
        phase.advance(0.1, behind, reference, 20.0)
        self.assertEqual(phase.rate, 0.1)

    def test_static_imu_average_starts_launch(self):
        pilot = RacePilot()
        pilot.validation = (True, "test settings")
        pitch = np.radians(-20.0)
        sample = np.array(
            [9.81 * np.sin(pitch), 0.0, -9.81 * np.cos(pitch), 0.0, 0.0, 0.0]
        )
        armed, _ = pilot.try_arm(
            pilot.state(0.0, np.zeros(3)),
            np.repeat(sample[None, :], 180, axis=0),
            COURSE,
        )
        self.assertTrue(armed)
        self.assertEqual(pilot.mode, "LAUNCH")
        snapshot = pilot.snapshot()
        self.assertGreater(len(snapshot["path"]), len(pilot.mission.route.waypoints_ned))
        np.testing.assert_allclose(
            snapshot["path"][0], pilot.mission.plan.trajectory.positions_ned[0]
        )
        np.testing.assert_allclose(
            snapshot["path"][-1], pilot.mission.plan.trajectory.positions_ned[-1]
        )

    def test_known_gate_camera_update_is_bounded(self):
        estimator = Estimator()
        estimator.release()
        before = estimator.p.copy()
        used = estimator.fuse_known_gate(
            np.array([10.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            1.0,
            0.1,
        )
        self.assertTrue(used)
        self.assertLess(estimator.p[1], before[1])
        self.assertLessEqual(np.linalg.norm(estimator.p - before), 0.08 + 1e-12)

    def test_vision_associates_gate_by_bearing_and_range(self):
        gate = np.array([20.0, 2.0, -1.0])
        direction = gate / np.linalg.norm(gate)
        pose = (0, np.eye(3), np.zeros(3), (0.2, np.radians(2.0)))
        match = Vision._match(
            gate,
            pose,
            [
                (direction, 5.0, 0.95),
                (np.array([0.7, -0.7, 0.0]), 20.0, 0.95),
                (direction, np.linalg.norm(gate), 0.95),
            ],
        )
        self.assertIsNotNone(match)
        np.testing.assert_allclose(match[1], direction)

    def test_box_size_provides_metric_range(self):
        _, distance = camera.ray(camera.CX, camera.CY, 54.0, 40.0)
        self.assertAlmostEqual(distance, camera.FX * camera.GATE_SIZE_M / 54.0)

    def test_vision_builds_a_stable_map_from_repeated_boxes(self):
        vision = Vision.__new__(Vision)
        vision.lock = threading.Lock()
        vision._map_tracks = []
        vision._map_stable_frames = 0
        pose = (0, np.eye(3), np.zeros(3), (0.2, np.radians(2.0)))
        gates = np.array([[20.0, 1.0, -2.0], [48.0, -3.0, 4.0]])
        for scale in (0.98, 1.02, 1.00, 0.99, 1.01, 1.00, 1.02, 0.98):
            detections = [
                (gate / np.linalg.norm(gate), np.linalg.norm(gate) * scale, 0.95)
                for gate in gates
            ]
            vision._map_frame(pose, detections)
        mapped = vision.mapped_gates(2)
        mapped = mapped[np.argsort(mapped[:, 0])]
        np.testing.assert_allclose(mapped, gates, atol=0.1)

        self.assertFalse(vision.map_ready())
        for _ in range(5):
            vision._map_frame(pose, detections)
        self.assertTrue(vision.map_ready())

    def test_perceived_gates_use_live_observations(self):
        vision = Vision.__new__(Vision)
        vision.lock = threading.Lock()
        vision._known_gates = np.array([[10.0, 0.0, 0.0]])
        vision._observed_gates = [
            deque([np.array([10.0, 1.0, -0.5])], maxlen=30)
        ]
        np.testing.assert_allclose(
            vision.perceived_gates(), [[10.0, 1.0, -0.5]]
        )

    def test_distant_pixel_quantization_cannot_collapse_gate_spacing(self):
        vision = Vision.__new__(Vision)
        vision.lock = threading.Lock()
        positions = np.array(
            [
                [20.0, 0.0, 0.0],
                [44.0, 1.0, 2.0],
                [70.0, -1.0, 6.0],
                [100.0, 4.0, 12.0],
                [108.0, 0.0, 11.0],
                [150.0, 3.0, 16.0],
            ]
        )
        vision._map_tracks = [
            deque([position.copy() for _ in range(8)], maxlen=60)
            for position in positions
        ]
        mapped = vision.mapped_gates(6)
        minimum = np.linalg.norm(mapped[1] - mapped[0])
        spacing = np.linalg.norm(np.diff(mapped, axis=0), axis=1)
        self.assertTrue(np.all(spacing >= minimum - 1e-9))

    def test_vision_sync_applies_the_queued_active_gate_correction(self):
        vision = Vision.__new__(Vision)
        vision.lock = threading.Lock()
        vision._known_gates = np.array([[10.0, 0.0, 0.0]])
        vision._active_gate = 0
        vision._poses = deque(maxlen=8)
        vision._corrections = deque(
            [(0, np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), 1.0, 0.1)],
            maxlen=8,
        )
        vision._clock = type("Clock", (), {"observe": lambda *_: None})()
        vision._last_sync_usec = None
        vision.camera = type("Camera", (), {"latest": lambda *_: None})()
        vision.corrections = 0
        estimator = Estimator()
        estimator.release()
        vision.sync(estimator, 1000, 0, True)
        self.assertEqual(vision.corrections, 1)
        self.assertLess(estimator.p[1], 0.0)

    def test_track_transfer_supplies_only_gate_count(self):
        link = SimulatorLink.__new__(SimulatorLink)
        link.gate_count = 0
        link._track_chunks = {17: {}}
        link._expected_track_chunks = {17: 1}
        message = type("Packet", (), {"seqnr": 0})()
        coordinates_that_must_not_be_used = struct.pack("<Hfffffffff", 8, *range(9))
        raw = struct.pack("<BH", TRACK_DATA, 17) + coordinates_that_must_not_be_used
        link._receive_track_chunk(message, raw)
        self.assertEqual(link.gate_count, 8)

    def test_commands_send_all_seven_mavlink_parameters(self):
        calls = []
        mav = type("Mav", (), {"command_long_send": lambda _, *args: calls.append(args)})()
        link = SimulatorLink.__new__(SimulatorLink)
        link.conn = type(
            "Connection", (), {"mav": mav, "target_system": 1, "target_component": 2}
        )()
        link.mavutil = type(
            "Mavutil", (), {"mavlink": type("Mavlink", (), {"MAV_CMD_COMPONENT_ARM_DISARM": 400})}
        )()
        link.reset()
        link.arm()
        self.assertEqual(calls[0], (1, 2, 31000, 0, 0, 0, 0, 0, 0, 0, 0))
        self.assertEqual(calls[1], (1, 2, 400, 0, 1, 0, 0, 0, 0, 0, 0))

    def test_race_status_supplies_progress_without_a_map(self):
        link = SimulatorLink.__new__(SimulatorLink)
        link.active_gate = 0
        link.race_finished = False
        link.race_status_updates = 0
        raw = struct.pack("<BQqqIq", 1, 100, 20, -1, 3, 12)
        link._receive_race_status(raw)
        self.assertEqual(link.active_gate, 3)
        self.assertFalse(link.race_finished)
        self.assertEqual(link.race_status_updates, 1)


if __name__ == "__main__":
    unittest.main()
