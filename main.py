from controller.runtime import fly
from radiomaster.link import SimulatorLink


MAVLINK_HOST = "127.0.0.1"
MAVLINK_PORT = 14550
CAMERA_PORT = 5600
DETECTOR = "orange"  # Use "yolo" to compare the optional detector.


def main() -> None:
    fly(
        SimulatorLink(MAVLINK_HOST, MAVLINK_PORT),
        camera_port=CAMERA_PORT,
        detector=DETECTOR,
    )


if __name__ == "__main__":
    main()
