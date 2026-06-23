import logging
import multiprocessing
import signal
import sys

import cv2
from ultralytics import YOLO

LEFT_CAMERA_URL = "rtsp://user1:M%40rtinrea2025@10.112.16.150/ch1/0"
RIGHT_CAMERA_URL = "rtsp://user1:M%40rtinrea2025@10.112.16.151/ch1/0"

LOG_FILE = "safety.log"


def _setup_logger(camera: str) -> logging.Logger:
    logger = logging.getLogger(camera)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(LOG_FILE)
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(name)s  %(levelname)s  %(message)s")
        )
        logger.addHandler(handler)
    return logger


def monitor_camera(camera: str, rtsp_url: str) -> None:
    """Run person detection on one camera stream indefinitely."""
    logger = _setup_logger(camera)
    logger.info("Camera process started (url=%s)", rtsp_url)

    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        logger.error("Failed to open stream — check RTSP URL")
        return

    def _shutdown(sig, frame):  # noqa: ANN001
        logger.info("Camera process shutting down")
        cap.release()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed — stream may have dropped")
                break

            results = model(frame, verbose=False)

            person_detected = any(
                int(box.cls[0]) == 0
                for result in results
                for box in result.boxes
            )

            if person_detected:
                logger.warning("PERSON IN FOV — simulating red-light alert")

    finally:
        cap.release()
        logger.info("Camera process stopped")


def main() -> None:
    processes = [
        multiprocessing.Process(
            target=monitor_camera,
            args=("left-camera", LEFT_CAMERA_URL),
            name="left-camera",
            daemon=True,
        ),
        multiprocessing.Process(
            target=monitor_camera,
            args=("right-camera", RIGHT_CAMERA_URL),
            name="right-camera",
            daemon=True,
        ),
    ]

    for p in processes:
        p.start()

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
