import argparse
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


def monitor_camera(camera: str, rtsp_url: str, model_path: str = "models/yolo11n.pt", eof_is_error: bool = True, display: bool = True) -> None:
    """Run person detection on one camera stream indefinitely."""
    logger = _setup_logger(camera)
    logger.info("Camera process started (url=%s)", rtsp_url)

    model = YOLO(model_path)
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        logger.error("Failed to open stream — check RTSP URL")
        return

    def _shutdown(sig, frame):  # noqa: ANN001
        logger.info("Camera process shutting down")
        cap.release()
        if display:
            cv2.destroyWindow(camera)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if eof_is_error:
                    logger.warning("Frame read failed — stream may have dropped")
                else:
                    logger.info("End of video file")
                break

            results = model(frame, verbose=False)

            person_detected = False
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:
                        person_detected = True
                        if display:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, "person", (x1, y1 - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if display:
                cv2.imshow(camera, frame)
                cv2.waitKey(1)

            if person_detected:
                logger.warning("PERSON IN FOV — simulating red-light alert")

    finally:
        cap.release()
        if display:
            cv2.destroyWindow(camera)
        logger.info("Camera process stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run person detection on live camera streams.")
    parser.add_argument("--model", default="models/yolo11n.pt", help="Path to YOLO model weights")
    args = parser.parse_args()

    processes = [
        multiprocessing.Process(
            target=monitor_camera,
            args=("left-camera", LEFT_CAMERA_URL, args.model),
            name="left-camera",
            daemon=True,
        ),
        multiprocessing.Process(
            target=monitor_camera,
            args=("right-camera", RIGHT_CAMERA_URL, args.model),
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
