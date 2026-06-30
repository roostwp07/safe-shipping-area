import argparse
import logging
import multiprocessing
import os
import signal
import sys

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

from tracker import CollisionDetector, ObjectTracker

load_dotenv()

LEFT_CAMERA_URL = os.environ["LEFT_CAMERA_URL"]
RIGHT_CAMERA_URL = os.environ["RIGHT_CAMERA_URL"]

LOG_FILE = "safety.log"

# Tunable thresholds — set in real-world meters once perspective calibration is done,
# or in pixels until then.
TTC_THRESHOLD = 3.0   # seconds
SAFETY_RADIUS = 1.5   # meters (or pixels without a ViewTransformer)

_FALLBACK_FPS = 30.0


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


def _annotate(frame, results, flagged_ids: set[int]) -> None:
    CLASS_COLORS = {0: (0, 0, 255)}      # person → red
    FORKLIFT_COLOR = (0, 165, 255)       # forklift → orange
    DANGER_COLOR = (0, 0, 255)

    for box in results[0].boxes:
        if box.id is None:
            continue
        cls = int(box.cls[0])
        obj_id = int(box.id)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == 0:
            color = DANGER_COLOR if obj_id in flagged_ids else CLASS_COLORS[0]
            label = f"person {obj_id}"
        else:
            color = DANGER_COLOR if obj_id in flagged_ids else FORKLIFT_COLOR
            label = f"vehicle {obj_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def monitor_camera(
    camera: str,
    rtsp_url: str,
    model_path: str = "models/yolo11n.pt",
    eof_is_error: bool = True,
    display: bool = True,
) -> None:
    logger = _setup_logger(camera)
    logger.info("Camera process started (url=%s)", rtsp_url)

    model = YOLO(model_path)
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        logger.error("Failed to open stream — check RTSP URL")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or _FALLBACK_FPS
    # Pass a ViewTransformer here once you have floor calibration points for this camera.
    obj_tracker = ObjectTracker(fps=fps)
    collision_detector = CollisionDetector(fps=fps, ttc_threshold=TTC_THRESHOLD, safety_radius=SAFETY_RADIUS)

    def _shutdown(sig, frame):  # noqa: ANN001
        logger.info("Camera process shutting down")
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

            results = model.track(frame, persist=True, verbose=False)
            obj_tracker.update(results)
            flagged_pairs = collision_detector.check(obj_tracker)

            if flagged_pairs:
                for forklift_id, person_id in flagged_pairs:
                    logger.warning(
                        "COLLISION RISK — forklift %d / person %d", forklift_id, person_id
                    )

            if display:
                flagged_ids = {i for pair in flagged_pairs for i in pair}
                _annotate(frame, results, flagged_ids)
                cv2.imshow(camera, frame)
                cv2.waitKey(1)

    finally:
        cap.release()
        if display:
            try:
                cv2.destroyWindow(camera)
            except cv2.error:
                pass
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
