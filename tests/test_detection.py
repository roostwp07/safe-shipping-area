"""
Runs the detection pipeline against local video files in place of live RTSP streams.
Useful for testing. Pass --left-video and --right-video to specify footage paths.
"""

import argparse
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import monitor_camera


def main() -> None:
    parser = argparse.ArgumentParser(description="Test person detection on local video files.")
    parser.add_argument("--left-video", required=True, help="Path or URL for the left camera video")
    parser.add_argument("--right-video", required=True, help="Path or URL for the right camera video")
    parser.add_argument("--model", default="models/yolo11n.pt", help="Path to YOLO model weights")
    args = parser.parse_args()

    processes = [
        multiprocessing.Process(
            target=monitor_camera,
            args=("left-camera", args.left_video, args.model),
            kwargs={"eof_is_error": False},
            name="left-camera",
            daemon=True,
        ),
        multiprocessing.Process(
            target=monitor_camera,
            args=("right-camera", args.right_video, args.model),
            kwargs={"eof_is_error": False},
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
