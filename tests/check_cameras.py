import os
import sys

import cv2
from dotenv import load_dotenv

load_dotenv()

CAMERAS = {
    "left-camera": os.environ["LEFT_CAMERA_URL"],
    "right-camera": os.environ["RIGHT_CAMERA_URL"],
}


def check_camera(name: str, url: str) -> bool:
    print(f"{name}: connecting...", end=" ", flush=True)
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("FAIL (could not open stream)")
        return False
    ret, _ = cap.read()
    cap.release()
    if not ret:
        print("FAIL (opened but could not read a frame)")
        return False
    print("OK")
    return True


if __name__ == "__main__":
    results = [check_camera(name, url) for name, url in CAMERAS.items()]
    sys.exit(0 if all(results) else 1)
