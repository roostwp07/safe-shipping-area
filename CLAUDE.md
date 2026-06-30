# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A computer vision safety system for the shipping area of a manufacturing plant. Two IP cameras — **left** and **right**, positioned to give full coverage of the area — stream video over RTSP. The system uses YOLOv11 with ByteTrack to detect and track people and forklifts, and triggers an alert when their trajectories indicate an imminent collision.

**Version 1:** logged a warning every time a person appeared in the FOV, simulating the red-light alert.

**Version 2 (current):** full collision risk detection. Each frame, velocity vectors are computed per tracked object via a Kalman-filtered position history. For every (forklift, person) pair, time-to-closest-approach is computed; if it falls below `TTC_THRESHOLD` seconds and the predicted closest distance falls below `SAFETY_RADIUS`, a `WARNING` is logged and the bounding boxes are highlighted red.

**Planned:** perspective transform calibration per camera (see `tracker.py: ViewTransformer`) to convert pixel distances to real-world meters, enabling accurate TTC in seconds and distance in meters instead of pixel units.

## Architecture

Two independent OS processes run in parallel (one per camera), both spawned by `main()`:

```
main()
 ├─ left-camera  process → monitor_camera("left-camera",  LEFT_CAMERA_URL)
 └─ right-camera process → monitor_camera("right-camera", RIGHT_CAMERA_URL)
```

Each process:
1. Opens the RTSP stream with OpenCV
2. Reads frames in a tight loop
3. Runs `model.track(frame, persist=True)` — YOLO inference + ByteTrack, producing persistent IDs
4. Feeds results into `ObjectTracker.update()` to maintain per-ID Kalman-filtered positions and velocity history
5. Calls `CollisionDetector.check()` to evaluate all (forklift, person) pairs
6. Logs a `WARNING` to `safety.log` for any flagged pair

## Key Files

- **`main.py`** — process entrypoint and frame loop
- **`tracker.py`** — all tracking and collision logic:
  - `ViewTransformer` — perspective homography from pixel coords to real-world meters (not yet wired up; requires per-camera floor calibration)
  - `ObjectTracker` — per-ID Kalman filter (4-state: x, y, vx, vy), rolling velocity history
  - `CollisionDetector` — time-to-closest-approach check for all (forklift, person) pairs
  - `DEFAULT_FORKLIFT_CLASSES` — COCO proxies (car, motorcycle, bus, truck) used until a fine-tuned forklift model is available

## Thresholds (tunable in `main.py`)

| Constant | Default | Meaning |
|---|---|---|
| `TTC_THRESHOLD` | `3.0` | Flag if time-to-closest-approach is below this many seconds |
| `SAFETY_RADIUS` | `1.5` | Flag if predicted (or current) separation is below this (meters once calibrated, pixels until then) |

## Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management with Python 3.13.

```bash
uv sync          # install dependencies into .venv
uv run main.py   # start both camera processes
```

Stop with `Ctrl+C` — both child processes are terminated cleanly.

## Key Details

- **Model**: `yolo11n.pt` (nano variant, downloaded automatically by Ultralytics on first run)
- **Forklift classes**: COCO proxies until fine-tuned — set `DEFAULT_FORKLIFT_CLASSES` in `tracker.py` or pass a custom `frozenset` to `ObjectTracker`
- **Logging**: all output goes to `safety.log` (not stdout); format is `timestamp  camera-name  level  message`
- **Camera URLs**: read from `.env` as `LEFT_CAMERA_URL` / `RIGHT_CAMERA_URL`
- Child processes are `daemon=True` so they die automatically if the parent is killed unexpectedly
