# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A computer vision safety system for the shipping area of a manufacturing plant. Two IP cameras — **left** and **right**, positioned to give full coverage of the area — stream video over RTSP. The system uses YOLOv11 to detect people and forklifts and is intended to warn when a collision risk is detected.

**Version 1 (current):** validates that the YOLO model can reliably detect people. A warning is logged to `safety.log` every time a person appears in either camera's field of view, simulating the red-light alert that will be wired to physical hardware in a later version.

**Planned (v2+):** detect forklifts alongside pedestrians, compute direction vectors for each, and trigger the alert only when trajectories indicate an imminent collision.

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
3. Runs YOLO inference on every frame
4. Logs a `WARNING` to `safety.log` when COCO class 0 (person) is detected

## Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management with Python 3.13.

```bash
uv sync          # install dependencies into .venv
uv run main.py   # start both camera processes
```

Stop with `Ctrl+C` — both child processes are terminated cleanly.

## Key Details

- **Model**: `yolo11n.pt` (nano variant, downloaded automatically by Ultralytics on first run)
- **COCO class 0** = person — the only class checked in v1
- **Logging**: all output goes to `safety.log` (not stdout); format is `timestamp  camera-name  level  message`
- **Camera URLs**: `LEFT_CAMERA_URL` / `RIGHT_CAMERA_URL` constants at the top of `main.py`
- Child processes are `daemon=True` so they die automatically if the parent is killed unexpectedly
