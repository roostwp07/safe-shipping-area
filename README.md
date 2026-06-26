# Safe Shipping Area

## Setup

### 1. Install uv

**Linux / macOS**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # or restart your shell
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> Full install docs: https://docs.astral.sh/uv/getting-started/installation/

### 2. Configure and run

```bash
cp .env.example .env   # then fill in your RTSP credentials
uv sync
uv run main.py
```

## Testing with local video files

Run detection against the MP4 files in the `videos/` directory instead of live RTSP streams:

```bash
uv run tests/test_detection.py --left-video videos/left.mp4 --right-video videos/right.mp4
```

Use `--model` to specify a different set of weights (defaults to `models/yolo11n.pt`):

```bash
uv run tests/test_detection.py --left-video videos/left.mp4 --right-video videos/right.mp4 --model models/yolo11s.pt
```

Detection output is written to `safety.log`. Press `Ctrl+C` to stop early.

## Project Structure

```text
safe-shipping-area/
├── models/            # YOLO model weights (gitignored)
├── videos/            # Test video files (gitignored)
├── main.py            # Entry point — spawns one process per camera
├── test_detection.py  # Run detection on local video files instead of live streams
├── safety.log         # Detection log output (gitignored)
├── pyproject.toml     # Dependencies (managed with uv)
└── README.md
```