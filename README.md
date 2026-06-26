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