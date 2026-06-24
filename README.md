# Safe Shipping Area

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