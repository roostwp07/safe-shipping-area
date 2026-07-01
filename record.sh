#!/usr/bin/env bash
set -euo pipefail
export $(grep -v '^#' .env | xargs)

TS=$(date +%Y%m%d_%H%M%S)

case "${1:-both}" in
  left)
    ffmpeg -rtsp_transport tcp -i "$LEFT_CAMERA_URL" -c copy "left_${TS}.mp4"
    ;;
  right)
    ffmpeg -rtsp_transport tcp -i "$RIGHT_CAMERA_URL" -c copy "right_${TS}.mp4"
    ;;
  both)
    ffmpeg -rtsp_transport tcp -i "$LEFT_CAMERA_URL" -c copy "left_${TS}.mp4" &
    ffmpeg -rtsp_transport tcp -i "$RIGHT_CAMERA_URL" -c copy "right_${TS}.mp4"
    ;;
  *)
    echo "Usage: $0 [left|right|both]" >&2
    exit 1
    ;;
esac
