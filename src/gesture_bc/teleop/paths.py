"""Filesystem locations of the teleoperation assets."""

from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# MediaPipe's hand model (float16, version 1, ~7.8 MB, Apache-2.0). It is not stored in git;
# scripts/fetch_assets.py downloads it into this folder and checks its SHA-256.
HAND_MODEL = ASSETS_DIR / "hand_landmarker.task"
