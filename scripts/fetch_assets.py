"""Download the third-party models: the Franka Emika Panda and MediaPipe's hand model.

Neither is stored in this repository.
    Panda (~36 MB of meshes): exactly one directory from a pinned MuJoCo Menagerie commit, so
        every machine (and CI) simulates the same robot. Each file is checked against the
        hash git stores for it in that commit.
    Hand model (~7.8 MB): a pinned version of MediaPipe's Hand Landmarker, checked against
        its SHA-256, so every machine tracks hands with the same network.
A corrupted or modified download is detected and rejected.

Usage (repository root, .venv activated):
    python scripts/fetch_assets.py           # download what is missing or outdated
    python scripts/fetch_assets.py --force   # download again
"""

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

from gesture_bc.sim.paths import PANDA_DIR
from gesture_bc.teleop.paths import HAND_MODEL

REPO = "google-deepmind/mujoco_menagerie"
# Pinned commit (2026-10-14). Change it only on purpose: a different model version can
# change the robot's dynamics and therefore the experiment results.
COMMIT = "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35"
MODEL_DIR = "franka_emika_panda"
# Written next to the downloaded model, so we know which commit is on disk.
COMMIT_MARKER = ".menagerie_commit"

# MediaPipe Hand Landmarker, float16, version 1 (published 2023-04-26; "latest" is the same
# file today, but a pinned version cannot change under us).
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
    "float16/1/hand_landmarker.task"
)
HAND_MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def get_json(url: str) -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    # GitHub Actions provides a token; it avoids the anonymous API rate limit (60 requests/h).
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def list_model_files() -> list[dict]:
    """Return git tree entries ({"path", "sha", ...}) of all files in MODEL_DIR at COMMIT."""
    api = f"https://api.github.com/repos/{REPO}/git/trees"
    root = get_json(f"{api}/{COMMIT}")
    model_tree_sha = next(entry["sha"] for entry in root["tree"] if entry["path"] == MODEL_DIR)
    model_tree = get_json(f"{api}/{model_tree_sha}?recursive=1")
    return [entry for entry in model_tree["tree"] if entry["type"] == "blob"]


def git_blob_sha1(content: bytes) -> str:
    """Hash file content the way git does ("blob <size>\\0<content>"), to compare with the tree."""
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def download_model(destination: Path) -> None:
    files = list_model_files()
    # Download into a separate folder first, so an interrupted download never looks
    # like a complete model.
    partial = destination.with_name(destination.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    for i, entry in enumerate(files, start=1):
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{MODEL_DIR}/{entry['path']}"
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        if git_blob_sha1(content) != entry["sha"]:
            raise RuntimeError(f"Checksum mismatch for {entry['path']}")
        target = partial / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        print(f"  [{i:2d}/{len(files)}] {entry['path']}")
    (partial / COMMIT_MARKER).write_text(COMMIT + "\n")
    shutil.rmtree(destination, ignore_errors=True)
    partial.rename(destination)


def installed_commit(model_dir: Path) -> str | None:
    marker = model_dir / COMMIT_MARKER
    return marker.read_text().strip() if marker.is_file() else None


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def download_hand_model(destination: Path) -> None:
    with urllib.request.urlopen(HAND_MODEL_URL, timeout=60) as response:
        content = response.read()
    if sha256(content) != HAND_MODEL_SHA256:
        raise RuntimeError("Checksum mismatch for the hand model")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


def hand_model_present() -> bool:
    return HAND_MODEL.is_file() and sha256(HAND_MODEL.read_bytes()) == HAND_MODEL_SHA256


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Panda and hand models.")
    parser.add_argument("--force", action="store_true", help="download even if already present")
    args = parser.parse_args()

    if not args.force and installed_commit(PANDA_DIR) == COMMIT:
        print(f"Panda model already present (Menagerie commit {COMMIT[:7]}): {PANDA_DIR}")
    else:
        print(f"Downloading {MODEL_DIR} from {REPO} @ {COMMIT[:7]} ...")
        download_model(PANDA_DIR)
        print(f"Done: {PANDA_DIR}")

    if not args.force and hand_model_present():
        print(f"Hand model already present: {HAND_MODEL}")
    else:
        print("Downloading the MediaPipe hand model ...")
        download_hand_model(HAND_MODEL)
        print(f"Done: {HAND_MODEL}")


if __name__ == "__main__":
    main()
