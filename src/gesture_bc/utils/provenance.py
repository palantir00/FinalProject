"""Record where a result came from (code version), so every dataset and run is traceable."""

import subprocess
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[3]


def git_commit() -> str:
    """Short hash of the current commit, with "+dirty" if there are uncommitted changes.

    Returns "unknown" outside a git repository (e.g. an installed copy of the package).
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}+dirty" if changes else commit
