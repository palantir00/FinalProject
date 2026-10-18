"""Filesystem locations of the simulation assets."""

from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Franka Emika Panda model from MuJoCo Menagerie (Apache-2.0). It is ~36 MB of meshes,
# so it is not stored in git; scripts/fetch_assets.py downloads it into this folder.
PANDA_DIR = ASSETS_DIR / "franka_emika_panda"
PANDA_SCENE_XML = PANDA_DIR / "scene.xml"


def require_panda_scene() -> Path:
    """Return the path of the Panda scene, or explain how to download the model."""
    if not PANDA_SCENE_XML.is_file():
        raise FileNotFoundError(
            f"Franka Panda model not found in {PANDA_DIR}.\n"
            "Download it with:  python scripts/fetch_assets.py"
        )
    return PANDA_SCENE_XML
