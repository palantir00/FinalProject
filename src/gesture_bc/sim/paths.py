"""Filesystem locations of the simulation assets."""

from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Franka Emika Panda model from MuJoCo Menagerie (Apache-2.0). It is ~36 MB of meshes,
# so it is not stored in git; scripts/fetch_assets.py downloads it into this folder.
PANDA_DIR = ASSETS_DIR / "franka_emika_panda"
PANDA_XML = PANDA_DIR / "panda.xml"
PANDA_SCENE_XML = PANDA_DIR / "scene.xml"

# Our pick-and-place scene (table, cube, target, cameras); the Panda is attached in scene.py.
PICK_PLACE_XML = ASSETS_DIR / "pick_place.xml"


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Franka Panda model not found in {PANDA_DIR}.\n"
            "Download it with:  python scripts/fetch_assets.py"
        )
    return path


def require_panda_scene() -> Path:
    """Return the path of Menagerie's Panda demo scene (robot on a floor)."""
    return _require(PANDA_SCENE_XML)


def require_panda_xml() -> Path:
    """Return the path of the Panda robot model itself."""
    return _require(PANDA_XML)
