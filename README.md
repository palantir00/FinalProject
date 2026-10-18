# Gesture-to-Geometry (Gesture-to-Geometry)

Projekt „Gesture-to-Geometry”: tworzenie i modyfikacja proceduralnych brył 3D gestami dłoni (kamera) + natychmiastowa analiza geometryczna (pole, objętość) oraz tryby Measure/Slice.

Repozytorium jest zainicjowane zgodnie ze specyfikacją z pracy inżynierskiej:
- **Struktura**: patrz `docs/SPEC_EXTRACTED.txt`, sekcja „5) Repozytorium – struktura”.
- **Protokół gestów**: patrz `docs/SPEC_EXTRACTED.txt`, sekcja „9) Protokół gestów”.

## Uruchomienie (frontend)

Frontend (Vite + React + TypeScript) znajduje się w folderze `app/`.

1) Zainstaluj Node.js (LTS) + npm  
2) W katalogu `app/`:

```bash
npm install
npm run dev
```

## Szybka nawigacja po kodzie

- `app/src/cv/HandTracker.ts`: pipeline wideo + MediaPipe Hands (landmarki)
- `app/src/cv/filters/oneEuro.ts`: One-Euro Filter (stabilizacja sygnału)
- `app/src/gestures/*`: mapping landmarków → gesty + FSM trybów
- `app/src/three/SceneManager.ts`: Three.js scena + pętla render
- `app/src/geometry/analytics.ts`: pole/objętość (sześcian, kula, walec)
- `app/src/ui/*`: HUD (autolabels) + tutorial overlay (MVP)

---

## Robot learning from hand-gesture demonstrations (work in progress)

New thesis direction: a simulated Franka Panda arm (MuJoCo) is teleoperated with hand gestures
tracked by an RGB camera (MediaPipe), the demonstrations are recorded, and a policy is trained
with behavior cloning (MLP and a small Transformer with action chunking).
The Python code lives in `src/gesture_bc/`.

### Setup (Windows, PowerShell)

Requires Python 3.11 or 3.12 and, for GPU training, an NVIDIA driver supporting CUDA 13.0.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# PyTorch with CUDA comes from PyTorch's own package index
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130

# Exact versions used in the experiments, then this project in editable mode
pip install -r requirements-lock.txt
pip install -e . --no-deps

# Download the Franka Panda model (~36 MB, not stored in this repository)
python scripts/fetch_assets.py

# Check the environment: versions, GPU, MuJoCo viewer, camera
python scripts/check_setup.py

# Drive the robot with the keyboard (arrows: left/right/up/down, W/S: away/towards,
# SPACE: gripper, V: record video)
python scripts/view_env.py

# Lint and tests
ruff check .
pytest
```

**Laptops with two GPUs (Intel + NVIDIA):** Windows may run Python's OpenGL on the integrated
GPU, where a frame takes ~60 ms instead of ~2 ms and the windows become unusable.
`check_setup.py` detects this. Fix: *Settings > System > Display > Graphics > Add desktop app*,
choose the base `python.exe` of Python 3.12 (`check_setup.py` prints its path), then
*Options > High performance*.

### Third-party assets

- **Franka Emika Panda model**: from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35/franka_emika_panda)
  (commit `62ab00f`), developed by Franka Emika and derived from the
  [franka_ros](https://github.com/frankaemika/franka_ros) description. Licensed under the
  [Apache License 2.0](https://github.com/google-deepmind/mujoco_menagerie/blob/62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35/franka_emika_panda/LICENSE).
  The model is not redistributed here: `scripts/fetch_assets.py` downloads it, together with its
  license file, from the pinned commit.
