# Robot learning from hand-gesture demonstrations

Engineering thesis (work in progress): a simulated Franka Panda arm (MuJoCo) is teleoperated
with the hand, tracked by an ordinary RGB camera (MediaPipe), the demonstrations are recorded,
and a policy is trained with behavior cloning (an MLP and a small Transformer with action
chunking). The central experiment compares policies learned from a scripted expert and from a
human teleoperating with gestures, with the same number of demonstrations and the same test scenes.

Status: the scripted expert, the MLP pipeline (99-100% success on 100 unseen scenes, see
`results/feasibility/`) and live hand teleoperation work; recording human demonstrations is next.

| Folder | Contents |
|---|---|
| `src/gesture_bc/` | the Python package: simulation, expert, teleoperation, data, models, training, evaluation |
| `configs/` | YAML parameters of the environment, expert, teleoperation and training |
| `scripts/` | command-line entry points (see below) |
| `tests/` | pytest suite (runs in CI without GPU or display) |
| `results/` | small result files and plots committed for reference |

The previous topic of this repository (Gesture-to-Geometry, a browser app) is preserved at the
git tag `v1-gesture-to-geometry`.

## Setup (Windows, PowerShell)

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

# Download the Franka Panda model (~36 MB) and MediaPipe's hand model (~7.8 MB);
# neither is stored in this repository
python scripts/fetch_assets.py

# Check the environment: versions, GPU, MuJoCo viewer, camera
python scripts/check_setup.py

# Drive the robot with the keyboard (arrows: left/right/up/down, W/S: away/towards,
# SPACE: gripper, V: record video)
python scripts/view_env.py

# Scripted-expert demonstrations (100 episodes, DART noise 0.3; --noise-std 0 for clean data)
python scripts/collect_expert.py --videos 3
python scripts/collect_expert.py --noise-std 0
python scripts/inspect_dataset.py data/expert_100_noise0.3.hdf5 --replay 10

# Feasibility experiment: train MLPs on the GPU, evaluate on 100 unseen scenes, plot, videos
python scripts/run_feasibility.py
python scripts/plot_feasibility.py
python scripts/make_comparison_videos.py

# Live demo: the trained network solves new random scenes
python scripts/demo.py

# Hand teleoperation preview (webcam; nothing is recorded): C calibrates, pinch closes the gripper
python scripts/teleop_hand.py --max-steps 0

# Lint, formatting and tests
ruff check .
ruff format --check .
pytest
```

**Laptops with two GPUs (Intel + NVIDIA):** Windows may run Python's OpenGL on the integrated
GPU, where a frame takes ~60 ms instead of ~2 ms and the windows become unusable.
`check_setup.py` detects this. Fix: *Settings > System > Display > Graphics > Add desktop app*,
choose the base `python.exe` of Python 3.12 (`check_setup.py` prints its path), then
*Options > High performance*.

**`ImportError: initialization failed` when importing MuJoCo:** Windows 11 *Smart App Control*
blocks MuJoCo's unsigned libraries (the underlying error reads "An Application Control policy has
blocked this file"). It can only be turned off as a whole: *Windows Security > App & browser
control > Smart App Control settings > Off*.

## Third-party assets

- **Franka Emika Panda model**: from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35/franka_emika_panda)
  (commit `62ab00f`), developed by Franka Emika and derived from the
  [franka_ros](https://github.com/frankaemika/franka_ros) description. Licensed under the
  [Apache License 2.0](https://github.com/google-deepmind/mujoco_menagerie/blob/62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35/franka_emika_panda/LICENSE).
  The model is not redistributed here: `scripts/fetch_assets.py` downloads it, together with its
  license file, from the pinned commit.
- **MediaPipe Hand Landmarker model** (`hand_landmarker.task`, float16, version 1) by Google,
  licensed under the Apache License 2.0 according to its
  [model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf).
  Not redistributed here: `scripts/fetch_assets.py` downloads it from
  [MediaPipe's model storage](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
  and checks its SHA-256.
