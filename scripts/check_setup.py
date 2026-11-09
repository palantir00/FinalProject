"""Check that the development environment is ready.

Five checks, from non-interactive to interactive:
  1. versions   - Python and package versions, MediaPipe Tasks API.
  2. pytorch    - is the CUDA GPU visible, and does it compute the same result as the CPU?
  3. mujoco     - load the Franka Panda, measure simulation speed, open the viewer window.
  4. rendering  - offscreen rendering speed, and which GPU OpenGL runs on.
  5. camera     - open the webcam with OpenCV and show a live preview.

Usage (repository root, .venv activated):
    python scripts/check_setup.py              # all checks (opens two windows)
    python scripts/check_setup.py --no-gui     # no windows: checks 1-4, MuJoCo without viewer
    python scripts/check_setup.py --camera 1   # use another camera index
"""

import argparse
import platform
import sys
import time
from importlib.metadata import version

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import torch

from gesture_bc.sim.paths import require_panda_scene
from gesture_bc.utils.device import resolve_device

PACKAGES = [
    "torch",
    "mujoco",
    "numpy",
    "mediapipe",
    "opencv-contrib-python",
    "h5py",
    "pyyaml",
    "matplotlib",
    "gymnasium",
    "imageio",
]

GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0


# --- 1. versions --------------------------------------------------------------------------


def check_versions() -> bool:
    print(f"  Python {platform.python_version()}  ({sys.executable})")
    for name in PACKAGES:
        print(f"  {name:22s} {version(name)}")

    if sys.version_info[:2] not in [(3, 11), (3, 12)]:
        print("  Python must be 3.11 or 3.12 (MediaPipe compatibility).")
        return False

    # Imported here because mediapipe is slow to import and only this check needs it.
    from mediapipe.tasks.python import vision

    has_hand_landmarker = hasattr(vision, "HandLandmarker")
    print(f"  MediaPipe Tasks API (HandLandmarker) available: {has_hand_landmarker}")
    return has_hand_landmarker


# --- 2. pytorch + cuda --------------------------------------------------------------------


def time_matmul(a: torch.Tensor, b: torch.Tensor, repeats: int = 10) -> float:
    """Average time of a @ b in milliseconds."""
    on_gpu = a.device.type == "cuda"
    a @ b  # warm-up: the first CUDA call also loads the kernels
    if on_gpu:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        a @ b
    if on_gpu:
        # GPU calls return immediately and run in the background; wait for them to finish.
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1000


def check_torch() -> bool:
    print(f"  PyTorch {torch.__version__}, built for CUDA {torch.version.cuda}")
    print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
    gpu = resolve_device("auto")
    if gpu.type != "cuda":
        print("  No GPU visible: training would fall back to the CPU.")
        return False

    props = torch.cuda.get_device_properties(gpu)
    print(
        f"  GPU: {props.name}, {props.total_memory / 2**30:.1f} GiB, "
        f"compute capability {props.major}.{props.minor}"
    )

    a = torch.randn(4096, 4096)
    b = torch.randn(4096, 4096)
    cpu_ms = time_matmul(a, b)
    gpu_ms = time_matmul(a.to(gpu), b.to(gpu))
    print(f"  4096x4096 matmul: CPU {cpu_ms:.1f} ms, GPU {gpu_ms:.1f} ms")

    # Same input, same math: the results may differ only by float32 rounding.
    same = torch.allclose(a @ b, (a.to(gpu) @ b.to(gpu)).cpu(), rtol=1e-3, atol=1e-2)
    print(f"  GPU result matches CPU result: {same}")
    return same


# --- 3. mujoco ----------------------------------------------------------------------------


def demo_controls(model: mujoco.MjModel, t: float) -> np.ndarray:
    """Home pose, with joint 1 swinging slowly and the gripper toggling every 2 s."""
    ctrl = model.key("home").ctrl.copy()
    ctrl[0] += 0.8 * np.sin(2 * np.pi * t / 4.0)
    ctrl[7] = GRIPPER_OPEN if int(t / 2.0) % 2 == 0 else GRIPPER_CLOSED
    return ctrl


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    frame_dt = 1 / 60  # redraw the window at 60 FPS
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            frame_start = time.perf_counter()
            # One frame = several small physics steps (2 ms each).
            frame_end_time = data.time + frame_dt
            while data.time < frame_end_time:
                data.ctrl[:] = demo_controls(model, data.time)
                mujoco.mj_step(model, data)
            viewer.sync()
            # Sleep for the rest of the frame, so the simulation plays in real time.
            remaining = frame_dt - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)


def check_mujoco(show_viewer: bool) -> bool:
    scene = require_panda_scene()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    print(f"  MuJoCo {mujoco.__version__}: loaded {scene}")
    print(
        f"  nq = {model.nq} joint coordinates, nu = {model.nu} actuators, "
        f"timestep = {model.opt.timestep * 1000:.0f} ms"
    )

    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    sim_seconds = 2.0
    start = time.perf_counter()
    while data.time < sim_seconds:
        data.ctrl[:] = demo_controls(model, data.time)
        mujoco.mj_step(model, data)
    wall_seconds = time.perf_counter() - start
    print(
        f"  simulated {sim_seconds:.0f} s in {wall_seconds:.2f} s "
        f"({sim_seconds / wall_seconds:.0f}x faster than real time)"
    )

    if show_viewer:
        print("  Viewer: the arm swings, the gripper opens/closes. Close the window to go on.")
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        run_viewer(model, data)
    return True


# --- 4. offscreen rendering ---------------------------------------------------------------


def check_rendering() -> bool:
    """Render offscreen (as for videos) and report which GPU OpenGL runs on.

    Laptops with two GPUs often give Python the integrated one by default; there a frame
    takes ~60 ms instead of ~2 ms and interactive windows become unusable.
    """
    from OpenGL import GL  # PyOpenGL is installed together with mujoco

    model = mujoco.MjModel.from_xml_path(str(require_panda_scene()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        renderer.update_scene(data)
        renderer.render()  # the first frame also creates the OpenGL context
        gl_renderer = GL.glGetString(GL.GL_RENDERER).decode()
        start = time.perf_counter()
        for _ in range(20):
            renderer.update_scene(data)
            renderer.render()
        frame_ms = (time.perf_counter() - start) / 20 * 1000
    finally:
        renderer.close()

    print(f"  OpenGL runs on: {gl_renderer}")
    print(f"  one 640x480 frame: {frame_ms:.1f} ms")
    if torch.cuda.is_available() and "NVIDIA" not in gl_renderer:
        print("  An NVIDIA GPU is present, but OpenGL uses another (slow) GPU. Fix:")
        print("  Settings > System > Display > Graphics > Add desktop app >")
        print(f"  {sys._base_executable}  >  Options > High performance")
        return False
    return True


# --- 5. camera ----------------------------------------------------------------------------


def check_camera(index: int) -> bool:
    # On Windows the DirectShow backend opens most webcams much faster than the default one.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        print(f"  Could not open camera {index}.")
        print("  Windows: Settings > Privacy & security > Camera > allow desktop apps.")
        return False

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    window = "check_setup: camera (q / Esc = close)"
    print("  Camera preview open. Press q or Esc in the window to close it.")

    frames = 0
    fps = 0.0
    start = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("  The camera opened, but returned no image.")
                return False
            frames += 1
            fps = frames / (time.perf_counter() - start)

            frame = cv2.flip(frame, 1)  # mirror image, like a selfie camera
            height, width = frame.shape[:2]
            label = f"{width}x{height}  {fps:.1f} FPS"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
            if key in (ord("q"), 27) or window_closed:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"  {width}x{height} at {fps:.1f} FPS ({frames} frames)")
    return True


# --- main ---------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the gesture_bc development environment.")
    parser.add_argument("--no-gui", action="store_true", help="skip the viewer and camera windows")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    args = parser.parse_args()

    checks = {
        "versions": check_versions,
        "pytorch + cuda": check_torch,
        "mujoco": lambda: check_mujoco(show_viewer=not args.no_gui),
        "rendering": check_rendering,
    }
    if not args.no_gui:
        checks["camera"] = lambda: check_camera(args.camera)

    results = {}
    for name, check in checks.items():
        print(f"\n=== {name} ===")
        try:
            results[name] = check()
        except Exception as error:  # report every failing check, not only the first one
            print(f"  ERROR: {type(error).__name__}: {error}")
            results[name] = False

    print("\n=== summary ===")
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
