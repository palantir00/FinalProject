"""Pick-and-place environment: Franka Panda, one cube, one target zone (Gymnasium API).

Action (4 numbers in [-1, 1]):
    [dx, dy, dz, gripper]  dx..dz move the TCP target by up to max_step_m per step,
                           gripper > 0 closes the gripper, gripper <= 0 opens it.
Observation (dict of float32 arrays, see OBS_KEYS):
    TCP position, gripper opening, cube position, target position and two relative vectors.
Episode end:
    terminated when the cube is placed in the target zone (success),
    truncated after max_episode_steps.
"""

from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np

from gesture_bc.sim import scene
from gesture_bc.sim.ik_controller import DiffIKController

# Observation entries in the order used when they are concatenated for a neural network.
OBS_KEYS = ["tcp_pos", "gripper_opening", "cube_pos", "target_pos", "tcp_to_cube", "cube_to_target"]
OBS_SIZES = {
    "tcp_pos": 3,
    "gripper_opening": 1,
    "cube_pos": 3,
    "target_pos": 3,
    "tcp_to_cube": 3,
    "cube_to_target": 3,
    "tcp_vel": 3,
}


@dataclass
class PickPlaceConfig:
    """All parameters of the environment; configs/env/pick_place.yaml can override them."""

    # Control: the policy acts 20 times per second, physics runs at 500 Hz in between.
    control_hz: int = 20
    # 30 s. The same limit is used to evaluate every policy, so it must leave room for slow
    # (human) demonstrations; it will be checked against the recorded demo lengths.
    max_episode_steps: int = 600
    # Action 1.0 on an axis moves the TCP target by this distance per step [m]
    # (0.015 m * 20 Hz = 0.3 m/s maximum speed).
    max_step_m: float = 0.015
    # The TCP target is kept inside this box (x, y, z) [m]. z_min keeps the fingertips just
    # above the table; at z_max the far corners are still reachable (arm not fully stretched).
    workspace_low: tuple[float, float, float] = (0.30, -0.30, 0.012)
    workspace_high: tuple[float, float, float] = (0.70, 0.30, 0.35)
    # The TCP target may lead the real TCP by at most this distance [m], so it cannot run
    # away when the gripper is blocked (e.g. pressing on the cube).
    max_target_lead_m: float = 0.04
    # TCP position at the start of every episode [m].
    start_tcp_pos: tuple[float, float, float] = (0.45, 0.0, 0.30)

    # Random (x, y) positions of the cube and of the target zone [m].
    cube_spawn_low: tuple[float, float] = (0.40, -0.20)
    cube_spawn_high: tuple[float, float] = (0.65, 0.20)
    target_spawn_low: tuple[float, float] = (0.40, -0.20)
    target_spawn_high: tuple[float, float] = (0.65, 0.20)
    min_cube_target_distance: float = 0.12

    # Success: cube center within this (x, y) distance of the target center [m],
    # cube resting on the table and released by the gripper.
    success_radius: float = 0.03

    # Robustness tests (all off by default).
    obs_noise_std: float = 0.0  # Gaussian noise on the observed TCP and cube positions [m]
    include_velocity: bool = False  # add the TCP velocity to the observation
    perturb_cube: bool = False  # push the cube aside once per episode, before the grasp
    perturb_step_range: tuple[int, int] = (10, 40)  # earliest step of the push
    perturb_distance_range: tuple[float, float] = (0.03, 0.06)  # [m]

    # Offscreen rendering (videos).
    render_camera: str = "angled"
    render_width: int = 640
    render_height: int = 480


def flatten_observation(obs: dict[str, np.ndarray], keys: list[str]) -> np.ndarray:
    """Concatenate observation entries into one vector, in a fixed order."""
    return np.concatenate([obs[key] for key in keys]).astype(np.float32)


class PickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: PickPlaceConfig | None = None):
        self.config = config or PickPlaceConfig()
        cfg = self.config
        self.metadata = {**self.metadata, "render_fps": cfg.control_hz}

        self.model = scene.build_model()
        self.data = mujoco.MjData(self.model)
        physics_hz = round(1.0 / self.model.opt.timestep)
        if physics_hz % cfg.control_hz != 0:
            raise ValueError(f"control_hz must divide the physics rate ({physics_hz} Hz)")
        self.n_substeps = physics_hz // cfg.control_hz

        # Indices into MuJoCo arrays, looked up by name (never hard-coded numbers).
        m = self.model
        self._arm_qpos_ids = m.jnt_qposadr[[m.joint(name).id for name in scene.ARM_JOINTS]]
        self._finger_qpos_ids = m.jnt_qposadr[[m.joint(name).id for name in scene.FINGER_JOINTS]]
        self._arm_actuator_ids = [m.actuator(f"actuator{i}").id for i in range(1, 8)]
        self._gripper_actuator_id = m.actuator(scene.GRIPPER_ACTUATOR).id
        self._tcp_site_id = m.site(scene.TCP_SITE).id
        self._cube_body_id = m.body(scene.CUBE_BODY).id
        self._cube_qpos_adr = m.jnt_qposadr[m.joint(scene.CUBE_JOINT).id]
        self._cube_qvel_adr = m.jnt_dofadr[m.joint(scene.CUBE_JOINT).id]
        self._target_mocap_id = m.body_mocapid[m.body(scene.TARGET_BODY).id]
        self._finger_body_ids = (
            m.body(scene.LEFT_FINGER_BODY).id,
            m.body(scene.RIGHT_FINGER_BODY).id,
        )
        self.cube_half_size = float(m.geom_size[m.geom(scene.CUBE_GEOM).id][0])

        self.ik = DiffIKController(m, scene.TCP_SITE, scene.ARM_JOINTS, scene.HOME_ARM_QPOS)
        # The gripper keeps the orientation of the home pose (pointing straight down).
        _, home_rotation = self.ik.forward_kinematics(scene.HOME_ARM_QPOS)
        self._tcp_quat = np.empty(4)
        mujoco.mju_mat2Quat(self._tcp_quat, home_rotation.flatten())
        # Joint angles of the start pose, solved once and reused at every reset.
        start_tcp = np.array(cfg.start_tcp_pos)
        self._start_arm_qpos = self.ik.solve(
            scene.HOME_ARM_QPOS, start_tcp, self._tcp_quat, max_iterations=200
        )
        start_reached, _ = self.ik.forward_kinematics(self._start_arm_qpos)
        if np.linalg.norm(start_reached - start_tcp) > 1e-3:
            raise ValueError(f"start_tcp_pos {cfg.start_tcp_pos} is not reachable")

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.obs_keys = OBS_KEYS + (["tcp_vel"] if cfg.include_velocity else [])
        self.observation_space = gym.spaces.Dict(
            {
                key: gym.spaces.Box(-np.inf, np.inf, shape=(OBS_SIZES[key],), dtype=np.float32)
                for key in self.obs_keys
            }
        )
        self._renderer: mujoco.Renderer | None = None

    # --- Gymnasium API ------------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Start a new episode. ``options`` may fix "cube_xy" and/or "target_xy"."""
        super().reset(seed=seed)  # seeds self.np_random, the only random generator we use
        options = options or {}
        m, d = self.model, self.data

        mujoco.mj_resetData(m, d)
        d.qpos[self._arm_qpos_ids] = self._start_arm_qpos
        d.qpos[self._finger_qpos_ids] = scene.FINGER_OPEN_QPOS
        d.ctrl[self._arm_actuator_ids] = self._start_arm_qpos
        d.ctrl[self._gripper_actuator_id] = scene.GRIPPER_OPEN_CTRL

        if "cube_xy" in options:
            cube_xy = np.asarray(options["cube_xy"], dtype=float)
        else:
            cube_xy = self.np_random.uniform(
                self.config.cube_spawn_low, self.config.cube_spawn_high
            )
        if "target_xy" in options:
            target_xy = np.asarray(options["target_xy"], dtype=float)
        else:
            target_xy = self._sample_target_xy(cube_xy)
        self._place_cube(cube_xy)
        d.mocap_pos[self._target_mocap_id] = [*target_xy, 0.0005]
        # The cube's center should end up here: on the table, above the target center.
        self._target_pos = np.array([*target_xy, self.cube_half_size])
        mujoco.mj_forward(m, d)

        self._arm_target = self._start_arm_qpos.copy()
        self._tcp_target = self._true_tcp_pos()
        self._tcp_vel = np.zeros(3)
        self._steps = 0
        self._cube_was_lifted = False
        self._perturbed = False
        self._perturb_from_step = (
            int(self.np_random.integers(*self.config.perturb_step_range, endpoint=True))
            if self.config.perturb_cube
            else None
        )
        return self._get_obs(), self._get_info()

    def step(self, action):
        cfg = self.config
        m, d = self.model, self.data
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)

        # 1) Move the TCP target; keep it close to the real TCP and inside the workspace.
        tcp_before = self._true_tcp_pos()
        target = self._tcp_target + action[:3] * cfg.max_step_m
        lead = target - tcp_before
        lead_norm = np.linalg.norm(lead)
        if lead_norm > cfg.max_target_lead_m:
            target = tcp_before + lead * (cfg.max_target_lead_m / lead_norm)
        self._tcp_target = np.clip(target, cfg.workspace_low, cfg.workspace_high)

        # 2) IK turns the TCP target into joint targets; the position servos track them.
        self._arm_target = self.ik.solve(self._arm_target, self._tcp_target, self._tcp_quat)
        d.ctrl[self._arm_actuator_ids] = self._arm_target
        close = action[3] > 0
        d.ctrl[self._gripper_actuator_id] = (
            scene.GRIPPER_CLOSED_CTRL if close else scene.GRIPPER_OPEN_CTRL
        )

        # 3) Simulate one control period (25 physics steps = 50 ms).
        for _ in range(self.n_substeps):
            mujoco.mj_step(m, d)
        self._steps += 1
        tcp_after = self._true_tcp_pos()
        self._tcp_vel = (tcp_after - tcp_before) * cfg.control_hz

        # 4) Optional disturbance for robustness tests (once, at the first safe moment).
        perturb_due = self._perturb_from_step is not None and self._steps >= self._perturb_from_step
        if perturb_due and not self._perturbed and self._can_perturb():
            self._perturb_cube()

        if self._true_cube_pos()[2] > self.cube_half_size + 0.02:
            self._cube_was_lifted = True

        obs = self._get_obs()
        info = self._get_info()
        terminated = info["is_success"]
        truncated = self._steps >= cfg.max_episode_steps
        reward = 1.0 if terminated else 0.0  # sparse; behavior cloning does not use it
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return self.render_camera(self.config.render_camera)

    def render_camera(self, camera: str) -> np.ndarray:
        """RGB image (H, W, 3, uint8) from one of the scene cameras."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.config.render_height, width=self.config.render_width
            )
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # --- State queries ------------------------------------------------------------------------

    def gripper_opening(self) -> float:
        """Distance between the fingers [m], 0 (closed) .. 0.08 (open)."""
        return float(self.data.qpos[self._finger_qpos_ids].sum())

    def is_grasping(self) -> bool:
        """True when both fingers touch the cube."""
        contact = self.data.contact
        body1 = self.model.geom_bodyid[contact.geom1]
        body2 = self.model.geom_bodyid[contact.geom2]

        def touches_cube(finger_body: int) -> bool:
            pairs = ((body1 == self._cube_body_id) & (body2 == finger_body)) | (
                (body2 == self._cube_body_id) & (body1 == finger_body)
            )
            return bool(np.any(pairs))

        left, right = self._finger_body_ids
        return touches_cube(left) and touches_cube(right)

    def is_success(self) -> bool:
        cube = self._true_cube_pos()
        in_zone = np.linalg.norm(cube[:2] - self._target_pos[:2]) < self.config.success_radius
        on_table = cube[2] < self.cube_half_size + 0.005
        released = self.gripper_opening() > 2 * self.cube_half_size + 0.005
        return bool(in_zone and on_table and released)

    # --- Internals ----------------------------------------------------------------------------

    def _true_tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._tcp_site_id].copy()

    def _true_cube_pos(self) -> np.ndarray:
        return self.data.xpos[self._cube_body_id].copy()

    def _get_obs(self) -> dict[str, np.ndarray]:
        tcp_pos = self._true_tcp_pos()
        cube_pos = self._true_cube_pos()
        if self.config.obs_noise_std > 0:
            tcp_pos += self.np_random.normal(0.0, self.config.obs_noise_std, size=3)
            cube_pos += self.np_random.normal(0.0, self.config.obs_noise_std, size=3)
        obs = {
            "tcp_pos": tcp_pos,
            "gripper_opening": np.array([self.gripper_opening()]),
            "cube_pos": cube_pos,
            "target_pos": self._target_pos.copy(),
            # Relative vectors are redundant, but save the network from learning subtraction.
            "tcp_to_cube": cube_pos - tcp_pos,
            "cube_to_target": self._target_pos - cube_pos,
            "tcp_vel": self._tcp_vel.copy(),
        }
        return {key: obs[key].astype(np.float32) for key in self.obs_keys}

    def _get_info(self) -> dict:
        return {
            "is_success": self.is_success(),
            "is_grasping": self.is_grasping(),
            "cube_was_lifted": self._cube_was_lifted,
            "tcp_target": self._tcp_target.copy(),  # absolute target position, for datasets
            "perturbed": self._perturbed,
            "step": self._steps,
        }

    def _sample_target_xy(self, cube_xy: np.ndarray) -> np.ndarray:
        """Uniform target position at least min_cube_target_distance away from the cube."""
        cfg = self.config
        for _ in range(1000):
            target_xy = self.np_random.uniform(cfg.target_spawn_low, cfg.target_spawn_high)
            if np.linalg.norm(target_xy - cube_xy) >= cfg.min_cube_target_distance:
                return target_xy
        raise RuntimeError("Could not place the target; min_cube_target_distance is too large")

    def _place_cube(self, xy: np.ndarray) -> None:
        """Put the cube flat on the table at (x, y), not moving."""
        adr = self._cube_qpos_adr
        self.data.qpos[adr : adr + 3] = [xy[0], xy[1], self.cube_half_size]
        self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]  # no rotation (quaternion)
        self.data.qvel[self._cube_qvel_adr : self._cube_qvel_adr + 6] = 0.0

    def _can_perturb(self) -> bool:
        """Only push a cube that rests on the table, with the gripper well above it."""
        cube_top = self._true_cube_pos()[2] + self.cube_half_size
        gripper_clear = self._true_tcp_pos()[2] > cube_top + 0.05
        return gripper_clear and not self.is_grasping() and not self._cube_was_lifted

    def _perturb_cube(self) -> None:
        cfg = self.config
        angle = self.np_random.uniform(0.0, 2 * np.pi)
        distance = self.np_random.uniform(*cfg.perturb_distance_range)
        new_xy = self._true_cube_pos()[:2] + distance * np.array([np.cos(angle), np.sin(angle)])
        self._place_cube(np.clip(new_xy, cfg.cube_spawn_low, cfg.cube_spawn_high))
        mujoco.mj_forward(self.model, self.data)
        self._perturbed = True
