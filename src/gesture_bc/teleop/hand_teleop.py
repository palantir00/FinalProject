"""Hand teleoperation: hand features -> TCP goal -> action (the same format as the expert's).

Mapping (relative to a calibration pose, like a mirror of the "front" camera view):
    palm moves right / left in the mirrored image  -> TCP moves right / left  (world y)
    palm moves up / down                           -> TCP moves up / down     (world z)
    palm gets larger / smaller (hand comes closer) -> TCP moves towards you / away (world x)
    pinch (thumb tip touches index tip)            -> gripper closes; open the fingers to release

Calibration (key C): "the hand as it is now = the TCP where it is now". Pressing C again
works like lifting a mouse off the desk: the robot stays and the hand can move to a more
comfortable place.
"""

from dataclasses import dataclass

import numpy as np

from gesture_bc.control import CLOSE, OPEN, pd_move_action
from gesture_bc.teleop.hand_features import HandFeatures


@dataclass
class HandTeleopConfig:
    """Parameters of hand teleoperation; configs/teleop/hand.yaml can override them."""

    camera_index: int = 0
    # Mapping gains [m of TCP motion per unit of hand motion]. Palm position is measured in
    # image widths, so lateral_gain 1.0 means: moving the palm across the whole image moves
    # the TCP by 1 m (the workspace is 0.6 m wide). depth_gain multiplies the relative change
    # of the palm size (1.2 = palm 20 % larger than at calibration). The vertical gain is
    # larger because the 4:3 image is only 0.75 widths high: with 1.5, the hand moves from
    # the middle of the image ~0.25 of its height down to bring the gripper to the table.
    lateral_gain: float = 1.0
    vertical_gain: float = 1.5
    depth_gain: float = 0.5
    # Gripper with hysteresis: closes below close_pinch, opens above open_pinch, keeps its
    # state in between, so noise around one threshold cannot make it flicker.
    close_pinch: float = 0.3
    open_pinch: float = 0.45
    # One-Euro filter for palm position and size (see one_euro.py).
    filter_min_cutoff_hz: float = 1.0
    filter_beta: float = 5.0
    filter_d_cutoff_hz: float = 1.0
    # No hand for this long [s] counts as "hand lost": the robot holds its last goal.
    lost_timeout_s: float = 0.25
    # PD controller: the same gains as the scripted expert (configs/expert/scripted.yaml).
    kp: float = 0.8
    kd: float = 0.06


class HandTeleop:
    def __init__(
        self,
        config: HandTeleopConfig,
        workspace_low: np.ndarray,
        workspace_high: np.ndarray,
        max_step_m: float,
    ):
        self.config = config
        self.workspace_low = np.asarray(workspace_low, dtype=float)
        self.workspace_high = np.asarray(workspace_high, dtype=float)
        # The environment's action scale: action 1.0 moves the TCP target by max_step_m.
        self.max_step_m = max_step_m
        self._reference: tuple[HandFeatures, np.ndarray] | None = None
        self.reset()

    def reset(self) -> None:
        """New episode: forget the goal and open the gripper. The calibration stays."""
        self.goal: np.ndarray | None = None
        self.gripper_closed = False

    @property
    def calibrated(self) -> bool:
        return self._reference is not None

    @property
    def reference(self) -> tuple[HandFeatures, np.ndarray] | None:
        """(hand features, TCP position) recorded at calibration."""
        return self._reference

    def calibrate(self, features: HandFeatures, tcp_pos: np.ndarray) -> None:
        self._reference = (features, np.array(tcp_pos, dtype=float))
        self.goal = None

    def goal_for(self, features: HandFeatures) -> np.ndarray:
        """TCP goal [m] for the current hand, kept inside the workspace."""
        if self._reference is None:
            raise RuntimeError("calibrate() first")
        ref_features, ref_tcp = self._reference
        cfg = self.config
        right, down = features.position - ref_features.position  # image y points down
        size_change = features.size / ref_features.size - 1.0
        offset = np.array(
            [cfg.depth_gain * size_change, cfg.lateral_gain * right, -cfg.vertical_gain * down]
        )
        return np.clip(ref_tcp + offset, self.workspace_low, self.workspace_high)

    def update_gripper(self, pinch: float) -> None:
        if pinch < self.config.close_pinch:
            self.gripper_closed = True
        elif pinch > self.config.open_pinch:
            self.gripper_closed = False

    def act(self, obs: dict[str, np.ndarray], features: HandFeatures | None) -> np.ndarray:
        """Action [dx, dy, dz, gripper]; features=None means the hand is not visible."""
        if features is not None and self.calibrated:
            self.goal = self.goal_for(features)
            self.update_gripper(features.pinch)
        if self.goal is None:
            self.goal = obs["tcp_pos"].copy()  # not calibrated yet: stay where you are
        cfg = self.config
        move = pd_move_action(
            self.goal, obs["tcp_pos"], obs["tcp_vel"], cfg.kp, cfg.kd, self.max_step_m
        )
        return np.append(move, CLOSE if self.gripper_closed else OPEN)
