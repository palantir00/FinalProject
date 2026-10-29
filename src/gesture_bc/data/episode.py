"""One recorded episode (demonstration or policy rollout)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    """Everything recorded at control steps t = 0 .. T-1.

    Row t of every array belongs to the same moment: the observation the demonstrator saw,
    the action it chose, and the simulation state at that moment (before the action).
    """

    obs: dict[str, np.ndarray]  # key -> (T, size): observation before action t
    actions: np.ndarray  # (T, 4): action chosen by the demonstrator (what a policy learns)
    executed_actions: np.ndarray  # (T, 4): action actually sent (differs only with noise)
    tcp_targets: np.ndarray  # (T, 3): absolute TCP target after action t
    qpos: np.ndarray  # (T, nq): simulation state before action t (for replay / re-render)
    qvel: np.ndarray  # (T, nv)
    seed: int  # environment seed: reset(seed) recreates the starting scene
    success: bool  # did the episode end with the cube placed on the target?

    @property
    def num_steps(self) -> int:
        return len(self.actions)
