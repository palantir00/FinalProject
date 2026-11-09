"""TCP controller shared by both demonstrators: the scripted expert and hand teleoperation.

Both decide WHERE the gripper should go (the expert from its state machine, the human with
the hand); this controller turns that goal into an action in the same way for both. The two
kinds of demonstrations then differ only in their goals, which keeps the comparison fair.
"""

import numpy as np

# Gripper command, the last element of an action (the environment closes for values > 0).
OPEN, CLOSE = -1.0, 1.0


def pd_move_action(
    goal: np.ndarray,
    tcp_pos: np.ndarray,
    tcp_vel: np.ndarray,
    kp: float,
    kd: float,
    max_step_m: float,
) -> np.ndarray:
    """PD controller: towards the goal, damped by the TCP velocity, scaled to [-1, 1].

    kp = fraction of the remaining distance commanded per step, kd [s] = damping by the TCP
    velocity (removes overshoot). Action 1.0 moves the TCP target by max_step_m.
    """
    command = kp * (goal - tcp_pos) - kd * tcp_vel
    return np.clip(command / max_step_m, -1.0, 1.0)
