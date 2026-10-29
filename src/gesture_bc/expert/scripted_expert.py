"""Scripted expert for pick-and-place: a finite-state machine on observations.

The expert sees only the observation, exactly like a learned policy. Its demonstrations are
therefore a function a network can learn (no hidden information about the simulation).

Phases (in order):
    REACH     move above the cube, gripper open           -> DESCEND when above the cube
    DESCEND   lower the open gripper around the cube      -> GRASP at grasp height
                                                             (REACH again if the cube moved away)
    GRASP     close the gripper and wait                  -> LIFT (REACH if it closed on nothing)
    LIFT      raise the cube to carry height              -> CARRY when high enough
    CARRY     move the cube above the target              -> LOWER when above the target
    LOWER     put the cube down on the target             -> RELEASE just above the table
    RELEASE   open the gripper and wait                   -> RETREAT
    RETREAT   move up                                     -> REACH (try again if not successful)

During LIFT, CARRY and LOWER the expert checks that the cube is still between the fingers;
if it was dropped, the expert starts again from REACH.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

OPEN, CLOSE = -1.0, 1.0


class Phase(Enum):
    REACH = "reach"
    DESCEND = "descend"
    GRASP = "grasp"
    LIFT = "lift"
    CARRY = "carry"
    LOWER = "lower"
    RELEASE = "release"
    RETREAT = "retreat"


CARRYING_PHASES = (Phase.LIFT, Phase.CARRY, Phase.LOWER)


@dataclass
class ExpertConfig:
    """Parameters of the scripted expert; configs/expert/scripted.yaml can override them."""

    # PD controller in TCP space: kp = fraction of the remaining distance commanded per
    # step, kd [s] = damping by the TCP velocity (removes overshoot, see notes).
    kp: float = 0.8
    kd: float = 0.06
    # Heights [m]: hover above the cube before descending (relative to the cube center),
    # TCP height while carrying (above the table), clearance under the cube when placing.
    hover_height: float = 0.08
    carry_height: float = 0.15
    place_clearance: float = 0.005
    # When is a position "reached" [m].
    xy_tolerance: float = 0.008
    z_tolerance: float = 0.005
    # Waiting times [control steps] for the gripper to close / open.
    grasp_steps: int = 6
    release_steps: int = 4
    # Gripper narrower than this after closing means it closed on nothing [m].
    empty_grasp_opening: float = 0.01
    # While carrying, cube farther than this from the TCP means it was dropped [m].
    lost_cube_distance: float = 0.03


class ScriptedExpert:
    """Policy interface: ``reset()`` at the start of an episode, then ``act(obs)`` per step."""

    def __init__(self, config: ExpertConfig, max_step_m: float):
        self.config = config
        # The environment's action scale: action 1.0 moves the TCP target by max_step_m.
        self.max_step_m = max_step_m
        self.reset()

    def reset(self) -> None:
        self.phase = Phase.REACH
        self._steps_in_phase = 0

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Action [dx, dy, dz, gripper] in [-1, 1] for the current observation."""
        self._update_phase(obs)
        self._steps_in_phase += 1
        goal, gripper = self._goal_and_gripper(obs)
        return np.append(self._move_towards(goal, obs), gripper)

    # --- phase logic --------------------------------------------------------------------------

    def _update_phase(self, obs: dict[str, np.ndarray]) -> None:
        cfg = self.config
        tcp, cube = obs["tcp_pos"], obs["cube_pos"]
        cube_distance = np.linalg.norm(cube - tcp)

        if self.phase in CARRYING_PHASES and cube_distance > cfg.lost_cube_distance:
            self._enter(Phase.REACH)  # the cube was dropped: start again
            return

        goal, _ = self._goal_and_gripper(obs)
        xy_error = np.linalg.norm(goal[:2] - tcp[:2])
        z_error = abs(goal[2] - tcp[2])

        if self.phase == Phase.REACH:
            if xy_error < cfg.xy_tolerance and z_error < 2 * cfg.xy_tolerance:
                self._enter(Phase.DESCEND)
        elif self.phase == Phase.DESCEND:
            if xy_error > 2 * cfg.xy_tolerance:
                self._enter(Phase.REACH)  # the cube moved (e.g. it was pushed): realign
            elif z_error < cfg.z_tolerance:
                self._enter(Phase.GRASP)
        elif self.phase == Phase.GRASP:
            if self._steps_in_phase >= cfg.grasp_steps:
                closed_on_nothing = obs["gripper_opening"][0] < cfg.empty_grasp_opening
                self._enter(Phase.REACH if closed_on_nothing else Phase.LIFT)
        elif self.phase == Phase.LIFT:
            if tcp[2] > cfg.carry_height - 2 * cfg.z_tolerance:
                self._enter(Phase.CARRY)
        elif self.phase == Phase.CARRY:
            if xy_error < cfg.xy_tolerance:
                self._enter(Phase.LOWER)
        elif self.phase == Phase.LOWER:
            if z_error < cfg.z_tolerance:
                self._enter(Phase.RELEASE)
        elif self.phase == Phase.RELEASE:
            if self._steps_in_phase >= cfg.release_steps:
                self._enter(Phase.RETREAT)
        elif self.phase == Phase.RETREAT and z_error < 2 * cfg.z_tolerance:
            self._enter(Phase.REACH)

    def _enter(self, phase: Phase) -> None:
        self.phase = phase
        self._steps_in_phase = 0

    def _goal_and_gripper(self, obs: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        """Where the TCP should go in the current phase, and the gripper command."""
        cfg = self.config
        tcp, cube, target = obs["tcp_pos"], obs["cube_pos"], obs["target_pos"]
        above_cube = cube + [0.0, 0.0, cfg.hover_height]
        # Where the TCP must be so that the *cube* (not the TCP) ends up at the target:
        # the cube may sit a few millimeters off-center between the fingers.
        tcp_for_cube_on_target = target - obs["tcp_to_cube"]

        if self.phase == Phase.REACH:
            return above_cube, OPEN
        if self.phase == Phase.DESCEND:
            return cube.copy(), OPEN
        if self.phase == Phase.GRASP:
            return tcp.copy(), CLOSE  # hold still while the fingers close
        if self.phase == Phase.LIFT:
            return np.array([tcp[0], tcp[1], cfg.carry_height]), CLOSE
        if self.phase == Phase.CARRY:
            return np.append(tcp_for_cube_on_target[:2], cfg.carry_height), CLOSE
        if self.phase == Phase.LOWER:
            return tcp_for_cube_on_target + [0.0, 0.0, cfg.place_clearance], CLOSE
        if self.phase == Phase.RELEASE:
            return tcp.copy(), OPEN
        return above_cube, OPEN  # RETREAT

    def _move_towards(self, goal: np.ndarray, obs: dict[str, np.ndarray]) -> np.ndarray:
        """PD controller: towards the goal, damped by the TCP velocity, scaled to [-1, 1]."""
        command = self.config.kp * (goal - obs["tcp_pos"]) - self.config.kd * obs["tcp_vel"]
        return np.clip(command / self.max_step_m, -1.0, 1.0)
