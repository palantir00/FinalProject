"""Differential inverse kinematics (IK) with damped least squares.

Given a desired position and orientation of the tool center point (TCP), find the 7 arm
joint angles that put the TCP there. The solver repeats small linear steps:

    error   = desired TCP pose - current TCP pose          (6 numbers: position + rotation)
    J       = d(TCP pose) / d(joint angles)                (6x7 Jacobian, from MuJoCo)
    dq      = J^T (J J^T + damping * I)^-1 error           (damped least squares)
    q       = q + dq

The damping keeps dq small near singular poses, where a plain inverse would explode.
The Panda has 7 joints but the task has only 6 dimensions, so one direction of joint motion
(the "null space") does not move the TCP at all. We use it to pull the arm towards its
home posture, which keeps the elbow in a natural position.
"""

import mujoco
import numpy as np


def orientation_error(target_quat: np.ndarray, current_xmat: np.ndarray) -> np.ndarray:
    """Rotation (axis * angle, world frame) that turns the current orientation into the target."""
    current_quat = np.empty(4)
    mujoco.mju_mat2Quat(current_quat, current_xmat)
    current_inverse = np.empty(4)
    mujoco.mju_negQuat(current_inverse, current_quat)
    error_quat = np.empty(4)
    mujoco.mju_mulQuat(error_quat, target_quat, current_inverse)
    error = np.empty(3)
    mujoco.mju_quat2Vel(error, error_quat, 1.0)
    return error


class DiffIKController:
    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str,
        joint_names: list[str],
        rest_qpos: np.ndarray,
        damping: float = 1e-4,
        nullspace_gain: float = 0.5,
        max_iterations: int = 20,
        position_tolerance: float = 1e-4,
        rotation_tolerance: float = 1e-3,
    ):
        self.model = model
        # Private copy of the simulation state: IK only "imagines" joint angles here and
        # never disturbs the real simulation.
        self.data = mujoco.MjData(model)
        self.site_id = model.site(site_name).id
        joint_ids = [model.joint(name).id for name in joint_names]
        self.qpos_ids = model.jnt_qposadr[joint_ids]
        self.dof_ids = model.jnt_dofadr[joint_ids]
        self.joint_low, self.joint_high = model.jnt_range[joint_ids].T
        self.rest_qpos = np.asarray(rest_qpos, dtype=float)
        self.damping = damping
        self.nullspace_gain = nullspace_gain
        self.max_iterations = max_iterations
        self.position_tolerance = position_tolerance
        self.rotation_tolerance = rotation_tolerance

    def forward_kinematics(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """TCP position (3,) and rotation matrix (3, 3) for arm joint angles q."""
        self._set_arm(q)
        position = self.data.site_xpos[self.site_id].copy()
        rotation = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return position, rotation

    def solve(
        self,
        q_start: np.ndarray,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        """Return arm joint angles that put the TCP at target_pos with orientation target_quat.

        q_start is the initial guess; during control it is the previous solution, so the
        joint targets change smoothly from one control step to the next.
        """
        q = np.array(q_start, dtype=float)
        n_joints = len(q)
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))

        for _ in range(max_iterations or self.max_iterations):
            self._set_arm(q)
            pos_error = target_pos - self.data.site_xpos[self.site_id]
            rot_error = orientation_error(target_quat, self.data.site_xmat[self.site_id])
            if (
                np.linalg.norm(pos_error) < self.position_tolerance
                and np.linalg.norm(rot_error) < self.rotation_tolerance
            ):
                break

            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.site_id)
            jac = np.vstack([jac_pos[:, self.dof_ids], jac_rot[:, self.dof_ids]])  # (6, 7)
            error = np.concatenate([pos_error, rot_error])

            # Damped least squares step towards the target pose.
            jjt_damped_inv = np.linalg.inv(jac @ jac.T + self.damping * np.eye(6))
            dq = jac.T @ jjt_damped_inv @ error

            # Null-space step: move towards the rest posture without moving the TCP.
            jac_pinv = jac.T @ jjt_damped_inv
            nullspace = np.eye(n_joints) - jac_pinv @ jac
            dq += nullspace @ (self.nullspace_gain * (self.rest_qpos - q))

            q = np.clip(q + dq, self.joint_low, self.joint_high)
        return q

    def _set_arm(self, q: np.ndarray) -> None:
        self.data.qpos[self.qpos_ids] = q
        # Position-dependent computations only: link poses and what the Jacobian needs.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
