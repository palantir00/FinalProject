import mujoco
import numpy as np
import pytest

from gesture_bc.sim import scene
from gesture_bc.sim.ik_controller import DiffIKController, orientation_error


@pytest.fixture(scope="module")
def ik() -> DiffIKController:
    model = scene.build_model()
    return DiffIKController(model, scene.TCP_SITE, scene.ARM_JOINTS, scene.HOME_ARM_QPOS)


@pytest.fixture(scope="module")
def pointing_down(ik) -> np.ndarray:
    """Quaternion of the TCP in the home pose (gripper pointing straight down)."""
    _, rotation = ik.forward_kinematics(scene.HOME_ARM_QPOS)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.flatten())
    return quat


def test_home_pose_points_gripper_down(ik):
    _, rotation = ik.forward_kinematics(scene.HOME_ARM_QPOS)
    tcp_z_axis = rotation[:, 2]  # the columns of a rotation matrix are the TCP's axes
    np.testing.assert_allclose(tcp_z_axis, [0.0, 0.0, -1.0], atol=1e-3)


def test_orientation_error_is_zero_for_the_same_orientation(ik, pointing_down):
    _, rotation = ik.forward_kinematics(scene.HOME_ARM_QPOS)
    np.testing.assert_allclose(orientation_error(pointing_down, rotation.flatten()), 0, atol=1e-6)


@pytest.mark.parametrize("target", [[0.5, 0.0, 0.1], [0.4, 0.2, 0.03], [0.6, -0.2, 0.3]])
def test_solve_reaches_the_target_pointing_down(ik, pointing_down, target):
    target = np.array(target)
    q = ik.solve(scene.HOME_ARM_QPOS, target, pointing_down, max_iterations=300)
    reached, rotation = ik.forward_kinematics(q)
    assert np.linalg.norm(reached - target) < 1e-3  # within 1 mm
    assert np.linalg.norm(orientation_error(pointing_down, rotation.flatten())) < 0.01  # < 0.6 deg


def test_solution_respects_joint_limits(ik, pointing_down):
    far_away = np.array([2.0, 0.0, 0.3])  # unreachable: IK must still return legal angles
    q = ik.solve(scene.HOME_ARM_QPOS, far_away, pointing_down, max_iterations=100)
    assert np.all(q >= ik.joint_low) and np.all(q <= ik.joint_high)
