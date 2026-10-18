"""Build the MuJoCo model of the pick-and-place scene with the Franka Panda attached."""

import mujoco
import numpy as np

from gesture_bc.sim.paths import PICK_PLACE_XML, require_panda_xml

# Names used by the environment and the controller.
TCP_SITE = "tcp"
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]
GRIPPER_ACTUATOR = "actuator8"
CUBE_BODY = "cube"
CUBE_GEOM = "cube"
CUBE_JOINT = "cube_joint"
TARGET_BODY = "target"

# Tool center point: the middle of the fingertip pads, 10.34 cm from the hand flange along
# its z axis (Franka's documented TCP offset; it matches the pad geoms in panda.xml).
TCP_OFFSET_M = 0.1034

# Arm configuration of the Menagerie "home" keyframe: gripper above the table, pointing down.
HOME_ARM_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# Gripper actuator command (Menagerie maps 0..0.04 m finger travel to 0..255).
GRIPPER_OPEN_CTRL = 255.0
GRIPPER_CLOSED_CTRL = 0.0
# Each finger slides 0..0.04 m, so the gripper opening (sum of both) is 0..0.08 m.
FINGER_OPEN_QPOS = 0.04
LEFT_FINGER_BODY = "left_finger"
RIGHT_FINGER_BODY = "right_finger"


def build_model() -> mujoco.MjModel:
    """Load our scene XML, attach the Panda at the origin, and compile."""
    scene = mujoco.MjSpec.from_file(str(PICK_PLACE_XML))
    panda = mujoco.MjSpec.from_file(str(require_panda_xml()))

    # A site is a named reference frame on a body. The controller moves this point.
    panda.body("hand").add_site(
        name=TCP_SITE,
        pos=[0.0, 0.0, TCP_OFFSET_M],
        size=[0.005, 0.005, 0.005],
        group=4,  # hidden by default, so it does not show up in videos
    )

    # Gravity compensation: cancels the weight of each robot link, like the real Panda's
    # internal controller does. Without it the position servos sag a little under gravity.
    for body in panda.bodies:
        if body.name != "world":
            body.gravcomp = 1.0

    # Attaching (instead of an XML <include>) lets MuJoCo resolve the Panda's mesh files
    # relative to panda.xml, wherever our scene file lives.
    scene.worldbody.add_frame().attach_body(panda.body("link0"), "", "")
    return scene.compile()
