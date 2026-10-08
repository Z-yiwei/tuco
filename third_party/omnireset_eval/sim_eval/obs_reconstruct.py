"""Stage 3: reconstruct the 200-D policy obs in the MuJoCo env (no IsaacLab).

Confirmed layout (empirically pinned by probe_obs_layout.py — term-major,
history oldest->newest, current step last):

  obs[0:30]    POSE block A   (6-D x 5)
  obs[30:65]   prev_actions   (7-D x 5)   = action[t-d-1]
  obs[65:110]  joint_pos      (9-D x 5)   = raw_state[0:9]
  obs[110:140] POSE block B   (6-D x 5)
  obs[140:170] POSE block C   (6-D x 5)
  obs[170:200] POSE block D   (6-D x 5)

The four 6-D POSE blocks are {ee_pose, peg_in_hand, hole_in_hand, peg_in_hole};
which block is which is resolved empirically by gate3 (no guessing).

Each pose term = [pos(3), axis_angle(3)] of target in a frame, per
target_asset_pose_in_root_asset_frame:
  ee_pose       : panda_hand in robot-root(panda_link0) frame
  peg_in_hand   : insertive_object(peg) in panda_hand frame
  hole_in_hand  : receptive_object(hole) in panda_hand frame
  peg_in_hole   : peg in hole frame

raw_state (57) layout (env-local; world quat/vel):
  [0:9] joint_pos  [9:18] joint_vel
  [18:25] robot root pose(7)  [25:31] root vel(6)
  [31:38] peg root pose(7)    [38:44] peg vel(6)
  [44:51] hole root pose(7)   [51:57] hole vel(6)
"""
import os

import numpy as np
import mujoco

import quat_utils as Q

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)  # omnireset_sim2sim
PANDA_XML = os.path.join(PROJ, "assets", "mujoco_menagerie", "franka_emika_panda", "panda.xml")

# obs block boundaries (start, width_per_step, n_hist)
HIST = 5
POSE_W, ACT_W, JP_W = 6, 7, 9
BLOCKS = {
    "poseA":        (0, POSE_W),
    "prev_actions": (30, ACT_W),
    "joint_pos":    (65, JP_W),
    "poseB":        (110, POSE_W),
    "poseC":        (140, POSE_W),
    "poseD":        (170, POSE_W),
}
POSE_BLOCK_NAMES = ["poseA", "poseB", "poseC", "poseD"]
POSE_TERM_NAMES = ["ee_pose", "peg_in_hand", "hole_in_hand", "peg_in_hole"]


class FrankaFK:
    """Minimal MuJoCo Franka used only for forward kinematics of panda_hand."""

    def __init__(self, xml=PANDA_XML):
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        self.hand_bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.root_bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "link0")

    def hand_in_root(self, joint_pos9):
        """Return (pos, quat) of panda_hand in robot-root frame for the given qpos."""
        self.d.qpos[:9] = joint_pos9
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        hp, hq = self.d.xpos[self.hand_bid].copy(), self.d.xquat[self.hand_bid].copy()
        rp, rq = self.d.xpos[self.root_bid].copy(), self.d.xquat[self.root_bid].copy()
        t, q = Q.subtract_frame_transforms(rp, rq, hp, hq)
        return t, q


def pose_terms_from_raw(raw, fk: FrankaFK):
    """Compute the 4 candidate 6-D pose terms [pos, axis_angle] from one raw_state.

    Returns dict term_name -> (6,) and also the hand-in-root (pos,quat) for reuse.
    """
    jp = raw[0:9]
    root_pos, root_quat = raw[18:21], raw[21:25]        # robot root world (env-local)
    peg_pos, peg_quat = raw[31:34], raw[34:38]
    hole_pos, hole_quat = raw[44:47], raw[47:51]

    hir_t, hir_q = fk.hand_in_root(jp)                   # panda_hand in root (from FK)
    # hand world pose in IsaacLab env-local frame = root_world ∘ hand_in_root
    hw_t, hw_q = Q.combine_frame_transforms(root_pos, root_quat, hir_t, hir_q)

    ee_pose = np.concatenate([hir_t, Q.axis_angle_from_quat(hir_q)])
    peg_in_hand = Q.pose_in_root(hw_t, hw_q, peg_pos, peg_quat)
    hole_in_hand = Q.pose_in_root(hw_t, hw_q, hole_pos, hole_quat)
    peg_in_hole = Q.pose_in_root(hole_pos, hole_quat, peg_pos, peg_quat)
    return {
        "ee_pose": ee_pose,
        "peg_in_hand": peg_in_hand,
        "hole_in_hand": hole_in_hand,
        "peg_in_hole": peg_in_hole,
    }, (hir_t, hir_q)
