# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime Panda/Research-3 naming contract for shared evaluation scripts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FrankaRobotContract:
    model: str
    arm_joint_names: tuple[str, ...]
    finger_joint_names: tuple[str, str]
    hand_body_name: str


def resolve_franka_robot_contract(robot) -> FrankaRobotContract:
    joint_names = set(robot.joint_names)
    body_names = set(robot.body_names)
    candidates = (
        FrankaRobotContract(
            model="research3",
            arm_joint_names=tuple(f"fr3_joint{i}" for i in range(1, 8)),
            finger_joint_names=("fr3_finger_joint1", "fr3_finger_joint2"),
            hand_body_name="fr3_hand",
        ),
        FrankaRobotContract(
            model="panda",
            arm_joint_names=tuple(f"panda_joint{i}" for i in range(1, 8)),
            finger_joint_names=("panda_finger_joint1", "panda_finger_joint2"),
            hand_body_name="panda_hand",
        ),
    )
    for contract in candidates:
        if set(contract.arm_joint_names).issubset(joint_names) and contract.hand_body_name in body_names:
            missing_fingers = set(contract.finger_joint_names) - joint_names
            if missing_fingers:
                raise RuntimeError(
                    f"{contract.model} asset is missing finger joints: {sorted(missing_fingers)}"
                )
            return contract
    raise RuntimeError(
        "Unsupported Franka articulation; expected panda_joint*/panda_hand or "
        "fr3_joint*/fr3_hand"
    )
