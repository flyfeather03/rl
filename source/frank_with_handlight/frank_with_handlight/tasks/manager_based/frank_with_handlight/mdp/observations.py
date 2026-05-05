# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object position expressed in robot base frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    object_pos_w = obj.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def object_grasped_mask(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    diff_threshold: float = 0.06,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Return grasp state per environment for a parallel gripper setup."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_close = torch.linalg.vector_norm(object_pos - ee_pos, dim=1) < diff_threshold

    # If finger contact sensor exists, require both fingers to have force over threshold.
    if "contact_grasp" in env.scene.keys() and env.scene["contact_grasp"] is not None:
        contact_force = env.scene["contact_grasp"].data.net_forces_w
        force_norm = torch.linalg.vector_norm(contact_force, dim=2)
        both_fingers_ok = torch.all(force_norm > force_threshold, dim=1)
        pose_close = torch.logical_and(pose_close, both_fingers_ok)

    # Require gripper to be closed enough compared to the configured open position.
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        left_closed = (
            torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]]) - env.cfg.gripper_open_val)
            > env.cfg.gripper_threshold
        )
        right_closed = (
            torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]]) - env.cfg.gripper_open_val)
            > env.cfg.gripper_threshold
        )
        pose_close = torch.logical_and(pose_close, left_closed)
        pose_close = torch.logical_and(pose_close, right_closed)

    return pose_close


def object_grasped_float(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    diff_threshold: float = 0.06,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Float view of grasp state (0 or 1), useful in observation vectors."""
    return object_grasped_mask(
        env,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        object_cfg=object_cfg,
        diff_threshold=diff_threshold,
        force_threshold=force_threshold,
    ).float().unsqueeze(-1)
