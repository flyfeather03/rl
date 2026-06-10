# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from .observations import object_grasped_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ee_and_object_pos(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[:, 0, :], obj.data.root_pos_w


def _gripper_closed_progress(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return 0 for open gripper and 1 for fully closed gripper."""
    robot: Articulation = env.scene[robot_cfg.name]
    if not hasattr(env.cfg, "gripper_joint_names"):
        return torch.zeros(env.num_envs, device=robot.data.joint_pos.device)

    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    finger_pos = torch.abs(robot.data.joint_pos[:, gripper_joint_ids])
    mean_finger_pos = torch.mean(finger_pos, dim=1)

    open_val = getattr(env.cfg, "gripper_open_val", 0.04)
    closed_val = getattr(env.cfg, "gripper_closed_val", 0.0)
    denom = max(open_val - closed_val, 1e-6)
    return torch.clamp((open_val - mean_finger_pos) / denom, min=0.0, max=1.0)


def _finger_contact_progress(env: ManagerBasedRLEnv, force_threshold: float) -> torch.Tensor:
    """Return a bounded contact signal from the gripper contact sensor."""
    if "contact_grasp" not in env.scene.keys() or env.scene["contact_grasp"] is None:
        return torch.zeros(env.num_envs, device=env.device)

    contact_force = env.scene["contact_grasp"].data.net_forces_w
    force_norm = torch.linalg.vector_norm(contact_force, dim=2)
    max_force = torch.max(force_norm, dim=1).values
    return torch.clamp(max_force / max(force_threshold, 1e-6), min=0.0, max=1.0)


def move_stage_reward(
    env: ManagerBasedRLEnv,
    std: float,
    pregrasp_z: float = 0.08,
    min_lift: float = 0.08,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Stage-1 reward: move end-effector to pre-grasp pose above object."""
    ee_pos, obj_pos = _ee_and_object_pos(env, object_cfg, ee_frame_cfg)

    pregrasp = obj_pos.clone()
    pregrasp[:, 2] += pregrasp_z
    dist = torch.linalg.vector_norm(pregrasp - ee_pos, dim=1)

    grasped = object_grasped_mask(env)
    lifted = obj_pos[:, 2] > min_lift
    stage_mask = (~grasped) & (~lifted)

    return stage_mask.float() * (1.0 - torch.tanh(dist / std))


def grasp_stage_reward(
    env: ManagerBasedRLEnv,
    pose_std: float = 0.06,
    min_lift: float = 0.08,
    close_distance: float = 0.09,
    close_bonus: float = 0.6,
    contact_bonus: float = 0.3,
    grasp_bonus: float = 1.0,
    force_threshold: float = 0.2,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Stage-2 reward: approach the object, close the gripper, and establish contact."""
    ee_pos, obj_pos = _ee_and_object_pos(env, object_cfg, ee_frame_cfg)
    dist = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1)

    grasped = object_grasped_mask(env)
    lifted = obj_pos[:, 2] > min_lift
    stage_mask = ~lifted

    pose_reward = 1.0 - torch.tanh(dist / pose_std)
    near_object = dist < close_distance
    close_reward = _gripper_closed_progress(env, robot_cfg)
    contact_reward = _finger_contact_progress(env, force_threshold)

    reward = pose_reward
    reward += near_object.float() * close_bonus * close_reward
    reward += near_object.float() * contact_bonus * contact_reward
    reward += grasp_bonus * grasped.float()

    return stage_mask.float() * reward


def lift_stage_reward(
    env: ManagerBasedRLEnv,
    min_lift: float = 0.08,
    target_height: float = 0.14,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Stage-3 reward: lift while maintaining grasp."""
    obj: RigidObject = env.scene[object_cfg.name]
    height = obj.data.root_pos_w[:, 2]

    grasped = object_grasped_mask(env)
    stage_mask = grasped & (height > min_lift)

    height_progress = torch.clamp((height - min_lift) / max(target_height - min_lift, 1e-6), min=0.0, max=1.0)
    return stage_mask.float() * height_progress


def lift_success_bonus(
    env: ManagerBasedRLEnv,
    target_height: float = 0.14,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminal-style bonus for grasp + target lift reached."""
    obj: RigidObject = env.scene[object_cfg.name]
    success = object_grasped_mask(env) & (obj.data.root_pos_w[:, 2] > target_height)
    return success.float()
