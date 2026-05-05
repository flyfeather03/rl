# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
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
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Stage-2 reward: stable object grasp before lifting."""
    ee_pos, obj_pos = _ee_and_object_pos(env, object_cfg, ee_frame_cfg)
    dist = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1)

    grasped = object_grasped_mask(env)
    lifted = obj_pos[:, 2] > min_lift
    stage_mask = grasped & (~lifted)

    return stage_mask.float() * (1.0 - torch.tanh(dist / pose_std))


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
