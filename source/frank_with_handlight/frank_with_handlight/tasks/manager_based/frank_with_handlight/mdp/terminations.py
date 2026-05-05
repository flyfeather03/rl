# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from .observations import object_grasped_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def success_lifted_and_grasped(
    env: ManagerBasedRLEnv,
    target_height: float = 0.14,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> bool:
    """Success when object is grasped and lifted above target height."""
    obj: RigidObject = env.scene[object_cfg.name]
    return object_grasped_mask(env) & (obj.data.root_pos_w[:, 2] > target_height)
