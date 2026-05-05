# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim import CuboidCfg, RigidBodyMaterialCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


@configclass
class ThreeStageObjectTableSceneCfg(InteractiveSceneCfg):
    """Scene for Franka three-stage lift with random long cuboids."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.52, 0.0, 0.045], rot=[1, 0, 0, 0]),
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                CuboidCfg(size=(0.12, 0.035, 0.035), physics_material=RigidBodyMaterialCfg(static_friction=0.9)),
                CuboidCfg(size=(0.10, 0.04, 0.03), physics_material=RigidBodyMaterialCfg(static_friction=0.9)),
                CuboidCfg(size=(0.14, 0.03, 0.04), physics_material=RigidBodyMaterialCfg(static_friction=0.9)),
                CuboidCfg(size=(0.09, 0.05, 0.03), physics_material=RigidBodyMaterialCfg(static_friction=0.9)),
            ],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
            random_choice=True,
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.707, 0.0, 0.0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # Two point-like lights mounted under panda_hand, so they move with gripper.
    grip_light_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/GripLightLeft",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.018, 0.015]),
        spawn=sim_utils.SphereLightCfg(color=(1.0, 1.0, 1.0), intensity=2500.0, radius=0.01),
    )
    grip_light_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/GripLightRight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, -0.018, 0.015]),
        spawn=sim_utils.SphereLightCfg(color=(1.0, 1.0, 1.0), intensity=2500.0, radius=0.01),
    )

    world_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2200.0),
    )

    # Finger contact sensor for grasp-stage reward.
    contact_grasp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_.*finger",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
    )

    # Global camera for visualization/debug (not required in policy observations).
    global_camera = CameraCfg(
        prim_path="/World/GlobalCamera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 50.0)),
        offset=CameraCfg.OffsetCfg(pos=(1.2, 0.0, 1.0), rot=(-0.3799, 0.5963, 0.5963, -0.3799), convention="ros"),
    )


@configclass
class CommandsCfg:
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.45, 0.62),
            pos_y=(-0.22, 0.22),
            pos_z=(0.15, 0.35),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        grasp = ObsTerm(func=mdp.object_grasped_float)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.10, 0.10), "y": (-0.22, 0.22), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    stage_move = RewTerm(func=mdp.move_stage_reward, params={"std": 0.08, "pregrasp_z": 0.08, "min_lift": 0.08}, weight=2.0)

    stage_grasp = RewTerm(func=mdp.grasp_stage_reward, params={"pose_std": 0.06, "min_lift": 0.08}, weight=5.0)

    stage_lift = RewTerm(
        func=mdp.lift_stage_reward,
        params={"min_lift": 0.08, "target_height": 0.14},
        weight=8.0,
    )

    success_bonus = RewTerm(func=mdp.lift_success_bonus, params={"target_height": 0.14}, weight=10.0)

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )

    success = DoneTerm(func=mdp.success_lifted_and_grasped, params={"target_height": 0.14})


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 50000},
    )


@configclass
class FrankaThreeStageLiftEnvCfg(ManagerBasedRLEnvCfg):
    scene: ThreeStageObjectTableSceneCfg = ThreeStageObjectTableSceneCfg(
        num_envs=1024,
        env_spacing=2.5,
        replicate_physics=False,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 6.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
