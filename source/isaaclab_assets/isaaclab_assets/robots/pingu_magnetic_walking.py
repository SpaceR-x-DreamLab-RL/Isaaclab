# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Pingu robot, assembled at spawn time from separate body and arms USD files."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

import isaacsim.core.utils.prims as prim_utils
import omni.log
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import schemas
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_usd_path_with_timeout

##
# Spawner: assembles the body + arms USD files into a single articulation.
##

_ARM_BASE_JOINT_NAMES = ("left_base_joint", "right_base_joint")

# The real robot's arms are mounted 5cm higher (in base_link's local +Z) than the split USD files'
# geometric attachment point -- applied to the arm-to-body joint frame below rather than editing the
# USD, since it's a real physical calibration offset, not a geometry bug.
_ARM_MOUNT_Z_OFFSET = 0.05


def _frame_matrix(pos: Gf.Vec3f, rot: Gf.Quatf) -> Gf.Matrix4d:
    """Build a Gf.Matrix4d representing a joint's local frame (position + rotation)."""
    matrix = Gf.Matrix4d().SetRotate(Gf.Quatd(rot.GetReal(), Gf.Vec3d(*rot.GetImaginary())))
    matrix.SetRow3(3, Gf.Vec3d(*pos))
    return matrix


def _matrix_to_pos_quat(matrix: Gf.Matrix4d) -> tuple[Gf.Vec3f, Gf.Quatf]:
    """Decompose a Gf.Matrix4d into a (position, rotation) pair for joint local-frame attributes."""
    translation = matrix.ExtractTranslation()
    quat = matrix.ExtractRotationQuat().GetNormalized()
    imaginary = quat.GetImaginary()
    return (
        Gf.Vec3f(translation[0], translation[1], translation[2]),
        Gf.Quatf(quat.GetReal(), imaginary[0], imaginary[1], imaginary[2]),
    )


@clone
def spawn_pingu_body_and_arms(
    prim_path: str,
    cfg: PinguBodyArmsCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the Pingu body and arms from separate USD files and stitch them into one articulation.

    The body USD (default prim carries the ``ArticulationRootAPI`` and ``base_link``) is referenced at
    ``prim_path``. The arms USD (its own standalone articulation, with a local ``world`` anchor so it can
    be inspected on its own) is referenced as a child at ``{prim_path}/{cfg.arms_prim_name}``. To turn the
    two into a single articulation:

    - the arms' ``ArticulationRootAPI`` is removed (an articulation can only have one root),
    - its ``left_base_joint``/``right_base_joint`` are re-targeted from the arms' local ``world`` prim to
      the body's ``base_link``: each joint's body0-side local frame is recomputed (via the two anchors'
      current world transforms) so its world-space attachment point/orientation is unchanged, only what
      it's expressed relative to changes. The body1-side frame (relative to the arm link) is left as
      authored, since that body didn't move. (An earlier version of this function copied attachment
      frames from a different, no-longer-used combined asset -- that asset's arm links carried a baked
      rotation that this split arms file's don't, which silently mis-rotated/destabilized the right arm.)
    - the arms' now-unused ``world`` prim and ``scene_to_world`` joint are deactivated.

    Args:
        prim_path: The prim path or pattern to spawn the asset at.
        cfg: The configuration instance.
        translation: The translation to apply to the body prim. Defaults to None.
        orientation: The orientation in (w, x, y, z) to apply to the body prim. Defaults to None.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned, stitched-together robot.

    Raises:
        FileNotFoundError: If the body or arms USD file does not exist at the given path.
        RuntimeError: If the expected arm-to-body joints are not found in the arms USD file.
    """
    if not check_usd_path_with_timeout(cfg.body_usd_path):
        raise FileNotFoundError(f"Body USD file not found at path: '{cfg.body_usd_path}'.")
    if not check_usd_path_with_timeout(cfg.arms_usd_path):
        raise FileNotFoundError(f"Arms USD file not found at path: '{cfg.arms_usd_path}'.")

    if prim_utils.is_prim_path_valid(prim_path):
        omni.log.warn(f"A prim already exists at prim path: '{prim_path}'.")
    else:
        # reference the body -- this prim carries the ArticulationRootAPI and `base_link`
        prim_utils.create_prim(
            prim_path,
            usd_path=cfg.body_usd_path,
            translation=translation,
            orientation=orientation,
            scale=cfg.scale,
        )
        # reference the arms as a child -- it comes with its own (redundant) articulation root
        arms_prim_path = f"{prim_path}/{cfg.arms_prim_name}"
        prim_utils.create_prim(arms_prim_path, usd_path=cfg.arms_usd_path)

        stage = get_current_stage()
        arms_prim = stage.GetPrimAtPath(arms_prim_path)
        if arms_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            arms_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)

        base_link_path = f"{prim_path}/{cfg.base_link_name}"
        base_link_prim = stage.GetPrimAtPath(base_link_path)
        arms_world_prim = stage.GetPrimAtPath(f"{arms_prim_path}/world")
        if not base_link_prim.IsValid() or not arms_world_prim.IsValid():
            raise RuntimeError(
                f"Expected prims not found: '{base_link_path}' and/or '{arms_prim_path}/world'."
            )
        # world-space transform of the two anchors the joints are currently expressed relative to (arms'
        # local "world") and the one we're re-targeting to (the body's base_link) -- used below to carry
        # each joint's *body0*-side frame across the re-target without moving its world-space attachment
        # point. The body1-side frame (relative to left_base/right_base) is untouched: that body didn't move.
        arms_world_to_world = UsdGeom.Xformable(arms_world_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        base_link_to_world = UsdGeom.Xformable(base_link_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        for joint_name in _ARM_BASE_JOINT_NAMES:
            joint_prim = stage.GetPrimAtPath(f"{arms_prim_path}/joints/{joint_name}")
            if not joint_prim.IsValid():
                raise RuntimeError(f"Expected joint prim not found: '{arms_prim_path}/joints/{joint_name}'.")
            joint = UsdPhysics.Joint(joint_prim)
            old_frame = _frame_matrix(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
            world_frame = old_frame * arms_world_to_world
            new_frame = world_frame * base_link_to_world.GetInverse()
            new_pos0, new_rot0 = _matrix_to_pos_quat(new_frame)
            new_pos0 = Gf.Vec3f(new_pos0[0], new_pos0[1], new_pos0[2] + _ARM_MOUNT_Z_OFFSET)

            joint.GetBody0Rel().SetTargets([Sdf.Path(base_link_path)])
            joint.GetLocalPos0Attr().Set(new_pos0)
            joint.GetLocalRot0Attr().Set(new_rot0)

        # the arms' own world anchor is now unused -- deactivate it and the joint that fixed it to world
        for unused_path in (f"{arms_prim_path}/joints/scene_to_world", f"{arms_prim_path}/world"):
            unused_prim = stage.GetPrimAtPath(unused_path)
            if unused_prim.IsValid():
                unused_prim.SetActive(False)

        # The base/shoulder links mount close enough to base_link's own body mesh that their collision
        # geometry overlaps it at rest (confirmed via bounding-box check and, definitively, by a live
        # test: temporarily offsetting the arms clear of the body eliminated an otherwise-persistent
        # undamped base_joint spin entirely). Exclude base/shoulder-vs-own-body collision response --
        # collision geometry (and thus auto-computed mass) is untouched, so this only suppresses contact
        # force against base_link specifically; arms still collide normally with everything else
        # (including each other), so the arm_contact_forces reward penalty still works for genuine
        # external collisions. The elbows are intentionally left out of this filter: with the 5cm mount
        # raise + calibrated joint limits above, they no longer geometrically overlap base_link at rest,
        # and elbow-vs-body contact is a real, physically-meaningful collision (e.g. arm folding against
        # the body) that should still register.
        for arm_link_name in ("left_base", "right_base", "left_shoulder", "right_shoulder"):
            arm_link_prim = stage.GetPrimAtPath(f"{arms_prim_path}/{arm_link_name}")
            if arm_link_prim.IsValid():
                filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(arm_link_prim)
                filtered_pairs.GetFilteredPairsRel().AddTarget(Sdf.Path(base_link_path))

    # modify rigid body / collision / mass / articulation root properties on the combined prim
    if cfg.rigid_props is not None:
        schemas.modify_rigid_body_properties(prim_path, cfg.rigid_props)
    if cfg.collision_props is not None:
        schemas.modify_collision_properties(prim_path, cfg.collision_props)
    if cfg.mass_props is not None:
        schemas.modify_mass_properties(prim_path, cfg.mass_props)
    if cfg.articulation_props is not None:
        schemas.modify_articulation_root_properties(prim_path, cfg.articulation_props)

    return prim_utils.get_prim_at_path(prim_path)


@configclass
class PinguBodyArmsCfg(RigidObjectSpawnerCfg):
    """Configuration for spawning the Pingu robot from separate body/arms USD files, stitched together."""

    func: Callable = spawn_pingu_body_and_arms

    body_usd_path: str = MISSING
    """Path to the USD file containing the Pingu body (base_link, thrusters, articulation root)."""

    arms_usd_path: str = MISSING
    """Path to the USD file containing the Pingu arms (left/right shoulder-elbow-tool chains)."""

    articulation_props: schemas.ArticulationRootPropertiesCfg | None = None
    """Properties to apply to the (body's) articulation root."""

    scale: tuple[float, float, float] | None = None
    """Scale of the body asset. Defaults to None, in which case the scale is not modified."""

    base_link_name: str = "base_link"
    """Name of the body prim that the arms are rigidly mounted to."""

    arms_prim_name: str = "arms"
    """Name of the child prim under which the arms USD is referenced."""


##
# Configuration
##

PINGU_MAGNETIC_WALKING_CFG = ArticulationCfg(
    spawn=PinguBodyArmsCfg(
        body_usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/SpaceR-TheDreamLab/UniluFP_RL/pingu_body_rotated.usdc",
        arms_usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/SpaceR-TheDreamLab/UniluFP_RL/pingu_arms_rotated.usdc",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),
        joint_pos={
            ".*shoulder_joint": 0.0,
            "right_elbow_joint": 0.4,
            "left_elbow_joint": -0.4,
        },
    ),
    actuators={
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["left_shoulder_joint", "left_elbow_joint"],
            effort_limit_sim=15.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["right_shoulder_joint", "right_elbow_joint"],
            effort_limit_sim=15.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "reaction_wheel": ImplicitActuatorCfg(
            joint_names_expr=["rw_joint"],
            effort_limit_sim=40000.0,
            velocity_limit_sim=100000.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for the Pingu robot, assembled at spawn time from separate body and arms USD files."""
