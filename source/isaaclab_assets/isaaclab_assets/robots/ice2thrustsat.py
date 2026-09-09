# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD generator for a 6U CubeSat body with attitude thrusters (cones) and a main thruster.

Usage (inside the task setup):

    from cubesat6u import CubeSat6UProps, generate_cubesat6u, CUBESAT6U_CFG

    # Option A: spawn via ArticulationCfg (recommended)
    from omni.isaac.lab.assets import Articulation
    from omni.isaac.lab.sim import SimulationContext

    # ... create sim & scene ...
    # scene.articulations["cubesat"] = Articulation(CUBESAT6U_CFG)

    # Option B: call generator directly for ad-hoc spawning
    # generate_cubesat6u("/World/CubeSat6U", CubeSat6UProps())

This module follows the structure of intball2.py with
fixed joints for thruster visuals and mass/collision properties on the sat-bus.
"""

from __future__ import annotations

import math
from dataclasses import field

import numpy as np
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import isaaclab.sim as sim_utils
import omni.physx.scripts.utils as physx_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.schemas import schemas, schemas_cfg
from isaaclab.utils import configclass
from pxr import Gf, Sdf, UsdGeom, UsdPhysics


# Configuration

@configclass
class CubeSat6UProps:
    """Physical and geometric properties for a 6U CubeSat.

    Default body is 0.204 m x 0.100 m x 0.352 m (W x D x H), based on reference spacecraft sizing.
    Attitude thrusters are 8 small cones at specific positions; a single main thruster is placed
    at [0.102, 0, 0] pointing in +X direction.
    """

    # Body dimensions (meters): width (X), depth (Y), height (Z)
    # Based on reference script thruster positions: [+/-0.102, +/-0.050, +/-0.176]
    width: float = 0.204  # 2 * 0.102
    depth: float = 0.100  # 2 * 0.050  
    height: float = 0.352  # 2 * 0.176

    # Mass properties
    mass: float = 12.0
    CoM: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Inertia matrix from reference script (kg·m²)
    inertia: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        (0.144, 0.0, 0.0),
        (0.0, 0.185, 0.0),
        (0.0, 0.0, 0.061)
    )

    enable_collision: bool = True

    # Visual/geometry refinement for the body mesh
    body_refinement_enable: bool = False
    body_refinement_level: int = 1

    # Attitude thrusters (ACS) — geometry (cones) and placement
    acs_cone_radius: float = 0.01
    acs_cone_height: float = 0.04
    acs_margin_from_face: float = 0.01  # meter inset so cones don't Z-fight the surface

    # Main thruster (e.g., delta-v) — geometry and placement
    main_cone_radius: float = 0.025
    main_cone_height: float = 0.08
    main_on_negative_x: bool = True  # if False, places on +X face

    # If provided, these override default corner layout. Positions in meters, directions unit-length.
    acs_positions: list[tuple[float, float, float]] = field(default_factory=list)
    acs_directions: list[tuple[float, float, float]] = field(default_factory=list)

    def __post_init__(self):
        assert self.width > 0 and self.depth > 0 and self.height > 0, "Dimensions must be positive."
        assert self.mass > 0, "Mass must be positive."
        assert len(self.CoM) == 3, "CoM must be a 3-tuple."

        # If no custom ACS layout is provided, use exact positions from reference script
        if not self.acs_positions or not self.acs_directions:
            # Exact thruster positions from reference script (meters)
            self.acs_positions = [
                (0.102,  0.050,  0.176), (-0.102,  0.050, -0.176),
                (0.102, -0.050,  0.176), (-0.102, -0.050, -0.176),
                (-0.102, -0.050,  0.176), (0.102,  0.050, -0.176),
                (-0.102,  0.050,  0.176), (0.102, -0.050, -0.176)
            ]

            # Exact thrust directions from reference script (already normalized)
            self.acs_directions = [
                (0.186,  0.695, -0.695), (-0.186,  0.695,  0.695),
                (0.186, -0.695, -0.695), (-0.186, -0.695,  0.695),
                (-0.186, -0.695, -0.695), (0.186,  0.695,  0.695),
                (-0.186,  0.695, -0.695), (0.186, -0.695,  0.695),
            ]


# USD Generator

def _quat_align(v_from: Gf.Vec3f, v_to: Gf.Vec3f) -> Gf.Quatd:
    """Quaternion rotating v_from onto v_to.

    Uses Gf.Rotation like intball2.py for consistency.
    """
    # Convert to Vec3d for Gf.Rotation (like intball2.py)
    from_vec = Gf.Vec3d(v_from[0], v_from[1], v_from[2])
    to_vec = Gf.Vec3d(v_to[0], v_to[1], v_to[2])
    
    # Use Gf.Rotation like intball2.py
    rotation = Gf.Rotation(from_vec, to_vec)
    quaternion = rotation.GetQuaternion()
    
    # Return Quatd (like intball2.py)
    return Gf.Quatd(quaternion.GetReal(), *quaternion.GetImaginary())


def _add_cone(parent_path: str, name: str, radius: float, height: float,
              position: Gf.Vec3f, direction: Gf.Vec3f, visual_offset: float = 0.0) -> str:
    """Create a cone (tip outward) under an Xform and orient it along `direction`.

    The cone's local +Z axis is aligned with `direction`. Its base is near the bus face,
    and its tip points outward by `height`.
    """
    thr_path = f"{parent_path}/{name}"

    # Create an Xform prim as the thruster frame
    thr_prim = prim_utils.create_prim(thr_path)
    
    # Apply visual offset to position cone base on satellite surface
    offset_position = Gf.Vec3f(
        position[0] + direction[0] * visual_offset,
        position[1] + direction[1] * visual_offset,
        position[2] + direction[2] * visual_offset
    )
    thr_prim.GetAttribute("xformOp:translate").Set(offset_position)

    # Orient +Z to `direction` (flipped to point inward)
    q = _quat_align(Gf.Vec3f(0, 0, 1), Gf.Vec3f(-direction[0], -direction[1], -direction[2]))
    thr_prim.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatd).Set(q)

    # Cone geometry (by default aligned to local +Z), with base sitting at z = 0
    geom_path = f"{thr_path}/Geometry"
    cone = prim_utils.create_prim(
        prim_path=geom_path,
        prim_type="Cone",
        attributes={"radius": float(radius), "height": float(height)},
    )
    # Lift the cone so that its base is just at the frame origin and the tip extends outward
    cone.GetAttribute("xformOp:translate").Set(Gf.Vec3f(0.0, 0.0, height * 0.5))

    # Enable physics authoring (massless visual by default; mass set on parent via schemas if needed)
    physx_utils.setPhysics(prim=thr_prim, kinematic=False)

    return thr_path


def _apply_mass_props_with_inertia(body_path: str, mass: float, com_xyz, inertia_3x3):
    """
    Apply mass/CoM/inertia to `body_path`. Accepts full 3x3 inertia about CoM (kg·m²).
    Tries Isaac Lab schema first; falls back to USD Physics MassAPI with
    principal moments + axes.
    """
    # First attempt: Isaac Lab schemas (if inertia is supported in your build)
    try:
        mp = schemas_cfg.MassPropertiesCfg(
            mass=float(mass),
            center_of_mass=tuple(map(float, com_xyz)),
            inertia=tuple(tuple(float(x) for x in row) for row in inertia_3x3),  # 3x3
        )
        schemas.define_mass_properties(body_path, mp)
        return
    except Exception:
        pass

    # Fallback: USD Physics MassAPI (diagonalize inertia)
    stage = stage_utils.get_current_stage()
    prim = stage.GetPrimAtPath(body_path)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(x) for x in com_xyz]))

    I = np.array(inertia_3x3, dtype=float)
    # Ensure symmetric (small numerical cleanup)
    I = 0.5 * (I + I.T)
    
    # For diagonal inertia matrix, we can use the diagonal elements directly
    # and set principal axes to identity (no rotation needed)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(float(I[0,0]), float(I[1,1]), float(I[2,2])))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))  # Identity quaternion (float)


def generate_cubesat6u(root_path: str, cfg: CubeSat6UProps):
    """Builds the 6U CubeSat USD subtree at `root_path`.

    - Bus is a scaled Cube (size=1, scaled to given W x D x H)
    - 8 ACS thrusters as cones (fixed joints to bus)
    - 1 main thruster cone on ±X face center (fixed joint to bus)
    """
    stage = stage_utils.get_current_stage()

    # Define articulation & physics schemas
    art_props = schemas_cfg.ArticulationRootPropertiesCfg(
        articulation_enabled=True,
        enabled_self_collisions=False,
        fix_root_link=False,
    )
    collider_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=cfg.enable_collision)
    mass_props = schemas_cfg.MassPropertiesCfg(mass=cfg.mass, center_of_mass=cfg.CoM)
    massless = schemas_cfg.MassPropertiesCfg(mass=1e-5)

    # Root prim
    prim_utils.create_prim(root_path)
    schemas.define_articulation_root_properties(root_path, art_props)

    # Body bus (scaled cube)
    body_path = f"{root_path}/body"
    body_prim = prim_utils.create_prim(
        prim_path=body_path,
        prim_type="Cube",
        attributes={"size": 1.0},
        scale=[cfg.width, cfg.depth, cfg.height],
    )

    if cfg.body_refinement_enable:
        body_prim.CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(cfg.body_refinement_level)
        body_prim.CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool).Set(True)

    # PhysX authoring
    physx_utils.setPhysics(prim=body_prim, kinematic=False)
    schemas.define_mass_properties(body_path, mass_props)
    schemas.define_collision_properties(body_path, collider_props)
    if cfg.inertia is not None:
        _apply_mass_props_with_inertia(
            body_path=body_path,
            mass=cfg.mass,
            com_xyz=cfg.CoM,
            inertia_3x3=cfg.inertia,   # 3x3 about CoM, kg·m²
        )
    else:
        schemas.define_mass_properties(body_path, mass_props)

    # Create a dummy link to avoid articulation registration quirks (mirrors the IntBall2 pattern)
    dummy_path = f"{root_path}/dummy"
    dummy_prim = prim_utils.create_prim(dummy_path)
    physx_utils.setPhysics(prim=dummy_prim, kinematic=False)
    schemas.define_mass_properties(dummy_path, massless)
    schemas.createJoint(
        stage=stage,
        joint_type="Revolute",
        from_prim=body_prim,
        to_prim=dummy_prim,
        joint_name="dummy_joint",
        joint_base_path=f"{root_path}/joints",
    ).GetAttribute("physics:axis").Set("Z")

    # ACS thrusters (small cones) — fixed to body
    assert len(cfg.acs_positions) == len(cfg.acs_directions), (
        "acs_positions and acs_directions must have the same length"
    )

    for i, (pos, dirn) in enumerate(zip(cfg.acs_positions, cfg.acs_directions)):
        thr_path = _add_cone(
            parent_path=root_path,
            name=f"acs_thruster_{i}",
            radius=cfg.acs_cone_radius,
            height=cfg.acs_cone_height,
            position=Gf.Vec3f(*pos),
            direction=Gf.Vec3f(*dirn),
            visual_offset=0.01,  # Small offset for visual - places cone base on satellite surface
        )
        # Massless visual link fixed to body
        schemas.define_mass_properties(thr_path, massless)
        schemas.createJoint(
            stage=stage,
            joint_type="Fixed",
            from_prim=stage.GetPrimAtPath(thr_path),
            to_prim=body_prim,
            joint_name=f"fixed_joint_acs_{i}",
            joint_base_path=f"{root_path}/joints",
        )

    # Main thruster (larger cone) — fixed to body
    # Use exact position from reference script with small offset to make it visible
    main_pos = Gf.Vec3f(0.102 + 0.02, 0.0, 0.0)  # Small offset to make cone visible, the offset is not used for the actual thrust vector in the simulation, just for visual.
    main_dir = Gf.Vec3f(1.0, 0.0, 0.0)  # Points in +X direction

    main_path = _add_cone(
        parent_path=root_path,
        name="main_thruster",
        radius=cfg.main_cone_radius,
        height=cfg.main_cone_height,
        position=main_pos,
        direction=main_dir,
    )
    schemas.define_mass_properties(main_path, massless)
    schemas.createJoint(
        stage=stage,
        joint_type="Fixed",
        from_prim=stage.GetPrimAtPath(main_path),
        to_prim=body_prim,
        joint_name="fixed_joint_main",
        joint_base_path=f"{root_path}/joints",
    )

    return root_path

ICE2THRUST_6UCSAT_CFG = ArticulationCfg(
    spawn=sim_utils.RobotFromCodeCfg(
        robot_gen_func=generate_cubesat6u,
        robot_gen_props=CubeSat6UProps(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
            disable_gravity=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={},
    ),
    actuators={},  # visuals only; add custom force/impulse mapping in your task code
)