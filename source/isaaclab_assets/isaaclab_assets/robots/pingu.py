# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple ackermann robot."""


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##

PINGU_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/SpaceR-TheDreamLab/Pingu/pingu.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            rigid_body_enabled=True,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
            # retain_accelerations=False,
            # linear_damping=0.0,
            # angular_damping=0.0,
            # max_linear_velocity=1000.0,
            # max_angular_velocity=1000.0,
            # max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            # sleep_threshold=0.005,
            # stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),
        rot=(0.924, 0.0, 0.0, 0.383),
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
            joint_names_expr=["rw_revolute_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=0.5,
        ),
    },
)
"""Configuration for a simple floating platform robot."""
