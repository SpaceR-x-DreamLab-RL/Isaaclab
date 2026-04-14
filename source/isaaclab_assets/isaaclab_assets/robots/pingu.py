# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple ackermann robot."""


from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

PINGU_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/SpaceR-TheDreamLab/UniluFP_RL/unilu_FP_pingu_rotated.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
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
            joint_names_expr=["reaction_wheel_joint"],
            effort_limit_sim=40000.0,
            velocity_limit_sim=100000.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for a simple floating platform robot."""
