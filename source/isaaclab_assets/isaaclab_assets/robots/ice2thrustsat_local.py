# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of Ice2Thrust 6U CubeSat Design."""

# import isaaclab.sim as sim_utils
# from isaaclab.assets import ArticulationCfg
# from isaaclab.utils.assets import REPO_ROOT_PATH
from isaaclab_assets.robots.ice2thrustsat import ICE2THRUST_6UCSAT_CFG


# ICE2THRUST_6UCSAT_LOCAL_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{REPO_ROOT_PATH}assets/robots/ice2thrust_6usat.usd",
#     ),
#     actuators={},
# )
ICE2THRUST_6UCSAT_LOCAL_CFG = ICE2THRUST_6UCSAT_CFG