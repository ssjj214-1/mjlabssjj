"""Ultra GameYaw v13: V10 (15-DoF passive ankle-roll) + hist10 gravel terrain.

V13 takes V10 (V9 with passive ankle-roll DOFs, 15-DoF robot XML) unchanged
and swaps the flat plane for GRAVEL_CURRICULUM_TERRAINS_CFG.
"""

from __future__ import annotations

from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import GRAVEL_CURRICULUM_TERRAINS_CFG

from .amp_him import RslRlAmpHimRunnerCfg
from .env_cfgs import (
  add_terrain_relative_base_height,
  apply_rough_terrain_sim_params,
)
from .env_cfgs_v10 import (
  ultra_game_yaw_amp_him_v10_runner_cfg,
  ultra_game_yaw_v10_env_cfg,
)


def ultra_game_yaw_v13_env_cfg(play: bool = False):
  """V13: V10 with gravel curriculum terrain (same as hist10)."""
  cfg = ultra_game_yaw_v10_env_cfg(play=play)
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=GRAVEL_CURRICULUM_TERRAINS_CFG,
    max_init_terrain_level=0,
  )
  # Terrain-relative base height for reward + critic obs (hist10 parity).
  add_terrain_relative_base_height(cfg, target_height=1.18)
  # Raise contact/constraint solver capacity for rough terrain.
  apply_rough_terrain_sim_params(cfg)
  return cfg


def ultra_game_yaw_amp_him_v13_runner_cfg() -> RslRlAmpHimRunnerCfg:
  cfg = ultra_game_yaw_amp_him_v10_runner_cfg()
  cfg.experiment_name = "ultra_game_yaw_amp_him_v13"
  cfg.wandb_project = "ultra_game_yaw_amp_him_v13"
  return cfg
