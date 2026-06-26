"""Ultra GameYaw v12: V9 + hist10 gravel curriculum terrain.

V12 takes V9 (hist10 rewards + curriculum + HIM history=10) unchanged and swaps
the flat plane for GRAVEL_CURRICULUM_TERRAINS_CFG (matching ultra_run_lab
hist10's terrain setup: flat 30%, random_rough 40%, mild slopes).
"""

from __future__ import annotations

from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import GRAVEL_CURRICULUM_TERRAINS_CFG

from .amp_him import RslRlAmpHimRunnerCfg
from .env_cfgs import (
  add_hist10_terrain_curriculum,
  add_terrain_relative_base_height,
  apply_rough_terrain_sim_params,
)
from .env_cfgs_v9 import (
  ultra_game_yaw_amp_him_v9_runner_cfg,
  ultra_game_yaw_v9_env_cfg,
)


def ultra_game_yaw_v12_env_cfg(play: bool = False):
  """V12: V9 with gravel curriculum terrain (same as hist10)."""
  cfg = ultra_game_yaw_v9_env_cfg(play=play)
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=GRAVEL_CURRICULUM_TERRAINS_CFG,
    max_init_terrain_level=0,
  )
  # Terrain-relative base height for reward + critic obs (hist10 parity).
  add_terrain_relative_base_height(cfg, target_height=1.18)
  # Raise contact/constraint solver capacity for rough terrain.
  apply_rough_terrain_sim_params(cfg)
  # Distance-gated terrain difficulty curriculum (hist10 parity).
  add_hist10_terrain_curriculum(cfg)
  return cfg


def ultra_game_yaw_amp_him_v12_runner_cfg() -> RslRlAmpHimRunnerCfg:
  cfg = ultra_game_yaw_amp_him_v9_runner_cfg()
  cfg.experiment_name = "ultra_game_yaw_amp_him_v12"
  cfg.wandb_project = "ultra_game_yaw_amp_him_v12"
  return cfg
