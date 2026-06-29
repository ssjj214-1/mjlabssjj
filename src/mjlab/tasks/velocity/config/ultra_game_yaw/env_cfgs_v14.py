"""Ultra GameYaw v14: identical to V12 (V9 + hist10 gravel curriculum terrain).

V14 reuses V12's setup verbatim -- V9 (hist10 rewards + curriculum + HIM
history=10) on ``GRAVEL_CURRICULUM_TERRAINS_CFG`` (the faithful port of
ultra_run_lab hist10's terrain: flat 30%, random_rough 40%, mild +/- slopes),
with terrain-relative base height, rough-terrain solver capacity, and the
distance-gated ``terrain_levels_vel`` difficulty curriculum. The only difference
from V12 is the experiment / wandb name, so V14 trains as an independent slot.
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


def ultra_game_yaw_v14_env_cfg(play: bool = False):
  """V14: same as V12 -- V9 with the hist10 gravel curriculum terrain."""
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
  # Reset spawn height: random 0..0.03 m above the (terrain-relative) base
  # height instead of the base cfg's 0..0.05 m. reset_root_state_uniform adds
  # env_origins (which carries the terrain surface z), so this is an upward-only
  # margin on top of the terrain, just a touch tighter than V12.
  cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.03)
  return cfg


def ultra_game_yaw_amp_him_v14_runner_cfg() -> RslRlAmpHimRunnerCfg:
  cfg = ultra_game_yaw_amp_him_v9_runner_cfg()
  cfg.experiment_name = "ultra_game_yaw_amp_him_v14"
  cfg.wandb_project = "ultra_game_yaw_amp_him_v14"
  return cfg
