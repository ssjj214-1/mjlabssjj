"""Ultra GameYaw v16: V9 flat training plus foot-height symmetry reward."""

from __future__ import annotations

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity.config.ultra_game_yaw import ultra_mdp
from mjlab.tasks.velocity.config.ultra_game_yaw.amp_him import RslRlAmpHimRunnerCfg
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs_v9 import (
  ultra_game_yaw_amp_him_v9_runner_cfg,
  ultra_game_yaw_v9_env_cfg,
)


def ultra_game_yaw_v16_env_cfg(play: bool = False):
  """V16 env: V9 on flat ground with left/right swing-height symmetry."""
  cfg = ultra_game_yaw_v9_env_cfg(play=play)
  cfg.rewards["feet_swing_height_symmetry"] = RewardTermCfg(
    func=ultra_mdp.feet_swing_height_symmetry,
    weight=-0.25,
    params={
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "foot_height_scan",
      "target_height": 0.1,
      "style_mask": [1, 2],
    },
  )
  return cfg


def ultra_game_yaw_amp_him_v16_runner_cfg() -> RslRlAmpHimRunnerCfg:
  """V16 runner: identical to V9, with independent experiment names."""
  cfg = ultra_game_yaw_amp_him_v9_runner_cfg()
  cfg.experiment_name = "ultra_game_yaw_amp_him_v16"
  cfg.wandb_project = "ultra_game_yaw_amp_him_v16"
  return cfg
