"""Ultra GameYaw v15: V9 flat training with the 14 m/s run AMP clip removed."""

from __future__ import annotations

from pathlib import Path

from mjlab.tasks.velocity.config.ultra_game_yaw.amp_him import RslRlAmpHimRunnerCfg
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs_v9 import (
  ultra_game_yaw_amp_him_v9_runner_cfg,
  ultra_game_yaw_v9_env_cfg,
)

_RUN_14MPS_MOTION = "run_ultrayaw_14mps.txt"


def ultra_game_yaw_v15_env_cfg(play: bool = False):
  """V15 env: identical to V9 on flat ground."""
  return ultra_game_yaw_v9_env_cfg(play=play)


def ultra_game_yaw_amp_him_v15_runner_cfg() -> RslRlAmpHimRunnerCfg:
  """V15 runner: V9, but remove the 14 m/s run motion from style_2."""
  cfg = ultra_game_yaw_amp_him_v9_runner_cfg()
  cfg.amp_motion_files_dict = {
    key: list(value) for key, value in cfg.amp_motion_files_dict.items()
  }
  cfg.amp_motion_files_dict["style_2"] = [
    path
    for path in cfg.amp_motion_files_dict["style_2"]
    if Path(path).name != _RUN_14MPS_MOTION
  ]
  cfg.experiment_name = "ultra_game_yaw_amp_him_v15"
  cfg.wandb_project = "ultra_game_yaw_amp_him_v15"
  return cfg
