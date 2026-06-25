"""Ultra GameYaw v8: v7 + step (stage-table) curriculum.

v8 inherits everything from v7 — command slew (rate-limited ramp), restored
domain randomization (pseudo_inertia / payload / base_com / encoder_bias /
reset_base_vel), stronger foot landing-impact penalties, and top-speed command
coverage — and changes exactly one thing:

Curriculum: reverted from v7's "full 0..15 m/s from step 0"
(``commands_vel_progressive`` with ``init_lin_vel_x = (-0.5, 15.0)``) back to
the baseline STEP (stage-table) curriculum (``mdp.commands_vel``), the same
table v6 uses.  The motivation: the progressive no-op in v7 skips the gradual
velocity ramp that the stage table provides, which may hurt early-stage gait
quality.  With the command slew already smoothing per-step transitions in v7,
the stage table's coarse curriculum is now the missing piece.

All other v7 settings (PD, DR, smoothness loss, heading, saturation shaping,
impact penalties, run_high_speed sampling, command slew) are unchanged.

Train independently from v7 (separate experiment/project names).
"""

from __future__ import annotations

from mjlab.tasks.velocity.config.ultra_game_yaw.amp_him import RslRlAmpHimRunnerCfg
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs import (
  ultra_game_yaw_aligned_env_cfg,
)
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs_v7 import (
  ultra_game_yaw_amp_him_v7_runner_cfg,
  ultra_game_yaw_v7_env_cfg,
)
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg


def ultra_game_yaw_v8_env_cfg(play: bool = False):
  """v7 env with the curriculum reverted to the baseline STEP table."""
  cfg = ultra_game_yaw_v7_env_cfg(play=play)

  if not play:
    baseline = ultra_game_yaw_aligned_env_cfg(play=play)
    cfg.curriculum["command_vel"] = baseline.curriculum["command_vel"]

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.rel_standing_envs = 0.1

  return cfg


def ultra_game_yaw_amp_him_v8_runner_cfg() -> RslRlAmpHimRunnerCfg:
  """v8 runner: identical to v7 but with separate log/project names."""
  cfg = ultra_game_yaw_amp_him_v7_runner_cfg()
  cfg.experiment_name = "ultra_game_yaw_amp_him_v8"
  cfg.wandb_project = "ultra_game_yaw_amp_him_v8"
  return cfg
