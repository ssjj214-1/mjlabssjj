"""Ultra GameYaw v9plus: V9 plus fall/get-up recovery training.

V9plus keeps V9's stand/walk/run AMP+HIM setup intact and adds the recovery
pipeline from AMP_mjlab: a subset of envs delay fall terminations, then resets
from fall/get-up motion frames so one policy sees locomotion and recovery states.
"""

from __future__ import annotations

from pathlib import Path

from mjlab import MJLAB_SRC_PATH
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from . import recovery_mdp
from .amp_him import RslRlAmpHimRunnerCfg
from .env_cfgs_v9 import (
  ultra_game_yaw_amp_him_v9_runner_cfg,
  ultra_game_yaw_v9_env_cfg,
)

_RECOVERY_MOTION_DIR: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "ultra_game_yaw" / "recovery_motions"
)
_AMP_MOTION_DIR: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "ultra_game_yaw" / "amp_motions"
)
_RECOVERY_AMP_MOTION_FILES: tuple[str, ...] = (
  str(_AMP_MOTION_DIR / "recovery_Take_040_Skeleton0.txt"),
  str(_AMP_MOTION_DIR / "recovery_fallAndGetUp1_subject1_best_getup.txt"),
)
_DELAY_RESET_ENV_RATIO = 0.4
_MAX_DELAY_STEPS = 250
_G1_STANDING_HEIGHT_M = 1.33
_ULTRA_STANDING_HEIGHT_M = 1.80
_G1_RECOVERY_HEIGHT_STD = 0.3
_ULTRA_RECOVERY_HEIGHT_STD = (
  _G1_RECOVERY_HEIGHT_STD * _ULTRA_STANDING_HEIGHT_M / _G1_STANDING_HEIGHT_M
)


def ultra_game_yaw_v9plus_env_cfg(play: bool = False):
  """V9 env with delayed fall termination and recovery-motion reset."""

  cfg = ultra_game_yaw_v9_env_cfg(play=play)
  delay_ratio = 1.0 if play else _DELAY_RESET_ENV_RATIO
  recovery_dir = str(_RECOVERY_MOTION_DIR)

  cfg.events["init_recovery_motion_loader"] = EventTermCfg(
    func=recovery_mdp.init_recovery_motion_loader,
    mode="startup",
    params={
      "recovery_dir": recovery_dir,
      "delay_reset_env_ratio": delay_ratio,
      "max_delay_steps": _MAX_DELAY_STEPS,
    },
  )
  cfg.events["reset_from_recovery_motion"] = EventTermCfg(
    func=recovery_mdp.reset_from_recovery_motion,
    mode="reset",
    params={
      "recovery_dir": recovery_dir,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.events["ultra_style_update"] = EventTermCfg(
    func=recovery_mdp.ultra_recovery_style_update,
    mode="step",
    params={"command_name": "twist", "recovery_style_id": 3},
  )

  cfg.rewards["recovery_root_height"] = RewardTermCfg(
    func=recovery_mdp.recovery_root_height_exp,
    weight=1.0,
    params={"std": _ULTRA_RECOVERY_HEIGHT_STD, "delay_env_rew_ratio": 3.5},
  )
  cfg.rewards["recovery_body_orientation"] = RewardTermCfg(
    func=recovery_mdp.recovery_body_orientation_exp,
    weight=1.0,
    params={
      "delay_env_rew_ratio": 1.0,
      "asset_cfg": SceneEntityCfg("robot", body_names=("waist_yaw_link",)),
    },
  )
  cfg.metrics["recovery_active_ratio"] = MetricsTermCfg(
    func=recovery_mdp.recovery_active_ratio,
  )
  cfg.metrics["recovery_delay_progress"] = MetricsTermCfg(
    func=recovery_mdp.recovery_delay_progress,
  )
  cfg.metrics["recovery_attempt_rate"] = MetricsTermCfg(
    func=recovery_mdp.RecoveryTransitionMetric,
    params={"mode": "attempt"},
  )
  cfg.metrics["recovery_success_rate"] = MetricsTermCfg(
    func=recovery_mdp.RecoveryTransitionMetric,
    params={"mode": "success"},
  )
  cfg.metrics["recovery_failure_rate"] = MetricsTermCfg(
    func=recovery_mdp.RecoveryTransitionMetric,
    params={"mode": "failure"},
  )
  cfg.metrics["recovery_success_episode"] = MetricsTermCfg(
    func=recovery_mdp.RecoveryEpisodeOutcomeMetric,
    params={"mode": "success"},
    reduce="last",
  )
  cfg.metrics["recovery_failure_episode"] = MetricsTermCfg(
    func=recovery_mdp.RecoveryEpisodeOutcomeMetric,
    params={"mode": "failure"},
    reduce="last",
  )

  # During active recovery, body contact and trunk angular motion are part of
  # the get-up maneuver. Keep these locomotion penalties on stand/walk/run only.
  for term_name in ("ang_vel_xy_l2", "body_contact_neg", "feet_stumble"):
    term = cfg.rewards[term_name]
    params = dict(term.params or {})
    params["style_mask"] = [0, 1, 2]
    term.params = params
  return cfg


def ultra_game_yaw_amp_him_v9plus_runner_cfg() -> RslRlAmpHimRunnerCfg:
  """Runner cfg for V9plus: same policy/AMP/HIM stack as V9."""

  cfg = ultra_game_yaw_amp_him_v9_runner_cfg()
  cfg.num_styles = 4
  cfg.amp_motion_files_dict = {
    **cfg.amp_motion_files_dict,
    "style_3": list(_RECOVERY_AMP_MOTION_FILES),
  }
  cfg.amp_reward_coef_dict = {
    **cfg.amp_reward_coef_dict,
    "style_3": 1.0,
  }
  cfg.experiment_name = "ultra_game_yaw_amp_him_v9plus"
  cfg.wandb_project = "ultra_game_yaw_amp_him_v9plus"
  return cfg
