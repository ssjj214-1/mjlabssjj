"""Ultra GameYaw v4: Isaac-faithful progressive curriculum + stand stability.

Motivation: with the joint-order permutation applied, the Isaac-Lab policy
stands rock-still in mjlab, but a policy *trained* in mjlab shuffles/drifts at
zero command. PD, robot, physics and the full reward set are already faithful
ports (an Isaac checkpoint transfers cleanly), so the standstill gap is a
training-dynamics issue. The dominant difference is the curriculum:

* baseline mjlab uses a fixed-step stage table (``commands_vel``) that widens
  the speed range on a clock regardless of mastery, and
* Isaac uses a performance-gated *progressive* ramp that only raises the speed
  cap by 0.2 m/s once run-style envs already survive long and track well.

v4 changes exactly two things on top of the baseline aligned AMP-HIM env, and
keeps the ORIGINAL expert trajectories (stand/walk/run) and the ORIGINAL
baseline PD (same as Isaac):

1. Curriculum: replace the step table with ``ultra_mdp.commands_vel_progressive``
   (the Isaac ramp: +0.2 m/s gated on run-style episode-length + tracking, with
   EMA smoothing and a speed-dependent reward gate).

2. Stand stability (belt-and-suspenders so "稳稳当当不晃动不乱走" holds even
   before the policy fully converges):
   * extend ``feet_slide`` to also cover the stand style (0) so foot shuffling
     at zero command is penalized (Isaac masks it to [1, 2]); and
   * add ``stand_base_vel`` — an L2 penalty on base horizontal speed in the
     stand style — to directly punish drifting.

Nothing else (PD gains, torque ceiling, AMP styles/coeffs, obs/critic layout)
changes vs the baseline task, so v4 stays as close to the Isaac task as
possible while fixing the standstill behaviour.
"""

from __future__ import annotations

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity.config.ultra_game_yaw import ultra_mdp
from mjlab.tasks.velocity.config.ultra_game_yaw.amp_him import (
  RslRlAmpHimRunnerCfg,
  ultra_game_yaw_amp_him_runner_cfg,
)
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs import (
  ultra_game_yaw_aligned_env_cfg,
)
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

# Stand-style base-drift penalty weight (negative). Modest so it shapes the
# stand pose without overpowering the AMP stand expert / stand_still reward.
_STAND_BASE_VEL_WEIGHT = -2.0


def ultra_game_yaw_v4_env_cfg(play: bool = False):
  """Aligned AMP-HIM env with progressive curriculum + stand-stability terms."""
  cfg = ultra_game_yaw_aligned_env_cfg(play=play)

  # ── 1) Progressive (Isaac-style) velocity curriculum ──────────────────
  # Replace the fixed-step ``command_vel`` table with the performance-gated
  # ramp. Skipped in play (the aligned env already clears curriculum on play).
  if not play:
    cfg.curriculum.pop("command_vel", None)
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
      func=ultra_mdp.commands_vel_progressive,
      params={
        "command_name": "twist",
        "init_lin_vel_x": (-0.5, 1.5),
        "update_step": 0.2,
        "max_lin_vel_x": 15.0,
        "len_ratio": 0.8,
        "rwd_threshold": 0.6,
        "rwd_threshold_min": 0.4,
        "rwd_v_lo": 5.0,
        "rwd_v_hi": 15.0,
        "track_term": "track_lin_vel_x",
        "run_style_id": 2,
      },
    )

  # ── 1b) Heading command alignment to Isaac ────────────────────────────
  # mjlab baseline drives 85% of envs by heading at stiffness 1.0, which makes
  # the policy more yaw-active (shows up as standstill yaw jiggle). Isaac uses
  # 0.6 / 0.5. Align so the yaw-command distribution matches Isaac.
  # NOTE deploy coupling: heading_control_stiffness here mirrors the deploy
  # heading-hold gain (LocoMode.yaml `heading_kp`). If a v4 policy is deployed,
  # set the deploy heading_kp to 0.5 so commanded wz stays in-distribution.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_heading_envs = 0.6
  twist_cmd.heading_control_stiffness = 0.5

  # ── 2) Stand-still stability ──────────────────────────────────────────
  # (a) Extend feet_slide to also penalize foot shuffling while standing.
  feet_slide = cfg.rewards.get("feet_slide")
  if feet_slide is not None:
    params = dict(feet_slide.params or {})
    sm = list(params.get("style_mask", []) or [])
    if 0 not in sm:
      params["style_mask"] = [0, *sm]
      feet_slide.params = params

  # (b) Penalize base horizontal drift in the stand style.
  cfg.rewards["stand_base_vel"] = RewardTermCfg(
    func=ultra_mdp.stand_base_vel_l2,
    weight=_STAND_BASE_VEL_WEIGHT,
    params={"style_mask": [0]},
  )

  return cfg


def ultra_game_yaw_amp_him_v4_runner_cfg() -> RslRlAmpHimRunnerCfg:
  """Runner cfg for v4: identical to baseline AMP-HIM but separate log/project."""
  cfg = ultra_game_yaw_amp_him_runner_cfg()
  cfg.experiment_name = "ultra_game_yaw_amp_him_v4"
  cfg.wandb_project = "ultra_game_yaw_amp_him_v4"
  return cfg
