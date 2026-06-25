"""Ultra GameYaw v2: motor-response alignment + anti-jitter PD gains.

Non-destructive variant of the AMP-HIM task. It reuses the *entire* aligned env
config (rewards, AMP, HIM, curriculum, domain randomization, 400 Hz / 100 Hz
timing) and changes ONLY the robot actuator block, to address the real-robot
standstill jitter observed in deployment.

Two groups of changes vs. the baseline ``ultra_constants.py`` actuators:

1. Motor friction alignment (``fs``/``fd``):
   The baseline leaves motor friction at 0, so the trained policy never sees the
   real motor's static/viscous friction. On hardware this shows up as
   high-frequency torque chatter on the low-load joints (arms worst, then
   waist/hip-yaw). The values below are ENGINEERING ESTIMATES (the deploy log
   shows ~0.3-0.5 N*m of torque on the arms at zero load that is uncorrelated
   with the PD command); CALIBRATE them against the motor datasheet / a bench
   test. Set fs=fd=0 to disable.

2. Damping (kd) reduction on the buzz-prone, low-load joints:
   kd multiplies the (noisy, 400 Hz finite-difference) joint-velocity estimate,
   so it is the main amplifier of torque chatter on joints that carry little
   gravity load. Load-bearing leg joints (hip_roll/pitch, knee) and the ankles
   are left UNCHANGED because the deploy log shows they are smooth.

IMPORTANT: kp/kd here MUST stay in sync with the deploy PD gains in
``ultra2026_rl_sdk/deploy_ultra_sim_real/config/env-ultra.yaml`` (both ``real``
and ``sim`` blocks). If you retune here, mirror the same numbers there.
"""

from __future__ import annotations

import dataclasses

from mjlab.asset_zoo.robots.ultra_game_yaw.ultra_constants import (
  ULTRA_ACT_ANKLE,
  ULTRA_ACT_HIP_KNEE_PITCH,
  ULTRA_ACT_HIP_ROLL,
  ULTRA_ACT_HIP_YAW,
  ULTRA_ACT_SHOULDER,
  ULTRA_ACT_WAIST,
  ULTRA_COLLISION,
  ULTRA_HOME_KEYFRAME,
  get_spec,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.tasks.velocity.config.ultra_game_yaw.amp_him import (
  RslRlAmpHimRunnerCfg,
  ultra_game_yaw_amp_him_runner_cfg,
)
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs import (
  ultra_game_yaw_aligned_env_cfg,
)
from mjlab.utils.spec_config import CollisionCfg

# Activation speed (rad/s) for the tanh static-friction term, shared by all groups.
_VA = 0.1

# ── v2 actuators: kd tweaks + motor friction (see module docstring) ──
# Friction (fs: Coulomb N*m, fd: viscous N*m*s/rad) — STARTING ESTIMATES.
_ACT_HIP_YAW_V2 = dataclasses.replace(
  ULTRA_ACT_HIP_YAW, damping=6.0, fs=1.5, fd=0.3, va=_VA
)  # kd 10 -> 6
_ACT_HIP_ROLL_V2 = dataclasses.replace(
  ULTRA_ACT_HIP_ROLL, fs=2.0, fd=0.4, va=_VA
)  # kd unchanged (20)
_ACT_HIP_KNEE_PITCH_V2 = dataclasses.replace(
  ULTRA_ACT_HIP_KNEE_PITCH, fs=2.0, fd=0.4, va=_VA
)  # kd unchanged (10)
_ACT_ANKLE_V2 = dataclasses.replace(
  ULTRA_ACT_ANKLE, fs=1.0, fd=0.2, va=_VA
)  # kd unchanged (6)
_ACT_WAIST_V2 = dataclasses.replace(
  ULTRA_ACT_WAIST, damping=5.0, fs=1.5, fd=0.3, va=_VA
)  # kd 8 -> 5
_ACT_SHOULDER_V2 = dataclasses.replace(
  ULTRA_ACT_SHOULDER, damping=1.0, fs=0.4, fd=0.1, va=_VA
)  # kd 3 -> 1

_ULTRA_ARTICULATION_V2 = EntityArticulationInfoCfg(
  actuators=(
    _ACT_HIP_YAW_V2,
    _ACT_HIP_ROLL_V2,
    _ACT_HIP_KNEE_PITCH_V2,
    _ACT_ANKLE_V2,
    _ACT_WAIST_V2,
    _ACT_SHOULDER_V2,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_ultra_robot_cfg_v2() -> EntityCfg:
  """Ultra GameYaw robot with motor-aligned / lowered-kd actuators."""
  collision: CollisionCfg = ULTRA_COLLISION
  return EntityCfg(
    init_state=ULTRA_HOME_KEYFRAME,
    collisions=(collision,),
    spec_fn=get_spec,
    articulation=_ULTRA_ARTICULATION_V2,
  )


def ultra_game_yaw_v2_env_cfg(play: bool = False):
  """Aligned AMP-HIM env, but with the v2 (motor-aligned, lowered-kd) robot."""
  cfg = ultra_game_yaw_aligned_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_ultra_robot_cfg_v2()}
  return cfg


def ultra_game_yaw_amp_him_v2_runner_cfg() -> RslRlAmpHimRunnerCfg:
  """Runner cfg for v2: identical to baseline AMP-HIM but separate log/project."""
  cfg = ultra_game_yaw_amp_him_runner_cfg()
  cfg.experiment_name = "ultra_game_yaw_amp_him_v2"
  cfg.wandb_project = "ultra_game_yaw_amp_him_v2"
  return cfg
