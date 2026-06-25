"""Ultra GameYaw v3: soft-stiffness / high-damping PD + lowered torque ceiling.

Non-destructive variant of the AMP-HIM task, in the same spirit as ``env_cfgs_v2``.
It reuses the *entire* aligned env config and changes ONLY the robot actuator
block. The motivation is the real-robot waist structural failure: the deployed PD
turns a target-vs-actual position error into torque via ``kp``, and with no torque
saturation on hardware a large transient error spiked torque past the structural
limit. v3 attacks that directly at the controller level.

Global transform vs. the ORIGINAL design in ``ultra_constants.py`` (NOT vs. v2):

1. Stiffness (kp) halved on every joint:
   lower kp means a given position error produces half the torque, so the worst
   case torque transient at policy take-over / divergence is much smaller.

2. Damping (kd) doubled on every joint:
   compensates the softer stiffness with more velocity damping so the closed loop
   stays well-damped (less overshoot / limit-cycle tendency) despite the lower kp.

3. Torque ceiling (the safety envelope) scaled to 80% on every joint:
   the Ultra actuator caps torque at ``min(torque-speed-envelope(Y1/Y2),
   effort_limit)``. All three (``y1``, ``y2``, ``effort_limit``) are multiplied by
   0.8 so the policy is trained against a torque ceiling 20% below the motor's
   nominal limit, leaving structural margin. ``x1``/``x2`` (speed knees) are
   unchanged.

Motor friction (fs/fd/va) is KEPT from v2 — it is a sim2real fidelity feature
independent of the gain strategy.

IMPORTANT: kp/kd here MUST stay in sync with the deploy PD gains in
``ultra2026_rl_sdk/deploy_ultra_sim_real/config/env-ultra.yaml`` and
``policy/loco_mode/config/LocoMode.yaml`` once a v3 checkpoint is deployed, and
the deploy-side torque saturation ``_SAFETY_TAU_LIM`` in ``main.py`` should match
the 80% ceiling below. Do NOT change the deploy configs until v3 is actually
trained and selected (that would break the currently-deployed v2 policy).
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

# Global multipliers (see module docstring).
_KP_SCALE = 0.5  # kp halved
_KD_SCALE = 2.0  # kd doubled
_TAU_SCALE = 0.8  # torque ceiling (y1/y2/effort_limit) to 80%

# ── v3 actuators: 0.5*kp, 2*kd, 0.8*torque-ceiling + v2 motor friction ──
# Base values (from ultra_constants):
#   hip_yaw : kp500 kd10 eff250 y1150 y2250
#   hip_roll: kp700 kd20 eff657 y1600 y2657
#   hip/knee: kp700 kd10 eff657 y1600 y2657
#   ankle   : kp100 kd6  eff400 y1350 y2400
#   waist   : kp400 kd8  eff250 y1150 y2250
#   shoulder: kp50  kd3  eff140 y1100 y2140
_ACT_HIP_YAW_V3 = dataclasses.replace(
  ULTRA_ACT_HIP_YAW,
  stiffness=500.0 * _KP_SCALE,  # 250
  damping=10.0 * _KD_SCALE,  # 20
  effort_limit=250.0 * _TAU_SCALE,  # 200
  y1=150.0 * _TAU_SCALE,  # 120
  y2=250.0 * _TAU_SCALE,  # 200
  fs=1.5,
  fd=0.3,
  va=_VA,
)
_ACT_HIP_ROLL_V3 = dataclasses.replace(
  ULTRA_ACT_HIP_ROLL,
  stiffness=700.0 * _KP_SCALE,  # 350
  damping=20.0 * _KD_SCALE,  # 40
  effort_limit=657.0 * _TAU_SCALE,  # 525.6
  y1=600.0 * _TAU_SCALE,  # 480
  y2=657.0 * _TAU_SCALE,  # 525.6
  fs=2.0,
  fd=0.4,
  va=_VA,
)
_ACT_HIP_KNEE_PITCH_V3 = dataclasses.replace(
  ULTRA_ACT_HIP_KNEE_PITCH,
  stiffness=700.0 * _KP_SCALE,  # 350
  damping=10.0 * _KD_SCALE,  # 20
  effort_limit=657.0 * _TAU_SCALE,  # 525.6
  y1=600.0 * _TAU_SCALE,  # 480
  y2=657.0 * _TAU_SCALE,  # 525.6
  fs=2.0,
  fd=0.4,
  va=_VA,
)
_ACT_ANKLE_V3 = dataclasses.replace(
  ULTRA_ACT_ANKLE,
  stiffness=100.0 * _KP_SCALE,  # 50
  damping=6.0 * _KD_SCALE,  # 12
  effort_limit=400.0 * _TAU_SCALE,  # 320
  y1=350.0 * _TAU_SCALE,  # 280
  y2=400.0 * _TAU_SCALE,  # 320
  fs=1.0,
  fd=0.2,
  va=_VA,
)
_ACT_WAIST_V3 = dataclasses.replace(
  ULTRA_ACT_WAIST,
  stiffness=400.0 * _KP_SCALE,  # 200
  damping=8.0 * _KD_SCALE,  # 16
  effort_limit=250.0 * _TAU_SCALE,  # 200
  y1=150.0 * _TAU_SCALE,  # 120
  y2=250.0 * _TAU_SCALE,  # 200
  fs=1.5,
  fd=0.3,
  va=_VA,
)
_ACT_SHOULDER_V3 = dataclasses.replace(
  ULTRA_ACT_SHOULDER,
  stiffness=50.0 * _KP_SCALE,  # 25
  damping=3.0 * _KD_SCALE,  # 6
  effort_limit=140.0 * _TAU_SCALE,  # 112
  y1=100.0 * _TAU_SCALE,  # 80
  y2=140.0 * _TAU_SCALE,  # 112
  fs=0.4,
  fd=0.1,
  va=_VA,
)

_ULTRA_ARTICULATION_V3 = EntityArticulationInfoCfg(
  actuators=(
    _ACT_HIP_YAW_V3,
    _ACT_HIP_ROLL_V3,
    _ACT_HIP_KNEE_PITCH_V3,
    _ACT_ANKLE_V3,
    _ACT_WAIST_V3,
    _ACT_SHOULDER_V3,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_ultra_robot_cfg_v3() -> EntityCfg:
  """Ultra GameYaw robot with soft-stiffness / high-damping / 80%-torque actuators."""
  collision: CollisionCfg = ULTRA_COLLISION
  return EntityCfg(
    init_state=ULTRA_HOME_KEYFRAME,
    collisions=(collision,),
    spec_fn=get_spec,
    articulation=_ULTRA_ARTICULATION_V3,
  )


def ultra_game_yaw_v3_env_cfg(play: bool = False):
  """Aligned AMP-HIM env, but with the v3 (0.5*kp, 2*kd, 0.8*torque) robot."""
  cfg = ultra_game_yaw_aligned_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_ultra_robot_cfg_v3()}
  return cfg


def ultra_game_yaw_amp_him_v3_runner_cfg() -> RslRlAmpHimRunnerCfg:
  """Runner cfg for v3: identical to baseline AMP-HIM but separate log/project."""
  cfg = ultra_game_yaw_amp_him_runner_cfg()
  cfg.experiment_name = "ultra_game_yaw_amp_him_v3"
  cfg.wandb_project = "ultra_game_yaw_amp_him_v3"
  return cfg
