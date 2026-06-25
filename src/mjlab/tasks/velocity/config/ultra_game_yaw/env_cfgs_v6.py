"""Ultra GameYaw v6: v5 (HoST smooth loss) + retuned PD + step curriculum.

v6 takes the v5 line (v4 standstill fixes + HoST smoothness loss) and changes
exactly two things the user asked for:

1. PD gains: retuned per-joint stiffness/damping (the "damping-ratio" tune
   below). Everything else on the actuators — effort limit, torque-speed
   envelope (x1/x2/y1/y2), command delay, friction — is kept from the baseline
   Ultra actuators (``ultra_constants``). This is NOT v3's global 0.5*kp / 2*kd
   transform; it's an explicit per-joint table.

2. Curriculum: reverted from v4/v5's performance-gated PROGRESSIVE ramp back to
   the baseline STEP (stage-table) curriculum (``mdp.commands_vel``).

3. Stronger standstill shaping (v5 still drifts AND sways at zero command):
   * the stand-still env fraction is left at the baseline 0.1 (10%) for EVERY
     stage -- the per-stage staging / pure-standing warmup was tried and dropped
     (reward stalled, no standstill benefit); and
   * v4's stand-style linear-drift penalty ``stand_base_vel`` is replaced by
     ``stand_base_motion`` -- a command-gated penalty (fires when the command is
     ~0) on FULL base motion (linear xy + angular), so it also kills the trunk
     sway/rocking that the linear-only term leaves untouched; and
   * ``stand_joint_vel`` -- a command-gated penalty on ALL joint velocities
     (ankle_pitch included), so the limbs (ankles/knees/elbows) also stop
     jittering in place once the base is still. Both fire only at ~zero command,
     so they never fight stepping. The weights here are stronger than the first
     v6 run (base-motion -2->-5 with 2x angular, joint-vel -0.02->-0.1), which
     still rocked the trunk front-back and let the ankle pitch oscillate.

4. Harder joint-limit + torque-saturation penalties (5 m/s hits saturation):
   * ``joint_pos_limits`` -10 -> -30 (all joints) and ``ankle_pitch_pos_limits``
     -60 -> -150; plus a NEW ``torque_saturation`` penalty (``torque_saturation_neg``)
     that squared-penalizes applied torque past 85% of the actuator's true
     velocity-dependent ceiling. Root cause of the saturation is the high v6 kd
     (damping torque kd*qd grows with joint speed), so this fights the symptom;
     if it caps top speed too hard, lower kd instead.

Everything else is inherited from v4/v5:
* v4's ``feet_slide`` on the stand style and the Isaac-aligned heading command
  (0.6 / 0.5) -- but v4's ``stand_base_vel`` is swapped for ``stand_base_motion``
  (see point 3);
* v5's HoST smoothness loss in the PPO update (policy + value smoothness).

Note: vs v5 this changes BOTH the PD and the curriculum at once, so a v6-vs-v5
comparison is not a clean single-variable ablation — it's a deliberate combined
config. The HoST smoothness loss is kept ON (same coefficients as v5); flip it
off by zeroing ``smoothness_lower_bound`` if you want a PD/curriculum-only run.

Deploy coupling: the retuned kp/kd below MUST match the deploy PD gains
(``ultra2026_rl_sdk`` env/LocoMode configs) once a v6 checkpoint is deployed.
Heading coupling is inherited from v4 (set deploy ``heading_kp`` to 0.5). Do NOT
touch the deploy configs until v6 is trained and selected.
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
  get_v10_spec,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity.config.ultra_game_yaw import ultra_mdp
from mjlab.tasks.velocity.config.ultra_game_yaw.amp_him import RslRlAmpHimRunnerCfg
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs import (
  ultra_game_yaw_aligned_env_cfg,
)
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs_v4 import (
  ultra_game_yaw_v4_env_cfg,
)
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs_v5 import (
  ultra_game_yaw_amp_him_v5_runner_cfg,
)
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.utils.spec_config import CollisionCfg

# Standstill shaping (replaces v4's stand-style linear-drift penalty with a
# command-gated penalty on FULL base motion + an all-joint velocity penalty).
# The stand-still env fraction is left at the baseline 0.1 (10%) for EVERY stage
# -- no staging and no pure-standing warmup (those didn't help; reward stalled).
# Stronger standstill shaping (the -2.0 / -0.02 run still rocked the trunk
# front-back and let the ankle pitch oscillate). Angular base motion (the
# front-back rock IS base-pitch rate) is weighted 2x the linear term, and the
# all-joint velocity penalty (ankle_pitch included) is 5x harder.
_STAND_BASE_MOTION_WEIGHT = -5.0  # weight of the command-gated base-motion penalty
_STAND_BASE_MOTION_ANG_SCALE = 2.0  # angular-vel term weight relative to linear
# Command-gated penalty on ALL joint velocities (ankle_pitch included -- this
# sums over every joint). base-motion stills the trunk, but the limbs (ankles/
# knees/elbows) still jitter in place. At zero command nothing should move.
_STAND_JOINT_VEL_WEIGHT = -0.1

# ── Joint-limit & torque-saturation penalties (user: "越狠越好") ───────────
# At 5 m/s the v6 high-kd PD demands torque past the speed-dependent envelope
# (kd*qd is large at high joint speed) -> saturation. Crank the joint-limit
# penalties AND add a dedicated saturation penalty that fires when applied
# torque pushes past ``margin`` * the actuator's true (velocity-dependent) ceiling.
_JOINT_POS_LIMITS_WEIGHT = -30.0  # baseline -10.0 (all joints)
_ANKLE_PITCH_LIMITS_WEIGHT = -150.0  # baseline -60.0
_TORQUE_SATURATION_WEIGHT = -6.0e-4  # squared torque-overshoot beyond margin*ceiling
_TORQUE_SATURATION_MARGIN = 0.85  # start penalizing at 85% of the ceiling

# Retuned per-joint PD gains (user-provided). hip_pitch and knee_pitch share the
# same actuator group, so they MUST share kp/kd here (both 600 / 40 — OK).
JOINT_STIFFNESS: dict[str, float] = {
  "hip_yaw_l_joint": 400.0,
  "hip_yaw_r_joint": 400.0,
  "hip_roll_l_joint": 600.0,
  "hip_roll_r_joint": 600.0,
  "hip_pitch_l_joint": 600.0,
  "hip_pitch_r_joint": 600.0,
  "knee_pitch_l_joint": 600.0,
  "knee_pitch_r_joint": 600.0,
  "ankle_pitch_l_joint": 100.0,
  "ankle_pitch_r_joint": 100.0,
  "waist_yaw_joint": 200.0,
  "shoulder_pitch_l_joint": 40.0,
  "shoulder_pitch_r_joint": 40.0,
}

JOINT_DAMPING: dict[str, float] = {
  "hip_yaw_l_joint": 20.0,
  "hip_yaw_r_joint": 20.0,
  "hip_roll_l_joint": 40.0,
  "hip_roll_r_joint": 40.0,
  "hip_pitch_l_joint": 40.0,
  "hip_pitch_r_joint": 40.0,
  "knee_pitch_l_joint": 40.0,
  "knee_pitch_r_joint": 40.0,
  "ankle_pitch_l_joint": 10.0,
  "ankle_pitch_r_joint": 10.0,
  "waist_yaw_joint": 10.0,
  "shoulder_pitch_l_joint": 4.0,
  "shoulder_pitch_r_joint": 4.0,
}


def _retune(act):
  """Return a copy of ``act`` with kp/kd from the v6 tables, everything else
  (effort/x1/x2/y1/y2/delay/friction) unchanged. All joints in the group must
  map to a single kp/kd; this asserts they agree."""
  names = act.target_names_expr
  kps = {JOINT_STIFFNESS[n] for n in names}
  kds = {JOINT_DAMPING[n] for n in names}
  assert len(kps) == 1, f"Inconsistent kp across actuator group {names}: {kps}"
  assert len(kds) == 1, f"Inconsistent kd across actuator group {names}: {kds}"
  return dataclasses.replace(act, stiffness=kps.pop(), damping=kds.pop())


_ULTRA_ARTICULATION_V6 = EntityArticulationInfoCfg(
  actuators=(
    _retune(ULTRA_ACT_HIP_YAW),
    _retune(ULTRA_ACT_HIP_ROLL),
    _retune(ULTRA_ACT_HIP_KNEE_PITCH),
    _retune(ULTRA_ACT_ANKLE),
    _retune(ULTRA_ACT_WAIST),
    _retune(ULTRA_ACT_SHOULDER),
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_ultra_robot_cfg_v6() -> EntityCfg:
  """Ultra GameYaw robot with the v6 retuned PD gains (baseline everything else)."""
  collision: CollisionCfg = ULTRA_COLLISION
  return EntityCfg(
    init_state=ULTRA_HOME_KEYFRAME,
    collisions=(collision,),
    spec_fn=get_spec,
    articulation=_ULTRA_ARTICULATION_V6,
  )


def get_ultra_v10_robot_cfg_v6() -> EntityCfg:
  """Ultra GameYaw V10 robot with the v6 retuned PD gains."""
  collision: CollisionCfg = ULTRA_COLLISION
  return EntityCfg(
    init_state=ULTRA_HOME_KEYFRAME,
    collisions=(collision,),
    spec_fn=get_v10_spec,
    articulation=_ULTRA_ARTICULATION_V6,
  )


def ultra_game_yaw_v6_env_cfg(play: bool = False):
  """v6 env: v4 env (stand fixes + heading) with the curriculum reverted to the
  baseline STEP table and the robot PD swapped for the v6 tune."""
  cfg = ultra_game_yaw_v4_env_cfg(play=play)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)

  # ── Curriculum: progressive (v4/v5) -> baseline STEP table. ──────────
  # No per-stage standstill staging and no pure-standing warmup: the stand-still
  # env fraction stays at the baseline 0.1 (10%) the whole run.
  # On play the aligned env clears the curriculum, so only swap when training.
  if not play:
    baseline = ultra_game_yaw_aligned_env_cfg(play=play)
    cfg.curriculum["command_vel"] = baseline.curriculum["command_vel"]
    twist_cmd.rel_standing_envs = 0.1

  # ── Command-gated base-sway penalty (replaces v4's linear-only one) ───
  # v4/v5 only penalize horizontal drift in the stand AMP style; v5 still rocks
  # the trunk at zero command. Penalize FULL base motion (linear xy + angular)
  # whenever the command is ~0, so the floating base actually stays still.
  cfg.rewards.pop("stand_base_vel", None)
  cfg.rewards["stand_base_motion"] = RewardTermCfg(
    func=ultra_mdp.stand_base_motion_l2,
    weight=_STAND_BASE_MOTION_WEIGHT,
    params={"ang_vel_scale": _STAND_BASE_MOTION_ANG_SCALE},
  )
  # Base stays put but knees/elbows still jitter in place -> also still every
  # joint at zero command. Gated on ~zero command, so it never fights stepping.
  # NOTE: this sums asset.data.joint_vel over ALL joints, so ankle_pitch is
  # already included -- the earlier sway was just the -0.02 weight being too weak.
  cfg.rewards["stand_joint_vel"] = RewardTermCfg(
    func=ultra_mdp.stand_joint_vel_l2,
    weight=_STAND_JOINT_VEL_WEIGHT,
  )

  # ── Harder joint-limit + torque-saturation penalties (5 m/s saturates) ─
  # Crank the existing limit penalties...
  cfg.rewards["joint_pos_limits"].weight = _JOINT_POS_LIMITS_WEIGHT
  cfg.rewards["ankle_pitch_pos_limits_neg"].weight = _ANKLE_PITCH_LIMITS_WEIGHT
  # ...and add a dedicated saturation penalty (squared overshoot past margin *
  # the velocity-dependent torque ceiling). This is the term that actually fights
  # the 5 m/s saturation; root cause is the high v6 kd (kd*qd grows with speed),
  # so if this caps top speed too hard, lower kd rather than fighting it here.
  cfg.rewards["torque_saturation"] = RewardTermCfg(
    func=ultra_mdp.torque_saturation_neg,
    weight=_TORQUE_SATURATION_WEIGHT,
    params={"margin": _TORQUE_SATURATION_MARGIN},
  )

  # ── Retuned PD gains ─────────────────────────────────────────────────
  cfg.scene.entities = {"robot": get_ultra_robot_cfg_v6()}

  return cfg


def ultra_game_yaw_amp_him_v6_runner_cfg() -> RslRlAmpHimRunnerCfg:
  """v6 runner: identical to v5 (HoST smoothness loss ON) but separate
  log/project names."""
  cfg = ultra_game_yaw_amp_him_v5_runner_cfg()
  cfg.experiment_name = "ultra_game_yaw_amp_him_v6"
  cfg.wandb_project = "ultra_game_yaw_amp_him_v6"
  return cfg
