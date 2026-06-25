"""Ultra GameYaw v7: v6 + command slew (ramp) + progressive curriculum.

v7 keeps everything from v6 (retuned per-joint PD, the stronger standstill
shaping ``stand_base_motion`` + ``stand_joint_vel``, the harder joint-limit and
``torque_saturation`` penalties, v5's HoST smoothness loss, and the Isaac-aligned
heading command) and changes exactly two things, both aimed at the "low-speed
lunge / overshoot" the user observed (commanding even 3 m/s produced a hard
forward-lean lunge that only straightened up once the speed was reached):

1. Command slew (ramp). The baseline velocity command is a STEP: at resample it
   jumps to the target instantly, so a 0 -> 3 m/s command gives a large velocity
   error that the (weight-15) tracking reward closes by lunging. v7 rate-limits
   the command toward the sampled target at ``_CMD_MAX_LIN_ACCEL`` (restoring
   ultra_run_lab's ``max_lin_accel``), so the policy tracks a smoothly rising
   command. Because ``body_orientation_speed_aware`` allows forward lean
   proportional to the *commanded* speed, the lean now grows with the ramping
   command -> posture adapts to the current speed instead of lunging. This also
   closes the train/deploy mismatch (deploy already slews via ``max_lin_accel``).

   A ``_CMD_STEP_SAMPLE_FRAC`` fraction of envs still step to the target (kept for
   robustness to abrupt commands); the rest ramp. Defaults are a touch gentler
   than ultra_run_lab (accel 4.0 vs 5.0, step-frac 0.25 vs 0.5); raise
   ``_CMD_MAX_LIN_ACCEL`` to 5.0 and ``_CMD_STEP_SAMPLE_FRAC`` to 0.5 to
   reproduce ultra_run_lab exactly, or lower ``_CMD_MAX_LIN_ACCEL`` toward 1.5
   for an even gentler start.

2. Curriculum: matched to the ultra_game_yaw_run reference experiment, which
   trained over the FULL 0..15 m/s command range from step 0 (its init range
   == its curriculum cap, so the progressive term never advanced). v7 mirrors
   this by starting ``init_lin_vel_x`` at the 15 m/s cap; the
   ``commands_vel_progressive`` term still runs but is a no-op (kept for parity
   / resume tooling). See ``_PROGRESSIVE_PARAMS``.

3. Stronger foot landing-impact penalties to cut the knee landing torque. The
   knee torque at touchdown tracks the peak vertical ground-reaction force,
   which is set by the foot's speed when it lands. v7 strengthens both
   ``contact_impact_vel`` (touchdown speed; -2.0 -> -5.0) and
   ``gait_feet_force_max_neg`` (peak per-foot vertical GRF; weight -0.01 -> -0.03
   and threshold 1500 -> 1000 N) so the policy plants the foot more gently and
   spreads the landing force instead of spiking it through the knee.

It additionally restores domain-randomization / command-sampling parity that the
ultra_run_lab -> mjlab migration had dropped (audit of 2026-06-12). These all
mirror ultra_run_lab's ``UltraGameYawMotionRunEnvCfg`` and only run during
training (skipped on ``play``):

* ``pseudo_inertia`` on all bodies (mass + inertia jointly scaled ~0.8-1.2; the
  physically consistent replacement for ultra's separate mass-scale + inertia-
  scale events) plus a ±3 kg additive payload on ``waist_yaw_link`` (ultra's
  ``add_base_mass``).
* Base initial-velocity randomization on reset (ultra's ``reset_base`` velocity
  range). NOTE this perturbs the start state and can fight the "起跑连贯性" you're
  tuning — see ``_RESET_BASE_VEL`` to scale it down or disable.
* Top-speed command coverage: ``run_high_speed_sample_frac`` resamples run-style
  envs into ``[0.7*vmax, vmax]`` so the progressive cap (up to 15 m/s) is actually
  sampled near the top. This is the sampling half of the command machinery whose
  *ramp* half was added above.
* Widened ``encoder_bias`` (±0.03 vs the mjlab default ±0.015) and ``base_com``
  offset ranges to match ultra.

Restitution / full-body friction randomization is intentionally NOT ported:
MuJoCo bounce lives in ``solref``/``solimp`` (no scalar restitution like Isaac),
and on flat ground only the foot<->floor contact matters, which the inherited
``foot_friction`` already randomizes over a wider range (0.3-1.2) than ultra.

Deploy coupling is inherited from v6: the retuned kp/kd must match the deploy PD
gains, and the deploy ``heading_kp`` should be 0.5. The deploy already slews the
command (``max_lin_accel`` in LocoMode.yaml); v7 makes training consistent with
that. Train independently from v6 (separate experiment/project names).
"""

from __future__ import annotations

import os

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.ultra_game_yaw import ultra_mdp
from mjlab.tasks.velocity.config.ultra_game_yaw.amp_him import RslRlAmpHimRunnerCfg
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs_v5 import (
  ultra_game_yaw_amp_him_v5_runner_cfg,
)
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs_v6 import (
  ultra_game_yaw_v6_env_cfg,
)
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

# ── Command slew (ramp) ───────────────────────────────────────────────────
# Close to ultra_run_lab (5.0). At 4 m/s^2 a 0 -> 3 m/s command rises in ~0.75 s,
# so the lean still tracks the (ramping) command instead of lunging to a step,
# but acceleration stays brisk. Raise toward 5.0 / step-frac 0.5 to reproduce
# ultra_run_lab exactly; lower toward 1.5 for an even gentler start.
# NOTE deploy coupling: deploy LocoMode.yaml max_lin_accel is 1.5 (softer); if
# you want train/deploy to match, bump the deploy value toward 4.0 too.
_CMD_MAX_LIN_ACCEL = 4.0  # m/s^2 (and rad/s^2 for non-heading yaw)
_CMD_STEP_SAMPLE_FRAC = 0.5  # fraction that still steps; rest ramp

# ── Velocity curriculum — matched to the ultra_game_yaw_run experiment ─────
# The reference run (logs/ultra_game_yaw_run/2026-06-02_18-36-40, model_19999)
# did NOT ramp: its initial command range and curriculum cap were both
# (-0.5, 15.0), so the cap started already at the max and the "progressive"
# term was a no-op — it trained over the full 0..15 m/s range from step 0.
# We reproduce that here by starting ``init_lin_vel_x`` at the cap (15.0). The
# gate params below are kept for parity but never fire (cap can't advance past
# max). ``update_step`` is left at 0.5 and the deploy-coupled slew stays 4.0
# per request; ultra_run_lab itself used update_step=0.2 / max_lin_accel=5.0.
_PROGRESSIVE_PARAMS = {
  "command_name": "twist",
  "init_lin_vel_x": (-0.5, 15.0),
  "update_step": 0.5,
  "max_lin_vel_x": 15.0,
  "len_ratio": 0.8,
  "rwd_threshold": 0.6,
  "rwd_threshold_min": 0.4,
  "rwd_v_lo": 5.0,
  "rwd_v_hi": 15.0,
  "track_term": "track_lin_vel_x",
  "run_style_id": 2,
}

# ── Restored domain randomization (ultra_run_lab parity) ──────────────────
# pseudo_inertia jointly scales mass + inertia; alpha is a log-density, and the
# body mass scales by e^(2*alpha). alpha in (ln0.8, ln1.2)/2 ≈ (-0.11, 0.09)
# reproduces ultra's mass-scale + inertia-scale of 0.8-1.2.
_INERTIA_ALPHA_RANGE = (-0.11, 0.09)
# Additive torso payload (ultra add_base_mass on waist_yaw_link, -3..+3 kg).
_BASE_PAYLOAD_BODY = "waist_yaw_link"
_BASE_PAYLOAD_KG = (-3.0, 3.0)
# COM offset on the base link (ultra randomizes the waist COM at this magnitude).
_BASE_COM_RANGES = {0: (-0.05, 0.05), 1: (-0.03, 0.03), 2: (-0.05, 0.05)}
# Encoder zero-offset (ultra randomize_joint_pos_bias ±0.03 rad).
_ENCODER_BIAS = (-0.03, 0.03)
# Base initial-velocity randomization on reset (ultra reset_base velocity_range).
# Set to {} (or scale these down) if it hurts startup consistency.
_RESET_BASE_VEL = {
  "x": (-1.0, 1.0),
  "y": (-0.5, 0.5),
  "z": (-0.5, 0.5),
  "roll": (-0.5, 0.5),
  "pitch": (-0.5, 0.5),
  "yaw": (-0.5, 0.5),
}

# ── Top-speed command coverage (ultra run_high_speed_*) ───────────────────
_RUN_HIGH_SPEED_SAMPLE_FRAC = 0.7
_RUN_HIGH_SPEED_THRESHOLD_FRAC = 0.7
_RUN_HIGH_SPEED_ENTER = 1.05  # matches the style scheduler's run_enter

# ── Foot landing-impact penalties (reduce knee landing torque) ────────────
# The knee landing torque is driven by the peak vertical ground-reaction force
# at touchdown, which is itself set by how fast the foot is moving when it
# hits. Strengthen both levers vs the v6/aligned baseline:
#   * contact_impact_vel: foot speed at the touchdown rising edge. Penalizing it
#     harder makes the policy plant the foot more gently -> smaller impulsive
#     force -> lower knee landing torque. (aligned baseline weight -2.0)
#   * gait_feet_force_max_neg: peak per-foot vertical GRF above a threshold.
#     Lower the threshold and raise the weight so the policy is pushed to spread
#     the landing force instead of spiking it through the knee.
#     (aligned baseline weight -0.01, threshold 1500 N)
_CONTACT_IMPACT_VEL_WEIGHT = -2
_FEET_FORCE_MAX_WEIGHT = -0.01
_FEET_FORCE_MAX_THRESHOLD = 1500.0


def ultra_game_yaw_v7_env_cfg(play: bool = False):
  """v6 env (retuned PD + standstill/saturation shaping + heading) with the
  command rate-limited (slew) and the curriculum reverted to the ultra_run_lab
  progressive ramp."""
  cfg = ultra_game_yaw_v6_env_cfg(play=play)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)

  # ── 1) Command slew (ramp) ──────────────────────────────────────────
  twist_cmd.max_lin_accel = _CMD_MAX_LIN_ACCEL
  twist_cmd.cmd_step_sample_frac = _CMD_STEP_SAMPLE_FRAC

  # ── 2) Progressive curriculum (replaces v6's step table) ────────────
  # On play the aligned env clears the curriculum, so only swap when training.
  if not play:
    prog_params = dict(_PROGRESSIVE_PARAMS)
    # init_lin_vel_x defaults to the full (-0.5, 15.0) range (matches
    # ultra_game_yaw_run: trains full-range from step 0, no ramp). The override
    # below only matters if you lower init_lin_vel_x to enable an actual ramp:
    # the curriculum cap lives on the live env (env._v4_cmd_curriculum) and is
    # NOT saved in the checkpoint, so set ULTRA_CURRICULUM_INIT_VMAX=<cap> on a
    # resume run to start the cap there instead of at init_lin_vel_x.
    init_vmax = os.environ.get("ULTRA_CURRICULUM_INIT_VMAX")
    if init_vmax is not None:
      lo = prog_params["init_lin_vel_x"][0]
      prog_params["init_lin_vel_x"] = (lo, float(init_vmax))
    cfg.curriculum.pop("command_vel", None)
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
      func=ultra_mdp.commands_vel_progressive,
      params=prog_params,
    )

  # ── 3) Top-speed command coverage (ultra run_high_speed_*) ──────────
  # Applies in both train and play (harmless: with 1 play env it just biases the
  # occasional run-style resample toward the top of the active range).
  twist_cmd.run_high_speed_sample_frac = _RUN_HIGH_SPEED_SAMPLE_FRAC
  twist_cmd.run_high_speed_threshold_frac = _RUN_HIGH_SPEED_THRESHOLD_FRAC
  twist_cmd.run_high_speed_enter = _RUN_HIGH_SPEED_ENTER

  # ── 4) Restored domain randomization (training only) ────────────────
  # Widen encoder bias + base COM to ultra magnitudes (these events already
  # exist in the baseline; just retune their ranges).
  if "encoder_bias" in cfg.events:
    cfg.events["encoder_bias"].params["bias_range"] = _ENCODER_BIAS
  if "base_com" in cfg.events:
    cfg.events["base_com"].params["ranges"] = dict(_BASE_COM_RANGES)

  if not play:
    # Mass + inertia (physically consistent joint scaling) on all bodies.
    cfg.events["randomize_inertia"] = EventTermCfg(
      func=envs_mdp.dr.pseudo_inertia,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
        "alpha_range": _INERTIA_ALPHA_RANGE,
      },
    )
    # Additive torso payload (ultra add_base_mass).
    cfg.events["randomize_base_payload"] = EventTermCfg(
      func=envs_mdp.dr.body_mass,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(_BASE_PAYLOAD_BODY,)),
        "operation": "add",
        "ranges": _BASE_PAYLOAD_KG,
      },
    )
    # Base initial-velocity randomization on reset (ultra reset_base velocity).
    if _RESET_BASE_VEL and "reset_base" in cfg.events:
      cfg.events["reset_base"].params["velocity_range"] = dict(_RESET_BASE_VEL)

  # ── 5) Stronger foot landing-impact penalties (lower knee landing torque) ──
  # Applied unconditionally (reward weights only affect training; harmless on
  # play). Targets the touchdown speed + peak vertical GRF that load the knee.
  if "contact_impact_vel" in cfg.rewards:
    cfg.rewards["contact_impact_vel"].weight = _CONTACT_IMPACT_VEL_WEIGHT
  if "gait_feet_force_max_neg" in cfg.rewards:
    cfg.rewards["gait_feet_force_max_neg"].weight = _FEET_FORCE_MAX_WEIGHT
    cfg.rewards["gait_feet_force_max_neg"].params["max_force"] = (
      _FEET_FORCE_MAX_THRESHOLD
    )

  return cfg


def ultra_game_yaw_amp_him_v7_runner_cfg() -> RslRlAmpHimRunnerCfg:
  """v7 runner: identical to v5/v6 (HoST smoothness loss ON) but separate
  log/project names."""
  cfg = ultra_game_yaw_amp_him_v5_runner_cfg()
  cfg.experiment_name = "ultra_game_yaw_amp_him_v7"
  cfg.wandb_project = "ultra_game_yaw_amp_him_v7"
  return cfg
