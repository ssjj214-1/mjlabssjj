"""Ultra GameYaw v11: tn-yaw-fix baseline with ultra_run_lab new_him DR parity
+ V9 retuned PD.

V11 takes the tn-yaw-fix training setup (AMP+HIM, aligned rewards, step-based
curriculum) and patches the domain-randomization / command-sampling / init-state
gaps identified in the new_him migration audit, plus retunes the PD gains to
match V9 (lower kp, higher kd — the "damping-ratio" tune).

Changes vs tn-yaw-fix (ultra_game_yaw_aligned_env_cfg):

1. DR additions:
   - body mass: waist_yaw_link add (±3 kg), others scale (0.8–1.2× via pseudo_inertia)
   - pseudo_inertia: all bodies 0.8–1.2× (mass + inertia together, physics-consistent)
   - full-body pair friction: startup-mode friction DR on all robot bodies
   - reset base velocity: all 6 DoF velocity randomized on reset
2. DR alignment:
   - COM: waist_yaw_link (was base_link), ranges widened to ultra_run_lab values
   - encoder bias: reset-mode ±0.03 rad (was startup-mode ±0.015 rad)
3. Commands:
   - resampling_time_range: (10, 10) (was (3, 8))
4. Init state:
   - base pos z: 1.23 (was 1.20)
   - action delay max: 3 steps (was 4)
5. Events:
   - push robot: interval 5–10 s (was 1–3 s)
   - reset joints: by_scale (0.5–1.5×) (was by_offset ±0.1 rad)
6. PD: retuned to V9 values (lower kp, higher kd per-joint table).

Everything else (rewards, HIM, AMP, style, noise, obs scales, decimation) is
inherited unchanged from the base aligned env cfg.
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
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr as mdp_dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.utils.spec_config import CollisionCfg

from .amp_him import RslRlAmpHimRunnerCfg
from .env_cfgs import ultra_game_yaw_aligned_env_cfg

# ── V9 retuned PD table (same as env_cfgs_v6.py) ──────────────────────────
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
  """Return a copy of ``act`` with kp/kd from the v9 tables, everything else
  (effort/x1/x2/y1/y2/delay/friction) unchanged."""
  names = act.target_names_expr
  kps = {JOINT_STIFFNESS[n] for n in names}
  kds = {JOINT_DAMPING[n] for n in names}
  assert len(kps) == 1, f"Inconsistent kp across actuator group {names}: {kps}"
  assert len(kds) == 1, f"Inconsistent kd across actuator group {names}: {kds}"
  return dataclasses.replace(act, stiffness=kps.pop(), damping=kds.pop())


_ULTRA_ARTICULATION_V9 = EntityArticulationInfoCfg(
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


def _get_ultra_robot_cfg_v9() -> EntityCfg:
  """Ultra GameYaw robot with V9 retuned PD gains."""
  return EntityCfg(
    init_state=ULTRA_HOME_KEYFRAME,
    collisions=(ULTRA_COLLISION,),
    spec_fn=get_spec,
    articulation=_ULTRA_ARTICULATION_V9,
  )


def ultra_game_yaw_v11_env_cfg(play: bool = False):
  """V11: tn-yaw-fix + new_him DR/command alignment."""

  cfg = ultra_game_yaw_aligned_env_cfg(play=play)

  # ── PD: replace robot with V9 retuned PD ───────────────────────────
  cfg.scene.entities["robot"] = _get_ultra_robot_cfg_v9()

  # ── Init state ──────────────────────────────────────────────────────
  # Base spawn z: 1.20 → 1.23 (match ultra_run_lab)
  cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, 1.23)

  # ── Action delay: max 4 → 3 ─────────────────────────────────────────
  for act_cfg in cfg.scene.entities["robot"].articulation.actuators:
    if hasattr(act_cfg, "delay_max_lag") and act_cfg.delay_max_lag > 3:
      act_cfg.delay_max_lag = 3

  # ── Commands: resampling (3,8) → (10,10) ────────────────────────────
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.resampling_time_range = (10.0, 10.0)
  twist_cmd.viz.z_offset = 1.23

  # ── Reset joints: by_scale instead of by_offset ─────────────────────
  cfg.events["reset_robot_joints"] = EventTermCfg(
    func=envs_mdp.reset_joints_by_scale,
    mode="reset",
    params={
      "position_range": (0.5, 1.5),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )

  # ── Reset base: add velocity randomization ──────────────────────────
  cfg.events["reset_base"].params["velocity_range"] = {
    "x": (-1.0, 1.0),
    "y": (-0.5, 0.5),
    "z": (-0.5, 0.5),
    "roll": (-0.5, 0.5),
    "pitch": (-0.5, 0.5),
    "yaw": (-0.5, 0.5),
  }

  # ── Push robot: interval 5–10 s (was 1–3 s).  Play mode pops this. ─
  if "push_robot" in cfg.events:
    cfg.events["push_robot"].interval_range_s = (5.0, 10.0)

  # ── COM: on waist_yaw_link with ultra_run_lab ranges ────────────────
  cfg.events["base_com"].params["asset_cfg"].body_names = ("waist_yaw_link",)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.05, 0.05),
    1: (-0.03, 0.03),
    2: (-0.05, 0.05),
  }

  # ── Encoder bias: reset-mode, ±0.03 rad (was startup, ±0.015) ──────
  cfg.events["encoder_bias"].mode = "reset"
  cfg.events["encoder_bias"].params["bias_range"] = (-0.03, 0.03)

  # ── Body mass: waist_yaw_link add ±3 kg ─────────────────────────────
  cfg.events["randomize_base_mass"] = EventTermCfg(
    func=mdp_dr.body_mass,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("waist_yaw_link",)),
      "ranges": (-3.0, 3.0),
      "operation": "add",
      "distribution": "uniform",
    },
  )

  # ── Inertia + body mass (others): pseudo-inertia 0.8–1.2× ───────────
  # Physics-consistent mass + inertia scaling for all bodies.
  cfg.events["randomize_body_inertia"] = EventTermCfg(
    func=mdp_dr.pseudo_inertia,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      "alpha_range": (0.8, 1.2),
      "distribution": "uniform",
    },
  )

  # ── Full-body pair friction (startup, all bodies) ───────────────────
  cfg.events["physics_material"] = EventTermCfg(
    func=mdp_dr.pair_friction,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      "ranges": (0.4, 1.0),
      "operation": "abs",
      "isotropic": True,
      "distribution": "uniform",
    },
  )

  return cfg


def ultra_game_yaw_amp_him_v11_runner_cfg() -> RslRlAmpHimRunnerCfg:
  """V11 runner: same as base AMP-HIM (tn-yaw-fix) with v11 experiment name."""

  from .amp_him import ultra_game_yaw_amp_him_runner_cfg

  cfg = ultra_game_yaw_amp_him_runner_cfg()
  cfg.experiment_name = "ultra_game_yaw_amp_him_v11"
  cfg.wandb_project = "ultra_game_yaw_amp_him_v11"
  cfg.run_name = "newhim补齐"
  return cfg
