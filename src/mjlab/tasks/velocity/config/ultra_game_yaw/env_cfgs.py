"""Ultra GameYaw velocity environment configurations.

A direct port of the Isaac Lab `ultra_game_yaw_run` task to mjlab. Phase-1 keeps
the standard mjlab velocity manager-based pipeline (no AMP/HIM yet) and dials
in: Ultra-13DoF actuators + flat terrain + speed curriculum (up to 15 m/s) +
domain randomization + contact-aware terminations.
"""

from __future__ import annotations

import math
import os

from mjlab.asset_zoo.robots.ultra_game_yaw import (
  ULTRA_ACTION_SCALE,
  get_ultra_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from . import ultra_mdp

# Foot site/body/geom names used in several places.
_FOOT_SITES = ("left_foot", "right_foot")
_FOOT_BODIES = ("ankle_pitch_l_link", "ankle_pitch_r_link")
_FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")
# Bodies whose contact with the ground should terminate the episode.
_TERMINATE_CONTACT_BODIES = (
  "base_link",
  "waist_yaw_link",
  "knee_pitch_l_link",
  "knee_pitch_r_link",
  "shoulder_pitch_l_link",
  "shoulder_pitch_r_link",
)


def _motor_dr_enabled_default() -> bool:
  """Default on/off for motor-related domain randomization.

  Switch: set env var ``ULTRA_MOTOR_DR=0`` (or ``false``/``no``/``off``) at
  launch to disable, or pass ``enable_motor_dr=False`` to the env factory.
  """
  return os.environ.get("ULTRA_MOTOR_DR", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
  )


def ultra_game_yaw_flat_env_cfg(
  play: bool = False, enable_motor_dr: bool | None = None
) -> ManagerBasedRlEnvCfg:
  """Ultra GameYaw flat-terrain velocity env (target task `Mjlab-Velocity-Flat-UltraGameYaw`).

  ``enable_motor_dr`` toggles motor-strength + PD-gain domain randomization
  (matches Ultra's ``randomize_motor_strength`` + ``randomize_actuator_gains``).
  ``None`` resolves from the ``ULTRA_MOTOR_DR`` env var (default: enabled).
  """
  if enable_motor_dr is None:
    enable_motor_dr = _motor_dr_enabled_default()
  cfg = make_velocity_env_cfg()

  # ── Sim ────────────────────────────────────────────────────────────
  cfg.sim.mujoco.timestep = 0.0025  # 400 Hz physics, matches Isaac cfg.
  cfg.sim.mujoco.iterations = 10
  cfg.sim.mujoco.ls_iterations = 10
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None
  cfg.sim.njmax = 300
  cfg.decimation = 4  # 100 Hz control.
  cfg.episode_length_s = 20.0

  # ── Robot ──────────────────────────────────────────────────────────
  cfg.scene.entities = {"robot": get_ultra_robot_cfg()}

  # ── Terrain: switch to flat ground ────────────────────────────────
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # ── Sensors ────────────────────────────────────────────────────────
  # Drop the rough-terrain raycast scanner; keep foot-height for swing reward.
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in _FOOT_SITES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=6)

  # Foot-ground contact + self-collision contact sensors.
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(ankle_pitch_l_link|ankle_pitch_r_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
    history_length=2,  # for contact_impact_vel rising-edge detection
  )
  undesired_contact_cfg = ContactSensorCfg(
    name="undesired_ground_contact",
    primary=ContactMatch(
      mode="body",  # `body` (not `subtree`) so leg→foot contact isn't counted.
      pattern=r"^(" + "|".join(_TERMINATE_CONTACT_BODIES) + r")$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    undesired_contact_cfg,
  )

  # No height-scan in flat mode.
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)

  # ── Actions ────────────────────────────────────────────────────────
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = ULTRA_ACTION_SCALE  # {".*": 0.25}

  # ── Viewer ─────────────────────────────────────────────────────────
  cfg.viewer.body_name = "base_link"

  # ── Commands ───────────────────────────────────────────────────────
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  # Initial training range: forward-biased, modest speed; curriculum widens it.
  twist_cmd.ranges.lin_vel_x = (-0.5, 1.5)
  twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
  twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)
  twist_cmd.ranges.heading = (-math.pi / 2, math.pi / 2)
  twist_cmd.heading_command = True
  twist_cmd.rel_standing_envs = 0.1
  # Train heading-hold (wz = stiffness * heading_error) in most envs so the
  # policy learns to *hold a straight heading* at speed — this is exactly the
  # signal the deployment heading-hold feeds. Stiffness must match the deploy
  # `heading_kp` (LocoMode.yaml) so the wz commands are in-distribution.
  twist_cmd.rel_heading_envs = 0.85
  twist_cmd.heading_control_stiffness = 1.0
  twist_cmd.viz.z_offset = 1.20

  # ── Curriculum: ramp lin_vel_x toward 15 m/s like the Isaac speed curriculum.
  # mjlab uses a stage table (env.common_step_counter triggered).
  cfg.curriculum.pop("terrain_levels", None)
  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": 0, "lin_vel_x": (-0.5, 1.5), "ang_vel_z": (-0.5, 0.5)},
        {"step": 2000 * 24, "lin_vel_x": (-0.5, 3.0)},
        {"step": 4500 * 24, "lin_vel_x": (-0.5, 5.0)},
        {"step": 6000 * 24, "lin_vel_x": (-0.5, 8.0)},
        {"step": 9000 * 24, "lin_vel_x": (-0.5, 12.0)},
        {"step": 12000 * 24, "lin_vel_x": (-0.5, 15.0)},
      ],
    },
  )

  # ── Events / Domain Rand ───────────────────────────────────────────
  # Geom-friction over the foot collision geoms.
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _FOOT_GEOMS
  # COM offset on base.
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  # Reset_base: keep a small position jitter; ultra MJCF spawns ~1.20m.
  cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.05)
  # Joint-pos jitter on reset.
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)

  # ── Motor domain randomization (Ultra parity) ──────────────────────
  # Mirrors Ultra's `randomize_motor_strength` (per-joint output-torque scale)
  # and `randomize_actuator_gains` (PD stiffness/damping scale). Startup-mode,
  # so values persist across episode resets. Disabled in play and when the
  # switch is off. Skipped here for play so the policy runs with nominal motors.
  if enable_motor_dr and not play:
    cfg.events["randomize_motor_strength"] = EventTermCfg(
      func=envs_mdp.dr.motor_strength,
      mode="startup",
      params={
        "strength_range": (0.7, 1.2),
        "asset_cfg": SceneEntityCfg("robot"),
      },
    )
    cfg.events["randomize_pd_gains"] = EventTermCfg(
      func=envs_mdp.dr.pd_gains,
      mode="startup",
      params={
        "kp_range": (0.7, 1.2),
        "kd_range": (0.7, 1.2),
        "operation": "scale",
        "distribution": "uniform",
        "asset_cfg": SceneEntityCfg("robot"),
      },
    )

  # ── Rewards ────────────────────────────────────────────────────────
  # Per-joint posture stds: tighter on stand still, looser when running.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.10}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.4,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee_pitch.*": 0.45,
    r".*ankle_pitch.*": 0.30,
    r".*waist_yaw.*": 0.15,
    r".*shoulder_pitch.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.7,
    r".*hip_roll.*": 0.20,
    r".*hip_yaw.*": 0.20,
    r".*knee_pitch.*": 0.8,
    r".*ankle_pitch.*": 0.4,
    r".*waist_yaw.*": 0.20,
    r".*shoulder_pitch.*": 0.8,
  }

  # Upright on the base link.
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)

  # Body angular velocity penalty on base.
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02

  # Foot clearance/slip use site refs.
  for n in ("foot_clearance", "foot_slip"):
    cfg.rewards[n].params["asset_cfg"].site_names = _FOOT_SITES

  # Air time (encourages stepping when commanded).
  cfg.rewards["air_time"].weight = 1.0

  # Add penalty for undesired body contacts (knee/torso/shoulder hitting the ground).
  cfg.rewards["undesired_contacts"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": undesired_contact_cfg.name, "force_threshold": 1.0},
  )

  # ── Terminations: drop the rough-terrain bound check (no terrain bounds on plane).
  cfg.terminations.pop("out_of_terrain_bounds", None)
  # Add explicit termination for ground contact on terminate-bodies.
  cfg.terminations["undesired_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": undesired_contact_cfg.name, "force_threshold": 1.0},
  )

  # ── Play overrides ─────────────────────────────────────────────────
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    # Lock command to a sensible runable range during play.
    twist_cmd.ranges.lin_vel_x = (-1.0, 4.0)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)
    cfg.curriculum = {}

  return cfg


# ===========================================================================
# Ultra-aligned env (rewards + critic + symmetry-ready) — used by the
# AMP+HIM training task. Keeps everything from `ultra_game_yaw_flat_env_cfg`
# but rewires Observations / Rewards / step-Events to match the Ultra
# Isaac-Lab `ultra_game_yaw_run` task.
# ===========================================================================


def _make_ultra_aligned_observations() -> dict[str, ObservationGroupCfg]:
  """Build Ultra-aligned actor + privileged-critic observation groups.

  Actor layout (Ultra ordering, drops base_lin_vel — HIM target):
    ang_vel(3) + grav(3) + cmd(3) + jpos(13) + jvel(13) + actions(13) = 48.

  Critic = actor + privileged extras:
    + lin_vel_yaw(3) + feet_vel_z(2) + foot_force_z(2) + contact(2)
    + foot_clearance(2) + domain_rand(foot_friction(2)+kp(13)+kd(13)+motor(13)=41) = 101.
  """
  from mjlab.utils.noise import UniformNoiseCfg as Unoise

  # Observation scales mirror the Ultra `ObsScalesCfg` (noise is applied to the
  # raw signal *before* scaling, matching Ultra's pipeline). These matter
  # because the AMP+HIM runner uses `empirical_normalization=False`, so the
  # network sees these manually-scaled observations directly.
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      scale=0.5,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
      scale=(0.2, 1.0, 1.0),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      scale=0.05,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  critic_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=ultra_mdp.root_lin_vel_yaw,
      scale=0.2,
    ),
    **{
      k: ObservationTermCfg(func=v.func, params=dict(v.params), scale=v.scale)
      for k, v in actor_terms.items()
    },
    "feet_vel_z": ObservationTermCfg(
      func=ultra_mdp.feet_vel_z,
    ),
    "foot_force_z": ObservationTermCfg(
      func=ultra_mdp.foot_force_z,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "feet_contact_state": ObservationTermCfg(
      func=ultra_mdp.feet_contact_state,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_clearance_priv": ObservationTermCfg(
      func=ultra_mdp.foot_clearance_priv,
    ),
    "domain_rand_features": ObservationTermCfg(
      func=ultra_mdp.domain_rand_features,
      params={"foot_geom_names": _FOOT_GEOMS},
    ),
  }
  return {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False
    ),
  }


def _make_ultra_aligned_rewards() -> dict[str, RewardTermCfg]:
  """Full Ultra GameYaw reward set (~30 terms) ported to mjlab API."""
  R = RewardTermCfg
  S = SceneEntityCfg

  return {
    # ── Stand still (style 0) ───────────────────────────────────────
    "stand_still": R(
      func=ultra_mdp.stand_still,
      weight=10.0,
      params={
        # Ultra stand pose (mjlab joint order resolved by name; unspecified
        # joints target 0.0). Matches the Isaac `stand_still` target angles.
        "target_angles": {
          "hip_yaw_l_joint": 0.0,
          "hip_yaw_r_joint": 0.0,
          "hip_roll_l_joint": 0.0,
          "hip_roll_r_joint": 0.0,
          "hip_pitch_l_joint": -0.25,
          "hip_pitch_r_joint": -0.25,
          "knee_pitch_l_joint": 0.5,
          "knee_pitch_r_joint": 0.5,
          "ankle_pitch_l_joint": -0.25,
          "ankle_pitch_r_joint": -0.25,
          "waist_yaw_joint": 0.0,
          "shoulder_pitch_l_joint": 0.0,
          "shoulder_pitch_r_joint": 0.0,
        },
        "style_mask": [0],
      },
    ),
    # ── Tracking ────────────────────────────────────────────────────
    "track_lin_vel_x": R(
      func=ultra_mdp.track_lin_vel_x_yaw_frame_exp,
      weight=15.0,
      params={
        "std": 0.5,
        "std_speed_max": 1.0,
        "speed_walk": 1.0,
        "speed_max": 15.0,
        "style_mask": [1, 2],
      },
    ),
    "track_lin_vel_y": R(
      func=ultra_mdp.track_lin_vel_y_yaw_frame_exp,
      weight=2.0,
      params={"std": 0.3, "style_mask": [1, 2]},
    ),
    "track_lin_vel_z": R(
      func=ultra_mdp.track_lin_vel_z_yaw_frame_exp,
      weight=0.05,
      params={"std": 0.3, "style_mask": [2]},
    ),
    "track_ang_vel_z": R(
      func=ultra_mdp.track_ang_vel_z_world_exp,
      weight=8.0,
      params={
        "std": 0.3,
        # Tighten yaw-rate tolerance hard at high speed: 0.08 rad/s residual at
        # 15 m/s already curves the path off a 100 m lane, so the reward must
        # not be "satisfied" with that. (exp-shaped, so it nudges, not clips.)
        "std_speed_max": 0.08,
        "speed_walk": 1.0,
        "speed_max": 15.0,
        "style_mask": [1, 2],
      },
    ),
    "track_lin_vel_x_bonus": R(
      func=ultra_mdp.track_lin_vel_x_yaw_frame_bonus,
      weight=2.0,
      params={
        "std": 0.5,
        "std_speed_max": 1.0,
        "speed_walk": 1.0,
        "speed_max": 15.0,
        "style_mask": [2],
      },
    ),
    "track_lin_vel_x_neg": R(
      func=ultra_mdp.track_lin_vel_x_yaw_frame_neg,
      weight=-1.0,
      params={"style_mask": [2]},
    ),
    "track_ang_vel_z_bonus": R(
      func=ultra_mdp.track_ang_vel_z_world_bonus,
      weight=3.0,
      params={
        "std": 0.3,
        "std_speed_max": 0.08,
        "speed_walk": 1.0,
        "speed_max": 15.0,
        "style_mask": [2],
      },
    ),
    # High-speed straight-line yaw-drift suppression (deployment-critical:
    # robot gets only a yaw-rate cmd, no heading feedback, so residual yaw
    # rate integrates into heading drift over a 100 m sprint).
    "ang_vel_z_straight_neg": R(
      func=ultra_mdp.ang_vel_z_straight_neg,
      weight=-10.0,
      params={
        "yaw_cmd_relax_at": 0.3,
        "speed_ref": 10.0,
        "style_mask": [2],
      },
    ),
    # ── Lifecycle ───────────────────────────────────────────────────
    "alive": R(func=ultra_mdp.is_alive, weight=2.0),
    "termination_penalty": R(func=ultra_mdp.is_terminated, weight=-200.0),
    # ── Regularization ──────────────────────────────────────────────
    "energy": R(func=ultra_mdp.energy, weight=-3.0e-4),
    "joint_acc_l2": R(func=ultra_mdp.joint_acc_l2, weight=-2.5e-7),
    "action_rate_l2": R(func=ultra_mdp.action_rate_l2, weight=-0.01),
    # Restore the hist10 `arm_dof_pos_neg` reward dropped during the mjlab
    # migration. Holds the elbow-less arms at their default pose during walk/run
    # (style 0 stand is handled by `stand_still`). The AMP run experts swing
    # shoulder_pitch far behind the torso (~-1.1 rad), so without this the policy
    # copies the backward arm swing and walks with its arms behind its back.
    # Speed-aware: penalty ramps from x1 (<=8 m/s) to x3 (>=15 m/s).
    "arm_dof_pos_neg": R(
      func=ultra_mdp.arm_dof_pos_neg,
      weight=-5.0,
      params={
        "speed_threshold": 8.0,
        "speed_max": 15.0,
        "penalty_scale_max": 3.0,
        "style_mask": [1, 2],
      },
    ),
    "joint_pos_limits": R(
      func=envs_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": S("robot", joint_names=(".*",))},
    ),
    "ankle_pitch_pos_limits_neg": R(
      func=ultra_mdp.ankle_pitch_pos_limits_neg, weight=-60.0
    ),
    "hip_yaw_pos_neg": R(
      func=ultra_mdp.hip_yaw_pos_neg,
      weight=-3.0,
      params={"yaw_cmd_relax_at": 0.5, "style_mask": [1, 2]},
    ),
    "hip_yaw_action": R(
      func=ultra_mdp.hip_yaw_action,
      weight=-1.0,
      params={"yaw_cmd_relax_at": 0.5, "style_mask": [1, 2]},
    ),
    # ── Per-joint xhumanoid ─────────────────────────────────────────
    "torques_weighted_neg": R(func=ultra_mdp.torques_weighted_neg, weight=-7.0e-6),
    "wholebody_vel_weighted_neg": R(
      func=ultra_mdp.wholebody_vel_weighted_neg, weight=-0.007
    ),
    "hip_yaw_vel_cmd_neg": R(func=ultra_mdp.hip_yaw_vel_cmd_neg, weight=-0.05),
    # ── Stability / posture ────────────────────────────────────────
    "base_height_neg": R(
      func=ultra_mdp.base_height_neg,
      weight=-10.0,
      params={"target_height": 1.18, "style_mask": [0, 1, 2]},
    ),
    "lin_vel_z_l2": R(
      func=ultra_mdp.lin_vel_z_l2, weight=-1.0, params={"style_mask": [0, 1]}
    ),
    "ang_vel_xy_l2": R(func=ultra_mdp.ang_vel_xy_l2, weight=-0.5),
    "body_orientation_exp": R(
      func=ultra_mdp.body_orientation_exp,
      weight=1.0,
      params={
        "asset_cfg": S("robot", body_names=("waist_yaw_link",)),
        "style_mask": [0, 1],
      },
    ),
    "body_orientation_speed_aware": R(
      func=ultra_mdp.body_orientation_speed_aware,
      weight=3.0,
      params={
        "asset_cfg": S("robot", body_names=("waist_yaw_link",)),
        "pitch_coef": 0.025,
        "roll_coef": 0.07,
        "style_mask": [2],
      },
    ),
    # ── Contact / feet ──────────────────────────────────────────────
    "body_contact_neg": R(
      func=ultra_mdp.body_contact_neg,
      weight=-1.0,
      params={"sensor_name": "undesired_ground_contact", "threshold": 1.0},
    ),
    "feet_slide": R(
      func=ultra_mdp.feet_slide,
      weight=-1.5,
      params={
        "sensor_name": "feet_ground_contact",
        "min_contact_fz": 150.0,
        "slip_speed_threshold": 0.35,
        "only_lateral_yaw": True,
        "style_mask": [1, 2],
      },
    ),
    "contact_impact_vel": R(
      func=ultra_mdp.contact_impact_vel,
      weight=-2.0,
      params={"sensor_name": "feet_ground_contact", "style_mask": [1, 2]},
    ),
    "feet_stumble": R(
      func=ultra_mdp.feet_stumble,
      weight=-2.0,
      params={"sensor_name": "feet_ground_contact", "ratio": 5.0},
    ),
    "gait_feet_force_max_neg": R(
      func=ultra_mdp.gait_feet_force_max_neg,
      weight=-0.01,
      params={
        "sensor_name": "feet_ground_contact",
        "max_force": 1500.0,
        "style_mask": [1, 2],
      },
    ),
    # ── Spacing ─────────────────────────────────────────────────────
    "gait_feet_distance": R(
      func=ultra_mdp.gait_feet_distance,
      weight=-8.0,
      params={
        "target_y": 0.15,
        "speed_scale": 0.5,
        "speed_max": 15.0,
        "style_mask": [1, 2],
      },
    ),
    "knee_y_distance": R(
      func=ultra_mdp.knee_y_distance,
      weight=-5.0,
      params={
        "target_y": 0.15,
        "speed_scale": 0.5,
        "speed_max": 15.0,
        "style_mask": [1, 2],
      },
    ),
    "feet_collision": R(
      func=ultra_mdp.feet_collision,
      weight=-30.0,
      params={"threshold_y": 0.14, "threshold_x": 0.24, "style_mask": [1, 2]},
    ),
  }


def ultra_game_yaw_aligned_env_cfg(
  play: bool = False, enable_motor_dr: bool | None = None
) -> ManagerBasedRlEnvCfg:
  """Ultra-aligned env: full Ultra reward set + privileged critic obs +
  ``ultra_style_update`` step event so reward functions see ``env.style_ids``.

  Built on top of :func:`ultra_game_yaw_flat_env_cfg`. Used by the AMP+HIM
  training task; the symmetry data-augmentation function must be enabled
  separately on the algorithm cfg. ``enable_motor_dr`` toggles motor-strength
  + PD-gain domain randomization (see :func:`ultra_game_yaw_flat_env_cfg`).
  """
  cfg = ultra_game_yaw_flat_env_cfg(play=play, enable_motor_dr=enable_motor_dr)

  # Replace Observation manager with Ultra-aligned groups.
  cfg.observations = _make_ultra_aligned_observations()

  # Replace reward set with full Ultra port.
  cfg.rewards = _make_ultra_aligned_rewards()

  # Add the per-step style-update event so env.style_ids is current at every
  # reward computation.
  cfg.events["ultra_style_update"] = EventTermCfg(
    func=ultra_mdp.ultra_style_update,
    mode="step",
    params={"command_name": "twist"},
  )

  return cfg


# ===========================================================================
# Ultra-aligned env + stand-to-run *launch* style — used by the parallel
# AMP+HIM "accel" training task. Identical to the aligned env (same tightened
# high-speed yaw rewards) but swaps in the 4-style scheduler so the start /
# acceleration phase is shaped by the stand-to-run AMP expert. The baseline
# aligned env above is left untouched so both tasks train in parallel.
# ===========================================================================


def ultra_game_yaw_accel_env_cfg(
  play: bool = False,
  enable_motor_dr: bool | None = None,
  accel_window_s: float = 2.4,
) -> ManagerBasedRlEnvCfg:
  """Aligned env with an extra accel-launch style (id 3).

  Built on :func:`ultra_game_yaw_aligned_env_cfg` (so it inherits the tightened
  high-speed yaw rewards). Two changes:

  * The style-update step event becomes :func:`ultra_mdp.ultra_style_update_accel`,
    which tags envs commanded to run with style 3 during the first
    ``accel_window_s`` of the episode (time-aligned to the stand-to-run expert).
  * Every reward whose ``style_mask`` covers run (2) also covers accel (3), so
    launch envs keep full velocity-tracking / regularization on top of the accel
    AMP style reward.
  """
  cfg = ultra_game_yaw_aligned_env_cfg(play=play, enable_motor_dr=enable_motor_dr)

  cfg.events["ultra_style_update"] = EventTermCfg(
    func=ultra_mdp.ultra_style_update_accel,
    mode="step",
    params={"command_name": "twist", "accel_window_s": accel_window_s},
  )

  # Extend run-style reward masks to also include the accel style (3).
  for term in cfg.rewards.values():
    sm = (term.params or {}).get("style_mask")
    if sm is not None and 2 in sm and 3 not in sm:
      term.params = dict(term.params)
      term.params["style_mask"] = list(sm) + [3]

  return cfg
