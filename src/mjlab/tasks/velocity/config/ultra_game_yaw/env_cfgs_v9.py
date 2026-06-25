"""Ultra GameYaw v9: hist10 reward set + hist10 curriculum, on the AMP+HIM stack.

v9 answers a specific request: take the ultra_run_lab ``ultra_game_yaw_hist10``
branch and reproduce its **reward set** and its **velocity curriculum** as
faithfully as mjlab allows, while *keeping* mjlab's HIM (GRU velocity-estimation)
actor — hist10 itself is a flat MLP with no HIM. Everything else (HIM
observations, per-joint PD, action_scale, sim dt/decimation, domain
randomization, command slew) is inherited unchanged from v7 so the other
versions' configs are untouched.

What changes vs v7:

1. Rewards: ``cfg.rewards`` is fully replaced by :func:`_make_hist10_rewards`, a
   1:1 port of ``UltraGameYawHist10RewardCfg`` (term names, weights, params and
   style masks copied verbatim). Several hist10 terms that the original mjlab
   migration had dropped are restored in ``ultra_mdp`` (EMA average-velocity
   tracking, 2nd-order ``action_smoothness``, ``action_l2``, ``arm_dof_pos_neg``,
   ``waist_yaw_dof_pos_neg``, ``hip_roll_action`` / ``hip_*_deviation_l1``,
   ``dof_vel_pen_l2s``, biped ``feet_air_time``). A zero-weight
   ``hist10_state`` term runs first each step to refresh the EMA velocity and
   action history the other terms read.

2. Curriculum: hist10's performance-gated cap ramp **with retreat**. It starts
   the cap at 4 m/s and advances/retreats by 0.5 m/s toward 15 m/s every 1000
   steps based on run-style episode length + ``track_lin_vel_x_exp``
   (advance ratio 0.75, retreat ratio 0.65, reward gate 0.50->0.45 over
   6->15 m/s, retreat offset 0.1). Implemented by enabling the optional
   ``retreat_*`` / ``check_interval_steps`` args of
   :func:`ultra_mdp.commands_vel_progressive` (off for v4-v8, so they're
   unaffected).

3. Command sampling matched to hist10 where mjlab exposes the knob:
   ``max_lin_accel=3.0``, ``cmd_step_sample_frac=0.3``, stand/walk/run bucket
   sampling (``rel_walk_envs=0.20``, etc.), and heading hold
   (``rel_heading_envs=0.6``, ``heading_control_stiffness=0.9``).

4. Observation history: hist10 uses 10-frame actor + 10-frame critic stacks.
   V9 sets ``actor_obs_history_length=10`` and ``critic_obs_history_length=10``
   in the runner (HIM is kept; the critic MLP sees the full 10-frame privileged
   stack like hist10). Older AMP+HIM versions keep ``critic_obs_history_length=1``.

Train independently from the other versions (separate experiment/project names).
"""

from __future__ import annotations

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.ultra_game_yaw import ultra_mdp
from mjlab.tasks.velocity.config.ultra_game_yaw.amp_him import RslRlAmpHimRunnerCfg
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs_v7 import (
  ultra_game_yaw_amp_him_v7_runner_cfg,
  ultra_game_yaw_v7_env_cfg,
)
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

# Stand pose target (mjlab joint order resolved by name); matches hist10's
# ``stand_still`` target list and the existing aligned-env target.
_STAND_TARGET_ANGLES: dict[str, float] = {
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
}

# hist10 velocity curriculum (CommandsCfg) mapped onto commands_vel_progressive.
_HIST10_CURRICULUM_PARAMS = {
  "command_name": "twist",
  "init_lin_vel_x": (-0.5, 4.0),  # curriculum_min_lin_vel_x
  "max_lin_vel_x": 15.0,  # curriculum_max_lin_vel_x[1]
  "update_step": 0.5,  # curriculum_update_step
  "len_ratio": 0.75,  # curriculum_min_mean_episode_ratio
  "rwd_threshold": 0.50,  # curriculum_rwd_threshold
  "rwd_threshold_min": 0.45,  # curriculum_rwd_threshold_min
  "rwd_v_lo": 6.0,  # curriculum_rwd_threshold_v_lo
  "rwd_v_hi": 15.0,  # curriculum_rwd_threshold_v_hi
  "track_term": "track_lin_vel_x_exp",  # gate metric (matches reward key)
  "run_style_id": 2,
  "retreat_len_ratio": 0.65,  # curriculum_retreat_mean_episode_ratio
  "retreat_rwd_offset": 0.1,  # curriculum_retreat_rwd_offset
  "retreat_floor_x": 4.0,  # floor == curriculum_min_lin_vel_x[1]
  "check_interval_steps": 1000,  # curriculum_check_interval_steps
}

# hist10 command-sampling knobs (CommandsCfg). The stand/walk/run bucket
# sampling is now ported into mjlab's UniformVelocityCommand (enabled by
# rel_walk_envs > 0), so V9 reproduces hist10's command distribution exactly.
_HIST10_MAX_LIN_ACCEL = 3.0
_HIST10_CMD_STEP_SAMPLE_FRAC = 0.3
_HIST10_RUN_HIGH_SPEED_SAMPLE_FRAC = 0.4
_HIST10_RUN_HIGH_SPEED_THRESHOLD_FRAC = 0.6
_HIST10_RUN_HIGH_SPEED_ENTER = 1.05
_HIST10_REL_STANDING_ENVS = 0.05
_HIST10_REL_WALK_ENVS = 0.20
_HIST10_WALK_LIN_VEL_X = (-0.5, 1.0)
_HIST10_RUN_LIN_VEL_X_MIN = 1.0
_HIST10_REL_HEADING_ENVS = 0.6
_HIST10_HEADING_CONTROL_STIFFNESS = 0.9
_HIST10_STAND_ENTER_SPEED = 0.08


def _make_hist10_rewards() -> dict[str, RewardTermCfg]:
  """1:1 port of ``UltraGameYawHist10RewardCfg`` to the mjlab API."""
  R = RewardTermCfg
  S = SceneEntityCfg

  return {
    # ── Task ──
    "stand_still": R(
      func=ultra_mdp.stand_still,
      weight=10.0,
      params={"target_angles": dict(_STAND_TARGET_ANGLES), "style_mask": [0]},
    ),
    "track_lin_vel_x_exp": R(
      func=ultra_mdp.track_lin_vel_x_yaw_frame_exp,
      weight=15.0,
      params={
        "std": 0.5,
        "std_speed_max": 2.0,
        "speed_walk": 1.0,
        "speed_max": 15.0,
        "style_mask": [1, 2],
      },
    ),
    "track_walk_lin_vel_xy_avg_exp": R(
      func=ultra_mdp.track_lin_vel_xy_yaw_frame_avg_exp,
      weight=8.0,
      params={"std": 0.1, "style_mask": [1]},
    ),
    "track_walk_lin_vel_y_avg_exp": R(
      func=ultra_mdp.track_lin_vel_y_yaw_frame_avg_exp,
      weight=8.0,
      params={"std": 0.25, "style_mask": [1]},
    ),
    "track_walk_lin_vel_y_neg": R(
      func=ultra_mdp.track_lin_vel_y_yaw_frame_neg,
      weight=-0.3,
      params={"style_mask": [1]},
    ),
    "track_lin_vel_y_exp": R(
      func=ultra_mdp.track_lin_vel_y_yaw_frame_exp,
      weight=2.0,
      params={"std": 0.3, "style_mask": [1, 2]},
    ),
    "track_lin_vel_z_exp": R(
      func=ultra_mdp.track_lin_vel_z_yaw_frame_exp,
      weight=0.05,
      params={"std": 0.3, "style_mask": [2]},
    ),
    "track_ang_vel_z_exp": R(
      func=ultra_mdp.track_ang_vel_z_world_exp,
      weight=5.0,
      params={"std": 0.3, "style_mask": [1, 2]},
    ),
    "track_ang_vel_z_avg_exp": R(
      func=ultra_mdp.track_ang_vel_z_world_avg_exp,
      weight=4.0,
      params={"std": 0.25, "style_mask": [2]},
    ),
    "track_walk_ang_vel_z_avg_exp": R(
      func=ultra_mdp.track_ang_vel_z_world_avg_exp,
      weight=8.0,
      params={"std": 0.1, "style_mask": [1]},
    ),
    "track_lin_vel_x_bonus": R(
      func=ultra_mdp.track_lin_vel_x_yaw_frame_bonus,
      weight=2.0,
      params={
        "std": 0.5,
        "std_speed_max": 2.0,
        "speed_walk": 1.0,
        "speed_max": 15.0,
        "style_mask": [2],
      },
    ),
    "track_lin_vel_x_neg": R(
      func=ultra_mdp.track_lin_vel_x_yaw_frame_neg,
      weight=-0.3,
      params={"style_mask": [2]},
    ),
    "track_ang_vel_z_bonus": R(
      func=ultra_mdp.track_ang_vel_z_world_bonus,
      weight=2.0,
      params={"std": 0.3, "style_mask": [2]},
    ),
    "track_ang_vel_z_neg": R(
      func=ultra_mdp.track_ang_vel_z_world_neg,
      weight=-0.5,
      params={"style_mask": [2]},
    ),
    "alive": R(func=ultra_mdp.is_alive, weight=2.0),
    "termination_penalty": R(func=ultra_mdp.is_terminated, weight=-200.0),
    # ── Regularization ──
    "energy": R(func=ultra_mdp.energy, weight=-3.0e-4),
    "dof_acc_l2": R(func=ultra_mdp.joint_acc_l2, weight=-2.5e-7),
    "action_rate_l2": R(
      func=ultra_mdp.action_rate_l2, weight=-0.1, params={"max_penalty": 1.0}
    ),
    "action_smoothness": R(
      func=ultra_mdp.action_smoothness, weight=-0.01, params={"max_penalty": 0.25}
    ),
    "action_l2_stand": R(
      func=ultra_mdp.action_l2,
      weight=-0.5,
      params={"style_mask": [0], "max_penalty": 2.0},
    ),
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
    "waist_yaw_dof_pos_neg": R(
      func=ultra_mdp.waist_yaw_dof_pos_neg,
      weight=-0.8,
      params={
        "speed_threshold": 8.0,
        "speed_max": 15.0,
        "penalty_scale_max": 3.0,
        "style_mask": [1, 2],
      },
    ),
    "feet_air_time": R(
      func=ultra_mdp.feet_air_time_positive_biped,
      weight=30.0,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold": 0.5,
        "style_mask": [1],
      },
    ),
    "feet_air_time_run": R(
      func=ultra_mdp.feet_air_time_positive_biped,
      weight=2.0,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold": 0.5,
        "style_mask": [2],
      },
    ),
    "dof_pos_limits": R(
      func=envs_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": S("robot", joint_names=(".*",))},
    ),
    "ankle_pitch_pos_limits": R(
      func=ultra_mdp.ankle_pitch_pos_limits_neg, weight=-60.0
    ),
    "hip_roll_action": R(
      func=ultra_mdp.hip_roll_action, weight=-1.5, params={"style_mask": [1, 2]}
    ),
    "hip_yaw_action": R(
      func=ultra_mdp.hip_yaw_action,
      weight=-1.5,
      params={"yaw_cmd_relax_at": 0.5, "style_mask": [1, 2]},
    ),
    "hip_roll_deviation_l1": R(
      func=ultra_mdp.hip_roll_deviation_l1,
      weight=-7.5,
      params={
        "asset_cfg": S("robot", joint_names=[".*hip_roll.*"]),
        "style_mask": [1, 2],
      },
    ),
    "hip_yaw_deviation_l1": R(
      func=ultra_mdp.hip_yaw_deviation_l1,
      weight=-7.5,
      params={
        "asset_cfg": S("robot", joint_names=[".*hip_yaw.*"]),
        "style_mask": [1, 2],
      },
    ),
    # ── Joint torque & velocity ──
    "torques_weighted_neg": R(func=ultra_mdp.torques_weighted_neg, weight=-7.0e-6),
    "wholebody_vel_weighted_neg": R(
      func=ultra_mdp.wholebody_vel_weighted_neg, weight=-0.007
    ),
    "dof_vel_pen_l2s": R(
      func=ultra_mdp.dof_vel_pen_l2s,
      weight=-1e-3,
      params={
        "asset_cfg": S(
          "robot", joint_names=[".*hip.*", ".*knee.*", ".*ankle.*", ".*shoulder.*"]
        ),
        "style_mask": [1, 2],
      },
    ),
    "hip_yaw_vel_cmd_neg": R(func=ultra_mdp.hip_yaw_vel_cmd_neg, weight=-0.05),
    # ── Stability & posture ──
    "base_height_neg": R(
      func=ultra_mdp.base_height_neg,
      weight=-10.0,
      params={"target_height": 1.18, "style_mask": [0, 1, 2]},
    ),
    "lin_vel_z_l2": R(
      func=ultra_mdp.lin_vel_z_l2,
      weight=-1.0,
      params={"style_mask": [0, 1], "max_penalty": 0.5},
    ),
    "ang_vel_xy_l2": R(func=ultra_mdp.ang_vel_xy_l2, weight=-1.0),
    "body_orientation_exp": R(
      func=ultra_mdp.body_orientation_exp,
      weight=1.0,
      params={
        "asset_cfg": S("robot", body_names=("waist_yaw_link",)),
        "style_mask": [0, 1],
      },
    ),
    # ── Style / contact ──
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
    # ── Feet spacing ──
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
    # ── Feet contact force ──
    "gait_feet_force_max_neg": R(
      func=ultra_mdp.gait_feet_force_max_neg,
      weight=-0.01,
      params={
        "sensor_name": "feet_ground_contact",
        "max_force": 1500.0,
        "style_mask": [1, 2],
      },
    ),
    # ── Run style ──
    "body_orientation_speed_aware": R(
      func=ultra_mdp.body_orientation_speed_aware,
      weight=4.0,
      params={
        "asset_cfg": S("robot", body_names=("waist_yaw_link",)),
        "pitch_coef": 0.025,
        "roll_coef": 0.05,
        "style_mask": [2],
      },
    ),
  }


def ultra_game_yaw_v9_env_cfg(play: bool = False):
  """v7 env (HIM + DR + PD + slew) with the reward set and velocity curriculum
  replaced by the ultra_run_lab hist10 branch."""
  cfg = ultra_game_yaw_v7_env_cfg(play=play)

  # ── Rewards: full hist10 set ────────────────────────────────────────
  cfg.rewards = _make_hist10_rewards()

  # ── Command sampling matched to hist10 ──────────────────────────────
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.max_lin_accel = _HIST10_MAX_LIN_ACCEL
  twist_cmd.cmd_step_sample_frac = _HIST10_CMD_STEP_SAMPLE_FRAC
  twist_cmd.run_high_speed_sample_frac = _HIST10_RUN_HIGH_SPEED_SAMPLE_FRAC
  twist_cmd.run_high_speed_threshold_frac = _HIST10_RUN_HIGH_SPEED_THRESHOLD_FRAC
  twist_cmd.run_high_speed_enter = _HIST10_RUN_HIGH_SPEED_ENTER
  twist_cmd.rel_heading_envs = _HIST10_REL_HEADING_ENVS
  twist_cmd.heading_control_stiffness = _HIST10_HEADING_CONTROL_STIFFNESS
  # stand/walk/run bucket sampling (hist10 parity).
  twist_cmd.rel_standing_envs = _HIST10_REL_STANDING_ENVS
  twist_cmd.rel_walk_envs = _HIST10_REL_WALK_ENVS
  twist_cmd.walk_lin_vel_x = _HIST10_WALK_LIN_VEL_X
  twist_cmd.run_lin_vel_x_min = _HIST10_RUN_LIN_VEL_X_MIN
  twist_cmd.stand_enter_speed = _HIST10_STAND_ENTER_SPEED

  if not play:
    # ── Curriculum: hist10 performance-gated ramp with retreat ────────
    cfg.curriculum.pop("command_vel", None)
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
      func=ultra_mdp.commands_vel_progressive,
      params=dict(_HIST10_CURRICULUM_PARAMS),
    )

  return cfg


def ultra_game_yaw_amp_him_v9_runner_cfg() -> RslRlAmpHimRunnerCfg:
  """v9 runner: hist10-aligned 10-frame actor + critic history (HIM kept)."""
  cfg = ultra_game_yaw_amp_him_v7_runner_cfg()
  cfg.experiment_name = "ultra_game_yaw_amp_him_v9"
  cfg.wandb_project = "ultra_game_yaw_amp_him_v9"
  cfg.actor_obs_history_length = 10
  cfg.critic_obs_history_length = 10
  return cfg
