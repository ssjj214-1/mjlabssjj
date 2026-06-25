"""Ultra GameYaw MDP terms ported from ``legged_lab.mdp`` to mjlab.

This module ports the Ultra Isaac-Lab reward set, privileged critic obs, and
symmetry data-augmentation function so the mjlab task can train against the
same reward landscape as the original library.

API notes (Isaac Lab -> mjlab translation):
- ``asset.data.root_quat_w``       -> ``asset.data.root_link_quat_w``
- ``asset.data.root_lin_vel_w``    -> ``asset.data.root_link_lin_vel_w``
- ``asset.data.root_pos_w``        -> ``asset.data.root_link_pos_w``
- ``asset.data.body_pos_w[:, ids]``-> ``asset.data.body_link_pos_w[:, ids]``
- ``asset.data.body_quat_w``       -> ``asset.data.body_link_quat_w``
- ``asset.data.body_lin_vel_w``    -> ``asset.data.body_link_lin_vel_w``
- ``asset.data.applied_torque``    -> ``asset.data.qfrc_actuator``
- ``asset.data.GRAVITY_VEC_W``     -> ``env.scene["robot"].data.gravity_vec_w``
- ``env.command_generator.command``-> ``env.command_manager.get_command("twist")``
- ``env.action_buffer._circular_buffer.buffer[:,-1]``
                                   -> ``env.action_manager.action``
- ``contact_sensor.data.net_forces_w`` (B, num_bodies, 3)
                                   -> ``contact_sensor.data.force`` (B, N, 3)
                                       with N = num_primaries (num_slots=1).
- ``contact_sensor.data.net_forces_w_history``
                                   -> ``contact_sensor.data.force_history``
                                       (B, N, H, 3); index 0 = most recent.

Per-env Ultra "style id" hysteresis (stand/walk/run, 0/1/2) is updated via the
``ultra_style_update`` event below, which writes ``env.style_ids``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# ---------------------------------------------------------------------------
# Ultra-specific id caches.
# ---------------------------------------------------------------------------

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Joint name conventions for Ultra 13-DoF (matches asset_zoo MJCF + AMP layout).
_HIP_YAW_JOINTS = ("hip_yaw_l_joint", "hip_yaw_r_joint")
_HIP_ROLL_JOINTS = ("hip_roll_l_joint", "hip_roll_r_joint")
_HIP_PITCH_JOINTS = ("hip_pitch_l_joint", "hip_pitch_r_joint")
_KNEE_JOINTS = ("knee_pitch_l_joint", "knee_pitch_r_joint")
_ANKLE_JOINTS = ("ankle_pitch_l_joint", "ankle_pitch_r_joint")
_SHOULDER_JOINTS = ("shoulder_pitch_l_joint", "shoulder_pitch_r_joint")
_WAIST_JOINTS = ("waist_yaw_joint",)
_LEFT_LEG_JOINTS = (
  "hip_yaw_l_joint",
  "hip_roll_l_joint",
  "hip_pitch_l_joint",
  "knee_pitch_l_joint",
  "ankle_pitch_l_joint",
)
_RIGHT_LEG_JOINTS = (
  "hip_yaw_r_joint",
  "hip_roll_r_joint",
  "hip_pitch_r_joint",
  "knee_pitch_r_joint",
  "ankle_pitch_r_joint",
)

_FOOT_BODY_CANDIDATES = (
  ("ankle_roll_l_link", "ankle_roll_r_link"),
  ("ankle_pitch_l_link", "ankle_pitch_r_link"),
)
_KNEE_BODIES = ("knee_pitch_l_link", "knee_pitch_r_link")
_BASE_BODY = ("base_link",)
_WAIST_BODY = ("waist_yaw_link",)


def select_foot_body_names(body_names: list[str] | tuple[str, ...]) -> tuple[str, str]:
  """Return the physical foot bodies for the given robot model.

  V10 inserts passive ankle-roll links as the actual foot bodies. Earlier
  variants only have ankle-pitch links, so we fall back to those.
  """
  body_name_set = set(body_names)
  for candidate in _FOOT_BODY_CANDIDATES:
    if all(name in body_name_set for name in candidate):
      return candidate
  raise KeyError(f"No supported foot bodies found in robot bodies: {body_names}")


def _ensure_id_cache(env: "ManagerBasedRlEnv") -> None:
  """Resolve and cache joint/body indices on ``env`` (idempotent)."""
  if getattr(env, "_ultra_ids_cached", False):
    return
  robot: Entity = env.scene["robot"]

  def _jids(names):
    ids, _ = robot.find_joints(list(names), preserve_order=True)
    return torch.tensor(ids, device=env.device, dtype=torch.long)

  def _bids(names):
    ids, _ = robot.find_bodies(list(names), preserve_order=True)
    return torch.tensor(ids, device=env.device, dtype=torch.long)

  env.hip_yaw_ids = _jids(_HIP_YAW_JOINTS)
  env.hip_roll_ids = _jids(_HIP_ROLL_JOINTS)
  env.hip_pitch_ids = _jids(_HIP_PITCH_JOINTS)
  env.knee_ids = _jids(_KNEE_JOINTS)
  env.ankle_pitch_ids = _jids(_ANKLE_JOINTS)
  env.shoulder_pitch_ids = _jids(_SHOULDER_JOINTS)
  env.waist_yaw_ids = _jids(_WAIST_JOINTS)
  env.left_leg_ids = _jids(_LEFT_LEG_JOINTS)
  env.right_leg_ids = _jids(_RIGHT_LEG_JOINTS)

  env.feet_body_ids = _bids(select_foot_body_names(robot.body_names))
  env.knee_body_ids = _bids(_KNEE_BODIES)
  env.base_body_id = _bids(_BASE_BODY)[0].item()
  env.waist_body_id = _bids(_WAIST_BODY)[0].item()

  # Per-joint xhumanoid weights (mapped by name -> joint index).
  joint_names = robot.joint_names

  def _by_name(weights: dict[str, float]) -> torch.Tensor:
    return torch.tensor(
      [float(weights.get(n, 1.0)) for n in joint_names],
      device=env.device,
      dtype=torch.float32,
    )

  env._xhumanoid_torque_weights = _by_name(_XHUMANOID_TORQUE_WEIGHTS)
  env._xhumanoid_vel_weights = _by_name(_XHUMANOID_VEL_WEIGHTS)

  # Initial style buffer.
  if not hasattr(env, "style_ids"):
    env.style_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  env._ultra_ids_cached = True


_XHUMANOID_TORQUE_WEIGHTS: dict[str, float] = {
  "hip_yaw_l_joint": 1.5,
  "hip_yaw_r_joint": 1.5,
  "hip_roll_l_joint": 1.0,
  "hip_roll_r_joint": 1.0,
  "hip_pitch_l_joint": 0.6,
  "hip_pitch_r_joint": 0.6,
  "knee_pitch_l_joint": 0.6,
  "knee_pitch_r_joint": 0.6,
  "ankle_pitch_l_joint": 1.2,
  "ankle_pitch_r_joint": 1.2,
  "waist_yaw_joint": 3.0,
  "shoulder_pitch_l_joint": 2.0,
  "shoulder_pitch_r_joint": 2.0,
}

_XHUMANOID_VEL_WEIGHTS: dict[str, float] = {
  "hip_yaw_l_joint": 1.0,
  "hip_yaw_r_joint": 1.0,
  "hip_roll_l_joint": 1.0,
  "hip_roll_r_joint": 1.0,
  "hip_pitch_l_joint": 0.2,
  "hip_pitch_r_joint": 0.2,
  "knee_pitch_l_joint": 0.2,
  "knee_pitch_r_joint": 0.2,
  "ankle_pitch_l_joint": 0.4,
  "ankle_pitch_r_joint": 0.4,
  "waist_yaw_joint": 3.0,
  "shoulder_pitch_l_joint": 0.5,
  "shoulder_pitch_r_joint": 0.5,
}


# ---------------------------------------------------------------------------
# Style update event (mode="step").
# ---------------------------------------------------------------------------


def ultra_style_update(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None = None,
  command_name: str = "twist",
  stand_enter: float = 0.08,
  stand_exit: float = 0.12,
  run_enter: float = 1.05,
  run_exit: float = 0.85,
) -> None:
  """Update ``env.style_ids`` from the current commanded |vx| with hysteresis.

  Style mapping: 0=stand, 1=walk, 2=run. Runs as a ``mode="step"`` event so
  it executes after command_manager.compute() and before the next reward
  computation.
  """
  _ensure_id_cache(env)
  cmd = env.command_manager.get_command(command_name)
  if cmd is None:
    return
  speed_x = cmd[:, 0].abs()
  s = env.style_ids
  new = s.clone()

  from_stand = s == 0
  new[from_stand & (speed_x >= run_enter)] = 2
  new[from_stand & (speed_x >= stand_exit) & (speed_x < run_enter)] = 1

  from_walk = s == 1
  new[from_walk & (speed_x >= run_enter)] = 2
  new[from_walk & (speed_x < stand_enter)] = 0

  from_run = s == 2
  new[from_run & (speed_x < run_exit) & (speed_x >= stand_enter)] = 1
  new[from_run & (speed_x < stand_enter)] = 0

  env.style_ids = new


def ultra_style_update_accel(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None = None,
  command_name: str = "twist",
  stand_enter: float = 0.08,
  stand_exit: float = 0.12,
  run_enter: float = 1.05,
  run_exit: float = 0.85,
  accel_window_s: float = 4.0,
) -> None:
  """4-style scheduler: stand(0) / walk(1) / run(2) + accel-launch(3).

  Same speed-driven hysteresis as :func:`ultra_style_update`, but envs commanded
  to run get style id ``3`` for the first ``accel_window_s`` seconds of the
  episode. Because the AMP positives are sampled by *episode time*, this makes
  the stand-to-run expert clip time-align with the robot's own launch — i.e. the
  accel discriminator shapes the standstill→sprint acceleration phase. After the
  window the env falls back to run(2).
  """
  _ensure_id_cache(env)
  cmd = env.command_manager.get_command(command_name)
  if cmd is None:
    return
  speed_x = cmd[:, 0].abs()
  s = env.style_ids
  # Treat current accel(3) like run(2) for the base stand/walk/run hysteresis.
  base = torch.where(s == 3, torch.full_like(s, 2), s)
  new = base.clone()

  from_stand = base == 0
  new[from_stand & (speed_x >= run_enter)] = 2
  new[from_stand & (speed_x >= stand_exit) & (speed_x < run_enter)] = 1

  from_walk = base == 1
  new[from_walk & (speed_x >= run_enter)] = 2
  new[from_walk & (speed_x < stand_enter)] = 0

  from_run = base == 2
  new[from_run & (speed_x < run_exit) & (speed_x >= stand_enter)] = 1
  new[from_run & (speed_x < stand_enter)] = 0

  # Launch-window override: early episode + commanded to run -> accel style.
  ep_t = env.episode_length_buf.to(torch.float32) * env.step_dt
  launch = (ep_t < accel_window_s) & (speed_x >= run_enter)
  new = torch.where(launch, torch.full_like(new, 3), new)

  env.style_ids = new


def _style_mask(env: "ManagerBasedRlEnv", style_mask: list[int] | None) -> torch.Tensor:
  if style_mask is None:
    return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
  ref = torch.tensor(style_mask, device=env.device, dtype=torch.long)
  return torch.isin(env.style_ids, ref).to(torch.float32)


# ---------------------------------------------------------------------------
# Stand-still stability (V4): directly penalize base drift so the robot does
# not wander / shuffle at zero command. Applied via ``style_mask=[0]`` so it
# never fights velocity tracking while walking/running.
# ---------------------------------------------------------------------------


def stand_base_vel_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """L2 penalty on base horizontal speed (world xy). Stand-style drift killer."""
  asset: Entity = env.scene[asset_cfg.name]
  vel_xy = asset.data.root_link_lin_vel_w[:, :2]
  cost = torch.sum(torch.square(vel_xy), dim=1)
  return cost * _style_mask(env, style_mask)


def stand_base_motion_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  command_name: str = "twist",
  cmd_threshold: float = 0.1,
  ang_vel_scale: float = 1.0,
) -> torch.Tensor:
  """L2 penalty on full base motion, gated on a ~zero velocity command.

  Unlike ``stand_base_vel_l2`` (style-gated, horizontal speed only), this fires
  whenever the *command* is ~0 (norm < ``cmd_threshold``) and penalizes both the
  base horizontal linear velocity (world xy) and the base angular velocity (body
  frame, all 3 axes). At a zero command the floating base should be still, so
  this directly punishes the trunk swaying / rocking that ``stand_base_vel_l2``
  (linear-only) leaves untouched.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  standing = (torch.norm(command, dim=1) < cmd_threshold).to(torch.float32)
  lin_xy = torch.sum(torch.square(asset.data.root_link_lin_vel_w[:, :2]), dim=1)
  ang = torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)
  return (lin_xy + ang_vel_scale * ang) * standing


def stand_joint_vel_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  command_name: str = "twist",
  cmd_threshold: float = 0.1,
) -> torch.Tensor:
  """L2 penalty on ALL joint velocities, gated on a ~zero velocity command.

  ``stand_base_motion_l2`` stills the floating base, but the limbs can still
  oscillate in place (knees/elbows jittering) while the base stays put. At a zero
  command nothing should move, so this penalizes every joint's velocity whenever
  the command is ~0 (norm < ``cmd_threshold``).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  standing = (torch.norm(command, dim=1) < cmd_threshold).to(torch.float32)
  cost = torch.sum(torch.square(asset.data.joint_vel), dim=1)
  return cost * standing


# ---------------------------------------------------------------------------
# Progressive velocity curriculum (V4) — Isaac-faithful, performance-gated.
#
# Replaces the fixed-step stage table. The forward-speed cap grows by
# ``update_step`` (m/s) ONLY when the run-style (style 2) envs both
#   (a) survive long enough  : mean episode length > ``len_ratio`` * max, and
#   (b) track forward speed   : mean exp tracking reward > a speed-dependent
#       gate that linearly decays ``rwd_threshold`` -> ``rwd_threshold_min`` as
#       the cap goes ``rwd_v_lo`` -> ``rwd_v_hi``.
# Metrics use the instant per-window mean (no EMA, matching ultra_run_lab) and
# the gate is evaluated once per
# episode-length window. Runs as a curriculum term in ``_reset_idx`` *before*
# the reward/episode buffers reset, so the just-ended episode's
# ``_episode_sums`` and ``episode_length_buf`` are read directly (the same data
# Isaac caches in ``_cache_episode_curriculum_metrics``).
# ---------------------------------------------------------------------------


def commands_vel_progressive(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | slice,
  *,
  command_name: str = "twist",
  init_lin_vel_x: tuple[float, float] = (-0.5, 1.5),
  update_step: float = 0.2,
  max_lin_vel_x: float = 15.0,
  len_ratio: float = 0.8,
  rwd_threshold: float = 0.6,
  rwd_threshold_min: float = 0.4,
  rwd_v_lo: float = 5.0,
  rwd_v_hi: float = 15.0,
  track_term: str = "track_lin_vel_x",
  run_style_id: int = 2,
  retreat_len_ratio: float | None = None,
  retreat_rwd_offset: float | None = None,
  retreat_floor_x: float | None = None,
  check_interval_steps: int | None = None,
) -> dict[str, torch.Tensor]:
  """Performance-gated velocity-cap ramp (ultra_run_lab parity).

  When ``retreat_*`` params are given (used by v9 to mirror the hist10
  curriculum), the cap can also *retreat* toward ``retreat_floor_x`` if the
  recent run-style episodes regress: either ``last_len < retreat_len_ratio`` or
  ``last_track < rwd_gate - retreat_rwd_offset``. Leaving them ``None`` (v4-v8)
  keeps the original advance-only behaviour. ``check_interval_steps`` overrides
  the per-``max_episode_length`` evaluation window (hist10 uses 1000 steps).
  """
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

  window = (
    int(check_interval_steps) if check_interval_steps else int(env.max_episode_length)
  )
  floor_x = (
    float(retreat_floor_x) if retreat_floor_x is not None else float(init_lin_vel_x[1])
  )

  st = getattr(env, "_v4_cmd_curriculum", None)
  if st is None:
    # Start the cap low; the ramp grows it as performance allows.
    cfg.ranges.lin_vel_x = init_lin_vel_x
    st = {
      "last_len": 0.0,
      "last_track": 0.0,
      "next_eval": int(window),
    }
    env._v4_cmd_curriculum = st  # type: ignore[attr-defined]

  if isinstance(env_ids, slice):
    idx = torch.arange(env.num_envs, device=env.device)[env_ids]
  else:
    idx = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).view(-1)

  # Accumulate just-ended run-style episode metrics (instant mean, no EMA).
  if idx.numel() > 0:
    ep_len = env.episode_length_buf[idx].float()
    sel = (ep_len > 0.5) & (env.style_ids[idx] == run_style_id)
    if torch.any(sel):
      try:
        ti = env.reward_manager._term_names.index(track_term)
        w = abs(float(env.reward_manager._term_cfgs[ti].weight))
      except (ValueError, AttributeError):
        w = 1.0
      w = w if w > 1e-6 else 1.0
      ep_time = (ep_len * env.step_dt).clamp(min=env.step_dt)
      track_sum = env.reward_manager._episode_sums[track_term][idx]
      raw_track = (track_sum / ep_time) / w  # ~ mean exp(.) in [0, 1]
      len_ratio_now = ep_len / float(env.max_episode_length)
      # Instant per-window mean (no EMA), matching ultra_run_lab.
      st["last_len"] = float(len_ratio_now[sel].mean())
      st["last_track"] = float(raw_track[sel].mean())

  # Evaluate advancement once per episode-length window.
  advanced = 0.0
  retreated = 0.0
  vmax = float(cfg.ranges.lin_vel_x[1])
  if env.common_step_counter >= st["next_eval"]:
    st["next_eval"] = int(env.common_step_counter) + int(window)
    if rwd_v_hi > rwd_v_lo + 1e-6:
      t = max(0.0, min(1.0, (vmax - rwd_v_lo) / (rwd_v_hi - rwd_v_lo)))
    else:
      t = 0.0
    rwd_gate = rwd_threshold + (rwd_threshold_min - rwd_threshold) * t
    if (
      st["last_len"] > len_ratio
      and st["last_track"] > rwd_gate
      and vmax < max_lin_vel_x - 1e-6
    ):
      new_max = min(vmax + update_step, max_lin_vel_x)
      cfg.ranges.lin_vel_x = (cfg.ranges.lin_vel_x[0], new_max)
      advanced = float(new_max > vmax)
      vmax = new_max
    elif retreat_len_ratio is not None and vmax > floor_x + 1e-6:
      # hist10-style retreat: regressing run episodes lower the cap.
      len_retreat = st["last_len"] < retreat_len_ratio
      track_retreat = retreat_rwd_offset is not None and st["last_track"] < (
        rwd_gate - retreat_rwd_offset
      )
      if len_retreat or track_retreat:
        new_max = max(vmax - update_step, floor_x)
        cfg.ranges.lin_vel_x = (cfg.ranges.lin_vel_x[0], new_max)
        retreated = float(new_max < vmax)
        vmax = new_max

  return {
    "lin_vel_x_max": torch.tensor(vmax),
    "last_len_ratio": torch.tensor(st["last_len"]),
    "last_track": torch.tensor(st["last_track"]),
    "advanced": torch.tensor(advanced),
    "retreated": torch.tensor(retreated),
  }


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _twist_cmd(env: "ManagerBasedRlEnv") -> torch.Tensor:
  cmd = env.command_manager.get_command("twist")
  assert cmd is not None
  return cmd


def _vel_yaw_frame(asset: Entity) -> torch.Tensor:
  """Linear velocity expressed in the yaw-only base frame. Shape (B, 3)."""
  return quat_apply_inverse(
    yaw_quat(asset.data.root_link_quat_w), asset.data.root_link_lin_vel_w[:, :3]
  )


def _ang_vel_world(asset: Entity) -> torch.Tensor:
  return asset.data.root_link_ang_vel_w


def _std_scaled_by_cmd_x(
  env: "ManagerBasedRlEnv",
  std: float,
  std_speed_max: float | None,
  speed_walk: float = 1.0,
  speed_max: float = 15.0,
) -> torch.Tensor:
  """Match Ultra ``_std_scaled_by_cmd_x``: smoothstep-grow std with |cmd_x|.

  ``alpha`` smoothsteps from 0 to 1 as ``|cmd_x|`` goes ``speed_walk``->
  ``speed_max``; if ``std_speed_max`` is unset the std is constant.
  """
  if std_speed_max is None:
    return torch.full((env.num_envs,), std, device=env.device, dtype=torch.float32)
  cmd_x = _twist_cmd(env)[:, 0].abs()
  alpha = ((cmd_x - speed_walk) / max(speed_max - speed_walk, 1e-3)).clamp(0.0, 1.0)
  alpha = alpha * alpha * (3.0 - 2.0 * alpha)
  return std + (std_speed_max - std) * alpha


def _cmd_axis_relax_scale(
  env: "ManagerBasedRlEnv", cmd_idx: int, range_name: str
) -> torch.Tensor:
  """Relax hip penalties as a command axis grows (ultra_run_lab parity).

  scale -> 1 when ``|cmd[cmd_idx]|`` is 0 and -> 0 at the configured range bound
  (hist10: ang_vel_z, lin_vel_y both in [-0.5, 0.5]).
  """
  command_term = env.command_manager.get_term("twist")
  r = getattr(command_term.cfg.ranges, range_name)
  m = max(abs(float(r[0])), abs(float(r[1])))
  m = m if m > 1e-6 else 1e-6
  return torch.abs(m - _twist_cmd(env)[:, cmd_idx].abs()) / m


# ---------------------------------------------------------------------------
# hist10 per-step state (EMA velocity + action history).
# ---------------------------------------------------------------------------


def _ensure_hist10_state(
  env: "ManagerBasedRlEnv",
  vel_window_s: float = 0.5,
  yaw_window_s: float = 0.3,
) -> None:
  """Refresh hist10 per-step state at most once per env step (idempotent).

  Maintained on ``env`` for the reward terms that need it:

  * ``avg_lin_vel_x_yaw`` / ``avg_lin_vel_y_yaw`` / ``avg_ang_vel_z`` — EMA of
    the yaw-frame base velocity (windows match hist10: 0.5 s / 0.3 s, i.e.
    ``alpha = step_dt / window``). Reset to 0 at episode start.
  * ``_hist10_a_t`` / ``_hist10_a_tm1`` / ``_hist10_a_tm2`` — the last three
    actions, for the 2nd-order ``action_smoothness`` term.

  Called from every reward term that reads this state. The guard on
  ``common_step_counter`` (unique per step, incremented before the reward pass)
  advances the EMA/history exactly once per step regardless of how many readers
  call it, so no extra (non-hist10) helper reward term is needed.
  """
  step = int(env.common_step_counter)
  if getattr(env, "_hist10_state_step", None) == step:
    return
  env._hist10_state_step = step

  asset: Entity = env.scene["robot"]
  first = env.episode_length_buf <= 1

  # ── EMA velocity ────────────────────────────────────────────────────
  vel_yaw = _vel_yaw_frame(asset)
  wz = _ang_vel_world(asset)[:, 2]
  alpha_v = float(env.step_dt) / vel_window_s
  alpha_yaw = float(env.step_dt) / yaw_window_s
  if not hasattr(env, "avg_lin_vel_x_yaw"):
    z = torch.zeros(env.num_envs, device=env.device)
    env.avg_lin_vel_x_yaw = z.clone()
    env.avg_lin_vel_y_yaw = z.clone()
    env.avg_ang_vel_z = z.clone()
  prev_x = torch.where(
    first, torch.zeros_like(env.avg_lin_vel_x_yaw), env.avg_lin_vel_x_yaw
  )
  prev_y = torch.where(
    first, torch.zeros_like(env.avg_lin_vel_y_yaw), env.avg_lin_vel_y_yaw
  )
  prev_w = torch.where(first, torch.zeros_like(env.avg_ang_vel_z), env.avg_ang_vel_z)
  env.avg_lin_vel_x_yaw = alpha_v * vel_yaw[:, 0] + (1.0 - alpha_v) * prev_x
  env.avg_lin_vel_y_yaw = alpha_v * vel_yaw[:, 1] + (1.0 - alpha_v) * prev_y
  env.avg_ang_vel_z = alpha_yaw * wz + (1.0 - alpha_yaw) * prev_w

  # ── Action history (a_t, a_{t-1}, a_{t-2}) ──────────────────────────
  am = env.action_manager
  a_t = am.action
  a_tm1 = am.prev_action
  first_col = first.unsqueeze(1)
  if not hasattr(env, "_hist10_a_tm2_store"):
    env._hist10_a_tm2_store = torch.zeros_like(a_t)
  env._hist10_a_t = a_t
  env._hist10_a_tm1 = torch.where(first_col, torch.zeros_like(a_t), a_tm1)
  env._hist10_a_tm2 = torch.where(
    first_col, torch.zeros_like(a_t), env._hist10_a_tm2_store
  )
  # Save current a_{t-1} so next step it becomes a_{t-2}.
  env._hist10_a_tm2_store = torch.where(first_col, torch.zeros_like(a_t), a_tm1)


def track_lin_vel_x_yaw_frame_avg_exp(
  env: "ManagerBasedRlEnv",
  std: float,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """EMA yaw-frame vx tracking (walk style; averages out swing-leg jitter)."""
  _ensure_hist10_state(env)
  err = torch.abs(_twist_cmd(env)[:, 0] - env.avg_lin_vel_x_yaw)
  return torch.exp(-err / (std * std)) * _style_mask(env, style_mask)


def track_lin_vel_xy_yaw_frame_avg_exp(
  env: "ManagerBasedRlEnv",
  std: float,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """EMA yaw-frame xy-speed tracking (walk style)."""
  _ensure_hist10_state(env)
  cmd = _twist_cmd(env)
  ex = cmd[:, 0] - env.avg_lin_vel_x_yaw
  ey = cmd[:, 1] - env.avg_lin_vel_y_yaw
  err = torch.sqrt(ex.square() + ey.square())
  return torch.exp(-err / (std * std)) * _style_mask(env, style_mask)


def track_lin_vel_y_yaw_frame_avg_exp(
  env: "ManagerBasedRlEnv",
  std: float,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """EMA yaw-frame vy tracking (walk style lateral)."""
  _ensure_hist10_state(env)
  err = torch.abs(_twist_cmd(env)[:, 1] - env.avg_lin_vel_y_yaw)
  return torch.exp(-err / (std * std)) * _style_mask(env, style_mask)


def track_ang_vel_z_world_avg_exp(
  env: "ManagerBasedRlEnv",
  std: float,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """EMA world yaw-rate tracking (suppresses heading drift)."""
  _ensure_hist10_state(env)
  err = torch.square(_twist_cmd(env)[:, 2] - env.avg_ang_vel_z)
  return torch.exp(-err / (std * std)) * _style_mask(env, style_mask)


def track_lin_vel_y_yaw_frame_neg(
  env: "ManagerBasedRlEnv",
  style_mask: list[int] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Linear yaw-frame vy tracking error penalty (no std/saturation)."""
  asset: Entity = env.scene[asset_cfg.name]
  err = torch.abs(_twist_cmd(env)[:, 1] - _vel_yaw_frame(asset)[:, 1])
  return err * _style_mask(env, style_mask)


def track_ang_vel_z_world_neg(
  env: "ManagerBasedRlEnv",
  style_mask: list[int] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Linear world yaw-rate tracking error penalty (no std/saturation)."""
  asset: Entity = env.scene[asset_cfg.name]
  err = torch.abs(_twist_cmd(env)[:, 2] - _ang_vel_world(asset)[:, 2])
  return err * _style_mask(env, style_mask)


# ---------------------------------------------------------------------------
# Tracking rewards.
# ---------------------------------------------------------------------------


def track_lin_vel_x_yaw_frame_exp(
  env: "ManagerBasedRlEnv",
  std: float,
  std_speed_max: float | None = None,
  speed_walk: float = 1.0,
  speed_max: float = 15.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  # Ultra uses an L1 (absolute) error inside the exponential, not L2.
  asset: Entity = env.scene[asset_cfg.name]
  vel = _vel_yaw_frame(asset)
  err = (_twist_cmd(env)[:, 0] - vel[:, 0]).abs()
  std_eff = _std_scaled_by_cmd_x(env, std, std_speed_max, speed_walk, speed_max)
  return torch.exp(-err / (std_eff * std_eff)) * _style_mask(env, style_mask)


def track_lin_vel_y_yaw_frame_exp(
  env: "ManagerBasedRlEnv",
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  vel = _vel_yaw_frame(asset)
  err = (_twist_cmd(env)[:, 1] - vel[:, 1]).abs()
  return torch.exp(-err / (std * std)) * _style_mask(env, style_mask)


def track_lin_vel_z_yaw_frame_exp(
  env: "ManagerBasedRlEnv",
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  vel = _vel_yaw_frame(asset)
  return torch.exp(-vel[:, 2].abs() / (std * std)) * _style_mask(env, style_mask)


def track_ang_vel_z_world_exp(
  env: "ManagerBasedRlEnv",
  std: float,
  std_speed_max: float | None = None,
  speed_walk: float = 1.0,
  speed_max: float = 15.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  # ``std_speed_max`` lets the yaw-rate tolerance *shrink* with forward speed
  # (pass a value < std), so heading is held much tighter at high speed where
  # drift over a 100 m straight is catastrophic.
  asset: Entity = env.scene[asset_cfg.name]
  err = (_twist_cmd(env)[:, 2] - _ang_vel_world(asset)[:, 2]).square()
  std_eff = _std_scaled_by_cmd_x(env, std, std_speed_max, speed_walk, speed_max)
  return torch.exp(-err / (std_eff * std_eff)) * _style_mask(env, style_mask)


def track_lin_vel_x_yaw_frame_bonus(
  env: "ManagerBasedRlEnv",
  std: float,
  std_speed_max: float | None = None,
  speed_walk: float = 1.0,
  speed_max: float = 15.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """Ultra bonus: ``exp(-|err_x| / std_eff^2) * |cmd_x|``.

  Same exponential shaping as :func:`track_lin_vel_x_yaw_frame_exp`, scaled by
  the commanded forward speed magnitude so high-speed tracking is rewarded more.
  """
  asset: Entity = env.scene[asset_cfg.name]
  vel = _vel_yaw_frame(asset)
  err = (_twist_cmd(env)[:, 0] - vel[:, 0]).abs()
  std_eff = _std_scaled_by_cmd_x(env, std, std_speed_max, speed_walk, speed_max)
  cmd_scale = _twist_cmd(env)[:, 0].abs()
  return (
    torch.exp(-err / (std_eff * std_eff)) * cmd_scale * _style_mask(env, style_mask)
  )


def track_lin_vel_x_yaw_frame_neg(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """Ultra penalty: raw absolute forward-velocity tracking error |cmd_x - v_x|."""
  asset: Entity = env.scene[asset_cfg.name]
  vel = _vel_yaw_frame(asset)
  err = (_twist_cmd(env)[:, 0] - vel[:, 0]).abs()
  return err * _style_mask(env, style_mask)


def track_ang_vel_z_world_bonus(
  env: "ManagerBasedRlEnv",
  std: float,
  std_speed_max: float | None = None,
  speed_walk: float = 1.0,
  speed_max: float = 15.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """Ultra bonus: ``exp(-err_z^2 / std_eff^2) * |cmd_x|`` (scaled by forward speed).

  ``std_speed_max`` tightens the yaw-rate tolerance as forward speed grows.
  """
  asset: Entity = env.scene[asset_cfg.name]
  err = (_twist_cmd(env)[:, 2] - _ang_vel_world(asset)[:, 2]).square()
  std_eff = _std_scaled_by_cmd_x(env, std, std_speed_max, speed_walk, speed_max)
  cmd_scale = _twist_cmd(env)[:, 0].abs()
  return (
    torch.exp(-err / (std_eff * std_eff)) * cmd_scale * _style_mask(env, style_mask)
  )


def ang_vel_z_straight_neg(
  env: "ManagerBasedRlEnv",
  yaw_cmd_relax_at: float = 0.3,
  speed_ref: float = 10.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """High-speed straight-line yaw-drift suppression (deployment-critical).

  On the real robot only a yaw-*rate* command is sent (no heading feedback), so
  any residual yaw rate integrates into heading drift — over a 100 m sprint even
  0.1 rad/s veers the robot off the track. This is an L1 yaw-rate-error penalty
  that (a) grows with forward speed (``cmd_x / speed_ref``, ~0 when walking) and
  (b) is gated to near-straight commands (``|cmd_yaw| < yaw_cmd_relax_at``), so
  it only bites exactly in the high-speed straight-running regime and never
  fights commanded turns.
  """
  asset: Entity = env.scene[asset_cfg.name]
  err = (_twist_cmd(env)[:, 2] - _ang_vel_world(asset)[:, 2]).abs()
  cmd_yaw = _twist_cmd(env)[:, 2].abs()
  straight = 1.0 - (cmd_yaw / yaw_cmd_relax_at).clamp(0.0, 1.0)
  speed_scale = (_twist_cmd(env)[:, 0].abs() / max(speed_ref, 1e-3)).clamp(min=0.0)
  return err * straight * speed_scale * _style_mask(env, style_mask)


# ---------------------------------------------------------------------------
# Lifecycle.
# ---------------------------------------------------------------------------


def is_alive(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Reward of 1 every step the env is alive."""
  return (~env.reset_buf).float()


def is_terminated(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Penalty applied only on non-timeout terminations."""
  if not hasattr(env, "termination_manager"):
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  terminated = env.termination_manager.terminated.float()
  time_outs = env.termination_manager.time_outs.float()
  return terminated * (1.0 - time_outs)


def stand_still(
  env: "ManagerBasedRlEnv",
  target_angles: dict[str, float] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """Ultra ``stand_still``: ``exp(-sum((q - q_target)^2))`` (typically style 0).

  ``target_angles`` is a ``{joint_name: angle}`` mapping resolved into mjlab
  joint order (unspecified joints target 0.0, matching the Ultra default pose);
  when ``None`` the asset default joint positions are used.
  """
  _ensure_id_cache(env)
  asset: Entity = env.scene[asset_cfg.name]
  if target_angles is None:
    ta = asset.data.default_joint_pos
  else:
    cache = getattr(env, "_stand_still_target", None)
    if cache is None:
      names = asset.joint_names
      cache = torch.tensor(
        [float(target_angles.get(n, 0.0)) for n in names],
        device=env.device,
        dtype=torch.float32,
      ).unsqueeze(0)
      env._stand_still_target = cache
    ta = cache
  angle = asset.data.joint_pos - ta
  return torch.exp(-torch.sum(angle.square(), dim=1)) * _style_mask(env, style_mask)


# ---------------------------------------------------------------------------
# Regularization.
# ---------------------------------------------------------------------------


def energy(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  # Ultra uses the L2 norm of the per-joint power vector, not the L1 sum.
  asset: Entity = env.scene[asset_cfg.name]
  power = (asset.data.qfrc_actuator * asset.data.joint_vel).abs()
  return torch.norm(power, dim=-1)


def joint_acc_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(asset.data.joint_acc.square(), dim=1)


def _clip_penalty(penalty: torch.Tensor, max_penalty: float | None) -> torch.Tensor:
  if max_penalty is not None and max_penalty > 0.0:
    return penalty.clamp(max=max_penalty)
  return penalty


def action_rate_l2(
  env: "ManagerBasedRlEnv", max_penalty: float | None = None
) -> torch.Tensor:
  """First-order action diff ||a_t - a_{t-1}||². ``max_penalty`` caps it
  (before ``weight``); ``None`` keeps the original uncapped behaviour (v4-v8)."""
  am = env.action_manager
  penalty = torch.sum((am.action - am.prev_action).square(), dim=1)
  return _clip_penalty(penalty, max_penalty)


def action_smoothness(
  env: "ManagerBasedRlEnv", max_penalty: float | None = None
) -> torch.Tensor:
  """2nd-order action diff ||a_t - 2 a_{t-1} + a_{t-2}||² (kills sawtooth
  chatter). Uses the action history kept by ``_ensure_hist10_state``."""
  _ensure_hist10_state(env)
  penalty = torch.sum(
    (env._hist10_a_t - 2.0 * env._hist10_a_tm1 + env._hist10_a_tm2).square(), dim=1
  )
  return _clip_penalty(penalty, max_penalty)


def action_l2(
  env: "ManagerBasedRlEnv",
  style_mask: list[int] | None = None,
  max_penalty: float | None = None,
) -> torch.Tensor:
  """Action-magnitude penalty ||a_t||² (stand bucket: action should be ~0)."""
  penalty = _clip_penalty(
    torch.sum(env.action_manager.action.square(), dim=1), max_penalty
  )
  return penalty * _style_mask(env, style_mask)


def ankle_pitch_pos_limits_neg(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  _ensure_id_cache(env)
  asset: Entity = env.scene[asset_cfg.name]
  ids = env.ankle_pitch_ids
  q = asset.data.joint_pos[:, ids]
  lim = asset.data.soft_joint_pos_limits[:, ids]
  out_of_lo = (lim[..., 0] - q).clamp(min=0.0)
  out_of_hi = (q - lim[..., 1]).clamp(min=0.0)
  return torch.sum(out_of_lo + out_of_hi, dim=1)


def arm_dof_pos_neg(
  env: "ManagerBasedRlEnv",
  target_angles: list[float] | None = None,
  speed_threshold: float = 8.0,
  speed_max: float = 15.0,
  penalty_scale_max: float = 3.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """Ultra ``arm_dof_pos_neg``: penalize arm (shoulder_pitch) joints deviating
  from their default pose, with a speed-dependent scale.

  Ported verbatim from the ultra_run_lab hist10 task (it was dropped during the
  mjlab migration). The arms are a single elbow-less ``shoulder_pitch`` DoF; the
  AMP run experts swing them far behind the torso (shoulder_pitch down to
  ~-1.1 rad), so without this penalty the policy copies the backward arm swing
  and walks with its arms behind its back. ``penalty_scale`` is 1.0 for
  ``|cmd_x| <= speed_threshold`` and ramps linearly to ``penalty_scale_max`` at
  ``|cmd_x| >= speed_max``, holding the arms tighter the faster the robot runs.
  ``target_angles`` (when given) overrides the per-joint default target.
  """
  _ensure_id_cache(env)
  asset: Entity = env.scene[asset_cfg.name]
  ids = env.shoulder_pitch_ids
  q = asset.data.joint_pos[:, ids]
  if target_angles is None:
    target = asset.data.default_joint_pos[:, ids]
  else:
    target = torch.tensor(target_angles, device=q.device, dtype=q.dtype)
    target = target.unsqueeze(0).expand(q.shape[0], -1)
  penalty = torch.sum((q - target).square(), dim=1)
  cmd_x = _twist_cmd(env)[:, 0].abs()
  alpha = ((cmd_x - speed_threshold) / (speed_max - speed_threshold)).clamp(0.0, 1.0)
  speed_scale = 1.0 + alpha * (penalty_scale_max - 1.0)
  return penalty * speed_scale * _style_mask(env, style_mask)


def waist_yaw_dof_pos_neg(
  env: "ManagerBasedRlEnv",
  target_angle: float | None = None,
  speed_threshold: float = 8.0,
  speed_max: float = 15.0,
  penalty_scale_max: float = 3.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """Ultra ``waist_yaw_dof_pos_neg``: penalize waist_yaw deviating from its
  default (or ``target_angle``), with the same speed-aware scale as the arm
  term. Ported from hist10 (dropped during the mjlab migration)."""
  _ensure_id_cache(env)
  asset: Entity = env.scene[asset_cfg.name]
  ids = env.waist_yaw_ids
  q = asset.data.joint_pos[:, ids]
  if target_angle is None:
    target = asset.data.default_joint_pos[:, ids]
  else:
    target = torch.full_like(q, float(target_angle))
  penalty = torch.sum((q - target).square(), dim=1)
  cmd_x = _twist_cmd(env)[:, 0].abs()
  alpha = ((cmd_x - speed_threshold) / (speed_max - speed_threshold)).clamp(0.0, 1.0)
  speed_scale = 1.0 + alpha * (penalty_scale_max - 1.0)
  return penalty * speed_scale * _style_mask(env, style_mask)


def hip_roll_action(
  env: "ManagerBasedRlEnv",
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """Ultra ``hip_roll_action``: L1 hip-roll action penalty; relaxes with
  |cmd_yaw| and |cmd_y| (omni-style). Ported from hist10."""
  _ensure_id_cache(env)
  z_scale = _cmd_axis_relax_scale(env, 2, "ang_vel_z")
  y_scale = _cmd_axis_relax_scale(env, 1, "lin_vel_y")
  actions = env.action_manager.action[:, env.hip_roll_ids]
  penalty = actions.abs().sum(dim=1) * z_scale * y_scale
  return penalty * _style_mask(env, style_mask)


def hip_roll_deviation_l1(
  env: "ManagerBasedRlEnv",
  vel_coef: float = 0.005,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """Ultra ``hip_roll_deviation_l1``: hip-roll position deviation + small joint
  velocity term; relaxes with |cmd_yaw| and |cmd_y|. Ported from hist10."""
  _ensure_id_cache(env)
  asset: Entity = env.scene[asset_cfg.name]
  ids = env.hip_roll_ids
  angle = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
  vel = asset.data.joint_vel[:, ids].abs().sum(dim=1)
  z_scale = _cmd_axis_relax_scale(env, 2, "ang_vel_z")
  y_scale = _cmd_axis_relax_scale(env, 1, "lin_vel_y")
  penalty = (
    angle.abs().sum(dim=1) * z_scale * y_scale + vel_coef * vel * z_scale * y_scale
  )
  return penalty * _style_mask(env, style_mask)


def hip_yaw_deviation_l1(
  env: "ManagerBasedRlEnv",
  vel_coef: float = 0.005,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """Ultra ``hip_yaw_deviation_l1``: hip-yaw position deviation + small joint
  velocity term; relaxes with |cmd_yaw|. Ported from hist10."""
  _ensure_id_cache(env)
  asset: Entity = env.scene[asset_cfg.name]
  ids = env.hip_yaw_ids
  angle = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
  vel = asset.data.joint_vel[:, ids].abs().sum(dim=1)
  z_scale = _cmd_axis_relax_scale(env, 2, "ang_vel_z")
  penalty = angle.abs().sum(dim=1) * z_scale + vel_coef * vel * z_scale
  return penalty * _style_mask(env, style_mask)


def dof_vel_pen_l2s(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """Ultra ``dof_vel_pen_l2s``: sum |joint_vel| over the joints selected by
  ``asset_cfg.joint_names``. Ported from hist10."""
  asset: Entity = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  jv = asset.data.joint_vel if ids is None else asset.data.joint_vel[:, ids]
  return jv.abs().sum(dim=-1) * _style_mask(env, style_mask)


def feet_air_time_positive_biped(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  threshold: float = 0.5,
  command_name: str = "twist",
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """Ultra/IsaacLab biped air-time reward: single-stance ``in_mode_time`` capped
  at ``threshold``, zeroed when |cmd_xy| < 0.1. Ported from hist10."""
  sensor = env.scene[sensor_name]
  air_time = sensor.data.current_air_time
  contact_time = sensor.data.current_contact_time
  assert air_time is not None and contact_time is not None
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.sum(in_contact.int(), dim=1) == 1
  reward = torch.min(
    torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
  )[0]
  reward = reward.clamp(max=threshold)
  cmd = env.command_manager.get_command(command_name)
  reward = reward * (torch.norm(cmd[:, :2], dim=1) > 0.1).float()
  return reward * _style_mask(env, style_mask)


def hip_yaw_pos_neg(
  env: "ManagerBasedRlEnv",
  yaw_cmd_relax_at: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """Ultra L1 hip_yaw position penalty; relaxes as |cmd_yaw| grows.

  ``penalty = sum(|q_hip_yaw|) * (1 - clamp(|cmd_yaw| / yaw_cmd_relax_at, 0, 1))``.
  """
  _ensure_id_cache(env)
  asset: Entity = env.scene[asset_cfg.name]
  ids = env.hip_yaw_ids
  penalty = asset.data.joint_pos[:, ids].abs().sum(dim=1)
  cmd_yaw = _twist_cmd(env)[:, 2].abs()
  relax = (cmd_yaw / yaw_cmd_relax_at).clamp(0.0, 1.0)
  return penalty * (1.0 - relax) * _style_mask(env, style_mask)


def hip_yaw_action(
  env: "ManagerBasedRlEnv",
  yaw_cmd_relax_at: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """Ultra L1 hip_yaw action penalty; relaxes as |cmd_yaw| grows."""
  _ensure_id_cache(env)
  am = env.action_manager
  ids = env.hip_yaw_ids
  penalty = am.action[:, ids].abs().sum(dim=1)
  cmd_yaw = _twist_cmd(env)[:, 2].abs()
  relax = (cmd_yaw / yaw_cmd_relax_at).clamp(0.0, 1.0)
  return penalty * (1.0 - relax) * _style_mask(env, style_mask)


# ---------------------------------------------------------------------------
# Per-joint xhumanoid penalties.
# ---------------------------------------------------------------------------


def torques_weighted_neg(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  _ensure_id_cache(env)
  asset: Entity = env.scene[asset_cfg.name]
  w = env._xhumanoid_torque_weights
  cmd_x = _twist_cmd(env)[:, 0].abs()
  # Speed scale: 0.2 * ((m+1) - cmd) where m = lin_vel_x[1] (15 here).
  # Read max from cfg.
  m = _fwd_cmd_max(env)
  speed_scale = 0.2 * ((m + 1.0) - cmd_x)
  penalty = torch.sum(w * asset.data.qfrc_actuator.square(), dim=1) * speed_scale
  return penalty * _style_mask(env, style_mask)


def _ensure_torque_envelope_cache(env: "ManagerBasedRlEnv", asset: "Entity") -> None:
  """Cache per-joint torque-speed envelope tensors (idempotent).

  Mirrors ``UltraDelayedPdActuator._clip_effort``: builds joint-ordered (B, N)
  tensors for y1/y2/x1/x2 and the continuous ``force_limit`` so a reward can
  recompute the *exact* velocity/direction-dependent torque ceiling the actuator
  clips to. Joints without an Ultra-style envelope get a huge ceiling (never
  triggers).
  """
  if getattr(env, "_ultra_tau_env_cached", False):
    return
  n = env.num_envs
  num_joints = len(asset.joint_names)
  dev = env.device
  big = torch.full((n, num_joints), 1.0e9, device=dev)
  y1 = big.clone()
  y2 = big.clone()
  flim = big.clone()
  # Distinct large x1<x2 so the decay branch (never selected for these joints) is
  # finite, not nan.
  x1 = torch.full((n, num_joints), 1.0e6, device=dev)
  x2 = torch.full((n, num_joints), 2.0e6, device=dev)
  for act in asset.actuators:
    tids = torch.as_tensor(act.target_ids, device=dev, dtype=torch.long)
    ey1 = getattr(act, "_effort_y1", None)
    if ey1 is not None:
      y1[:, tids] = ey1
      y2[:, tids] = act._effort_y2  # type: ignore[attr-defined]
      x1[:, tids] = act._velocity_x1  # type: ignore[attr-defined]
      x2[:, tids] = act._velocity_x2  # type: ignore[attr-defined]
    fl = getattr(act, "force_limit", None)
    if fl is not None:
      flim[:, tids] = fl
  env._tau_y1, env._tau_y2 = y1, y2
  env._tau_x1, env._tau_x2 = x1, x2
  env._tau_flim = flim
  env._ultra_tau_env_cached = True


def torque_saturation_neg(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  margin: float = 0.85,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """Squared-overshoot penalty for driving actuators into saturation.

  Reconstructs the same velocity/direction-dependent torque ceiling that
  ``UltraDelayedPdActuator`` clips to, then penalizes how far the *applied*
  torque pushes past ``margin`` * ceiling. Because the applied torque is already
  clipped to the ceiling, this is maximal exactly when a joint is fully
  saturated and zero when comfortably inside the envelope -- a direct,
  speed-aware "stop demanding torque you don't have" signal.
  """
  asset: Entity = env.scene[asset_cfg.name]
  _ensure_torque_envelope_cache(env, asset)
  tau = asset.data.qfrc_actuator
  vel = asset.data.joint_vel
  same_dir = (vel * tau) > 0
  max_eff = torch.where(same_dir, env._tau_y1, env._tau_y2)
  k = -max_eff / (env._tau_x2 - env._tau_x1)
  decay = (k * (vel.abs() - env._tau_x1) + max_eff).clamp_min(0.0)
  max_eff = torch.where(vel.abs() < env._tau_x1, max_eff, decay)
  max_eff = torch.minimum(max_eff, env._tau_flim)
  over = (tau.abs() - margin * max_eff).clamp(min=0.0)
  penalty = torch.sum(over.square(), dim=1)
  return penalty * _style_mask(env, style_mask)


def wholebody_vel_weighted_neg(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  _ensure_id_cache(env)
  asset: Entity = env.scene[asset_cfg.name]
  w = env._xhumanoid_vel_weights
  cmd_x = _twist_cmd(env)[:, 0].abs()
  m = _fwd_cmd_max(env)
  speed_scale = (0.2 * (m - cmd_x)).clamp(min=1.0)
  penalty = torch.sum(w * asset.data.joint_vel.square(), dim=1) * speed_scale
  return penalty * _style_mask(env, style_mask)


def hip_yaw_vel_cmd_neg(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  _ensure_id_cache(env)
  asset: Entity = env.scene[asset_cfg.name]
  ids = env.hip_yaw_ids
  vel_cost = torch.sum(asset.data.joint_vel[:, ids].square(), dim=1)
  m = _yaw_cmd_max(env)
  cmd_ratio = (_twist_cmd(env)[:, 2].abs() / m).clamp(0.0, 1.0)
  cmd_w = 1.0 - 0.8 * cmd_ratio
  return vel_cost * cmd_w * _style_mask(env, style_mask)


def _fwd_cmd_max(env: "ManagerBasedRlEnv") -> float:
  ranges = env.command_manager.get_term("twist").cfg.ranges
  v = float(ranges.lin_vel_x[1])
  return v if v > 1e-3 else 1e-3


def _yaw_cmd_max(env: "ManagerBasedRlEnv") -> float:
  ranges = env.command_manager.get_term("twist").cfg.ranges
  r = ranges.ang_vel_z
  m = max(abs(float(r[0])), abs(float(r[1])))
  return m if m > 1e-3 else 1e-3


# ---------------------------------------------------------------------------
# Stability / posture.
# ---------------------------------------------------------------------------


def _base_height_above_terrain(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg,
  terrain_height_sensor: str | None,
) -> torch.Tensor:
  """Base height to use for the height reward / observation.

  With ``terrain_height_sensor`` set, returns the base height *above the terrain
  directly below it* (``root_z - terrain_z``) read from a single down-ray
  ``TerrainHeightSensor`` on the base. Without it, falls back to the absolute
  world-frame z (correct only on a flat plane).
  """
  if terrain_height_sensor is not None:
    # TerrainHeightSensor.data.heights is [B, F] = frame_z - hit_z. The base
    # sensor has a single frame (F=1), so column 0 is the base clearance.
    return env.scene[terrain_height_sensor].data.heights[:, 0]
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2]


def base_height_neg(
  env: "ManagerBasedRlEnv",
  target_height: float = 1.20,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
  terrain_height_sensor: str | None = None,
) -> torch.Tensor:
  # Only penalize the base for being *below* the target, clamped to [-0.5, 0]
  # and applied linearly (matching ultra_run_lab's `base_height_neg`).
  #
  # With ``terrain_height_sensor`` set (V11/V12/V13 on terrain), the height is
  # measured *relative to the terrain under the base*, so slopes / rough tiles
  # don't bias it. Without the sensor (flat tasks) it uses absolute world z;
  # that is already terrain-safe because the one-sided clamp ignores the base
  # rising onto raised ground -- but on a slope it would still mis-penalize the
  # downhill case, which is exactly what the sensor removes.
  h = _base_height_above_terrain(env, asset_cfg, terrain_height_sensor)
  height_error = (h - target_height).clamp(min=-0.5, max=0.0)
  return height_error.abs() * _style_mask(env, style_mask)


def base_height_priv(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  target_height: float = 1.18,
) -> torch.Tensor:
  """Privileged critic observation: terrain-relative base height error.

  Returns ``clip(root_z - terrain_z - target_height, -0.5, 0.5)`` from the base
  down-ray ``TerrainHeightSensor``, matching ultra_run_lab's critic
  ``base_height`` term. Shape ``[B, 1]``.
  """
  h = env.scene[sensor_name].data.heights[:, 0:1]
  return torch.clip(h - target_height, -0.5, 0.5)


def lin_vel_z_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
  max_penalty: float | None = None,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  penalty = _vel_yaw_frame(asset)[:, 2].square()
  penalty = _clip_penalty(penalty, max_penalty)
  return penalty * _style_mask(env, style_mask)


def ang_vel_xy_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(_ang_vel_world(asset)[:, :2].square(), dim=1) * _style_mask(
    env, style_mask
  )


def body_orientation_exp(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """exp(-100 * |proj_g_xy|^2) on a chosen body's projected gravity."""
  _ensure_id_cache(env)
  asset: Entity = env.scene[asset_cfg.name]
  body_id = (
    asset_cfg.body_ids[0] if asset_cfg.body_ids is not None else env.waist_body_id
  )
  bq = asset.data.body_link_quat_w[:, body_id, :]
  g = asset.data.gravity_vec_w
  proj = quat_apply_inverse(bq, g)
  err = proj[:, 0].square() + proj[:, 1].square()
  return torch.exp(-100.0 * err) * _style_mask(env, style_mask)


def body_orientation_speed_aware(
  env: "ManagerBasedRlEnv",
  pitch_coef: float = 0.025,
  roll_coef: float = 0.07,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """High-speed-aware body orientation tracking.

  Allows forward lean proportional to ``cmd_x``, side roll proportional to
  ``|cmd_x| * cmd_yaw``.
  """
  _ensure_id_cache(env)
  asset: Entity = env.scene[asset_cfg.name]
  body_id = (
    asset_cfg.body_ids[0] if asset_cfg.body_ids is not None else env.base_body_id
  )
  bq = asset.data.body_link_quat_w[:, body_id, :]
  g = asset.data.gravity_vec_w
  proj_g = quat_apply_inverse(bq, g)
  cmd = _twist_cmd(env)
  cmd_x = cmd[:, 0].clamp(min=0.0)
  cmd_yaw = cmd[:, 2]
  pitch_target = pitch_coef * cmd_x
  roll_target = roll_coef * cmd_x.abs() * cmd_yaw
  err = (pitch_target - proj_g[:, 0]).square() + (roll_target - proj_g[:, 1]).square()
  return torch.exp(-100.0 * err) * _style_mask(env, style_mask)


# ---------------------------------------------------------------------------
# Contact / feet.
# ---------------------------------------------------------------------------


def _foot_force(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
  """Per-foot contact force in world frame, shape (B, 2, 3)."""
  cs: ContactSensor = env.scene[sensor_name]
  assert cs.data.force is not None, "Sensor must have 'force' field configured."
  # ContactData.force shape: (B, N, 3) where N = num_primaries * num_slots.
  # Configured with num_slots=1 -> N = num_primaries (= 2 feet).
  return cs.data.force


def _foot_force_history(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
  """Per-foot contact force history, shape (B, N, H, 3)."""
  cs: ContactSensor = env.scene[sensor_name]
  assert cs.data.force_history is not None, (
    "Sensor must have 'force_history' field configured."
  )
  return cs.data.force_history


def body_contact_neg(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  threshold: float = 1.0,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """Penalize contact on configured bodies of ``sensor_name`` (history-aware)."""
  cs: ContactSensor = env.scene[sensor_name]
  if cs.data.force_history is not None:
    forces = cs.data.force_history  # (B, N, H, 3)
    # Max over history of force magnitude.
    mag = torch.norm(forces, dim=-1)  # (B, N, H)
    is_contact = mag.max(dim=-1).values > threshold  # (B, N)
  else:
    assert cs.data.force is not None
    is_contact = torch.norm(cs.data.force, dim=-1) > threshold
  penalty = is_contact.float().sum(dim=1)
  return penalty * _style_mask(env, style_mask)


def feet_slide(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  min_contact_fz: float = 1.0,
  slip_speed_threshold: float = 0.0,
  only_lateral_yaw: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """Penalize foot xy speed in body yaw frame while in contact."""
  _ensure_id_cache(env)
  asset: Entity = env.scene[asset_cfg.name]
  forces = _foot_force(env, sensor_name)  # (B, 2, 3)
  in_contact = forces[..., 2] > min_contact_fz  # (B, 2)

  feet_ids = env.feet_body_ids
  foot_vel_w = asset.data.body_link_lin_vel_w[:, feet_ids, :]  # (B, 2, 3)
  yq = yaw_quat(asset.data.root_link_quat_w)  # (B, 4)
  yq_b = yq[:, None, :].expand(-1, foot_vel_w.shape[1], -1)
  foot_vel_yaw = quat_apply_inverse(yq_b, foot_vel_w)  # (B, 2, 3)

  if only_lateral_yaw:
    speed = foot_vel_yaw[..., 1].abs()
  else:
    speed = torch.norm(foot_vel_yaw[..., :2], dim=-1)

  speed = (speed - slip_speed_threshold).clamp(min=0.0)
  cost = (speed * in_contact.float()).sum(dim=1)
  return cost * _style_mask(env, style_mask)


def contact_impact_vel(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """Penalize foot speed at touchdown (rising contact edge)."""
  _ensure_id_cache(env)
  asset: Entity = env.scene[asset_cfg.name]
  fh = _foot_force_history(env, sensor_name)  # (B, N, H, 3)
  fz = fh[..., 2]  # (B, N, H)
  contact = fz > 1.0
  if contact.shape[-1] >= 2:
    rising = contact[..., 0] & (~contact[..., 1])  # index 0 = most recent
  else:
    rising = contact[..., 0]
  feet_ids = env.feet_body_ids
  foot_spd = torch.norm(asset.data.body_link_lin_vel_w[:, feet_ids, :], dim=-1)
  score = (rising.float() * foot_spd).sum(dim=1)
  cmd_x = _twist_cmd(env)[:, 0]
  vmax = max(_fwd_cmd_max(env), 1e-3)
  cmd_w = 0.5 + (cmd_x - 3.0).clamp(min=0.0, max=vmax) / vmax
  return score * cmd_w * _style_mask(env, style_mask)


def feet_stumble(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  ratio: float = 5.0,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """Detect feet stumble: lateral force >> vertical force."""
  forces = _foot_force(env, sensor_name)  # (B, 2, 3)
  lat = torch.norm(forces[..., :2], dim=-1)
  vert = forces[..., 2].abs() + 1e-6
  stumble = (lat > ratio * vert).float().sum(dim=1)
  return stumble * _style_mask(env, style_mask)


def gait_feet_force_max_neg(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  max_force: float = 1500.0,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  """Penalize per-foot vertical contact forces above ``max_force``."""
  forces = _foot_force(env, sensor_name)  # (B, 2, 3)
  fz = forces[..., 2]
  extra = (fz - max_force).clamp(min=0.0, max=100.0)
  return extra.sum(dim=1) * _style_mask(env, style_mask)


# ---------------------------------------------------------------------------
# Spacing.
# ---------------------------------------------------------------------------


def _foot_pos_in_body(asset: Entity, feet_ids: torch.Tensor) -> torch.Tensor:
  """Foot positions expressed in root_link frame, shape (B, 2, 3)."""
  foot_w = asset.data.body_link_pos_w[:, feet_ids, :]  # (B, 2, 3)
  root_w = asset.data.root_link_pos_w[:, None, :]  # (B, 1, 3)
  rel_w = foot_w - root_w
  rq = asset.data.root_link_quat_w
  rq_b = rq[:, None, :].expand(-1, feet_ids.numel(), -1)
  return quat_apply_inverse(rq_b, rel_w)


def gait_feet_distance(
  env: "ManagerBasedRlEnv",
  target_y: float = 0.15,
  lateral_cmd_relax_at: float = 0.5,
  speed_scale: float = 1.0,
  speed_max: float = 15.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  _ensure_id_cache(env)
  asset: Entity = env.scene[asset_cfg.name]
  fp = _foot_pos_in_body(asset, env.feet_body_ids)  # (B, 2, 3)
  lateral_span = fp[:, 0, 1] - fp[:, 1, 1]
  penalty = (lateral_span - 2.0 * target_y).abs()
  cmd_vy = _twist_cmd(env)[:, 1].abs()
  relax = (cmd_vy / lateral_cmd_relax_at).clamp(0.0, 1.0)
  penalty = penalty * (1.0 - relax)
  cmd_vx = _twist_cmd(env)[:, 0].abs().clamp(max=speed_max)
  penalty = penalty * speed_scale * cmd_vx
  return penalty * _style_mask(env, style_mask)


def knee_y_distance(
  env: "ManagerBasedRlEnv",
  target_y: float = 0.15,
  speed_scale: float = 1.0,
  speed_max: float = 15.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  _ensure_id_cache(env)
  asset: Entity = env.scene[asset_cfg.name]
  kp = _foot_pos_in_body(asset, env.knee_body_ids)
  score_left = 2.0 * (target_y - kp[:, 0, 1]).abs()
  score_right = 2.0 * (target_y + kp[:, 1, 1]).abs()
  penalty = score_left + score_right
  cmd_vx = _twist_cmd(env)[:, 0].abs().clamp(max=speed_max)
  penalty = penalty * speed_scale * cmd_vx
  cmd_scale = 0.5 - _twist_cmd(env)[:, 1].abs().clamp(max=0.5)
  return penalty * cmd_scale * _style_mask(env, style_mask)


def feet_collision(
  env: "ManagerBasedRlEnv",
  threshold_y: float = 0.14,
  threshold_x: float = 0.24,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  style_mask: list[int] | None = None,
) -> torch.Tensor:
  _ensure_id_cache(env)
  asset: Entity = env.scene[asset_cfg.name]
  fp = _foot_pos_in_body(asset, env.feet_body_ids)
  dy = (fp[:, 0, 1] - fp[:, 1, 1]).abs()
  dx = (fp[:, 0, 0] - fp[:, 1, 0]).abs()
  collision = ((dy < threshold_y) & (dx < threshold_x)).float()
  return collision * _style_mask(env, style_mask)


# ---------------------------------------------------------------------------
# Privileged critic observations.
# ---------------------------------------------------------------------------


def root_lin_vel_yaw(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Linear velocity in yaw-only base frame (B, 3)."""
  asset: Entity = env.scene[asset_cfg.name]
  return _vel_yaw_frame(asset)


def feet_vel_z(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Vertical velocity of each foot (B, 2)."""
  _ensure_id_cache(env)
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.body_link_lin_vel_w[:, env.feet_body_ids, 2]


def foot_force_z(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
) -> torch.Tensor:
  """Per-foot vertical contact force (B, 2). Scaled by 1/200 to ~unit range."""
  forces = _foot_force(env, sensor_name)
  return forces[..., 2] / 200.0


def feet_contact_state(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  threshold: float = 1.0,
) -> torch.Tensor:
  """Binary per-foot contact state (B, 2)."""
  forces = _foot_force(env, sensor_name)
  return (forces[..., 2] > threshold).float()


def foot_clearance_priv(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Per-foot height above terrain (B, 2). Approximated with absolute z."""
  _ensure_id_cache(env)
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.body_link_pos_w[:, env.feet_body_ids, 2]


def domain_rand_features(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  foot_geom_names: tuple[str, ...] = ("left_foot_collision", "right_foot_collision"),
) -> torch.Tensor:
  """Privileged domain-rand features mirroring Ultra's critic input (B, 41).

  Layout: foot_friction(2) + kp_scale(13) + kd_scale(13) + motor_strength(13).
  """
  asset: Entity = env.scene[asset_cfg.name]
  n = env.num_envs

  # foot friction: resolve geom ids once via SceneEntityCfg, then read sim model.
  if not hasattr(env, "_dr_foot_geom_ids"):
    foot_cfg = SceneEntityCfg(asset_cfg.name, geom_names=list(foot_geom_names))
    foot_cfg.resolve(env.scene)
    env._dr_foot_geom_ids = asset.indexing.geom_ids[foot_cfg.geom_ids]  # type: ignore[attr-defined]
  geom_ids = env._dr_foot_geom_ids  # (2,)
  foot_friction = env.sim.model.geom_friction[:, geom_ids, 0]  # (n_envs, 2)

  # kp / kd scale and motor strength. mjlab keeps per-env, per-joint gains and
  # the motor-strength DR scale on the PD ACTUATOR objects (Ideal/UltraDelayed),
  # not on ``asset.data``. ``asset.actuators`` is a list; assemble joint-ordered
  # (B, num_joints) tensors by scattering each actuator's targets into their
  # entity joint slots. Joints with no PD actuator stay at the nominal scale 1.0.
  num_joints = len(asset.joint_names)
  kp_scale = torch.ones(n, num_joints, device=env.device)
  kd_scale = torch.ones(n, num_joints, device=env.device)
  motor_strength = torch.ones(n, num_joints, device=env.device)
  for act in asset.actuators:
    tids = torch.as_tensor(act.target_ids, device=env.device, dtype=torch.long)
    stiffness = getattr(act, "stiffness", None)
    default_stiffness = getattr(act, "default_stiffness", None)
    if stiffness is not None and default_stiffness is not None:
      kp_scale[:, tids] = stiffness / default_stiffness.clamp(min=1e-6)
    damping = getattr(act, "damping", None)
    default_damping = getattr(act, "default_damping", None)
    if damping is not None and default_damping is not None:
      kd_scale[:, tids] = damping / default_damping.clamp(min=1e-6)
    ms = getattr(act, "motor_strength", None)
    if ms is not None:
      motor_strength[:, tids] = ms

  return torch.cat([foot_friction, kp_scale, kd_scale, motor_strength], dim=-1)


# ---------------------------------------------------------------------------
# Symmetry data augmentation.
# ---------------------------------------------------------------------------

# mjlab Ultra actor obs layout (see env_cfgs.py "actor" group, Ultra ordering):
#   ang_vel(3) + grav(3) + cmd(3) + joint_pos(13) + joint_vel(13) + actions(13)
#   = 48 dims.
#
# Critic obs layout (Ultra ordering with privileged extras):
#   ang_vel(3) + grav(3) + cmd(3) + joint_pos(13) + joint_vel(13) + actions(13)
#   + base_lin_vel_yaw(3) + feet_vel_z(2) + foot_force_z(2) + contact(2)
#   + foot_clearance(2) = 60 dims.

# Joint name -> mjlab joint index resolved lazily and cached on env.
_LEFT_LEG = (
  "hip_yaw_l_joint",
  "hip_roll_l_joint",
  "hip_pitch_l_joint",
  "knee_pitch_l_joint",
  "ankle_pitch_l_joint",
)
_RIGHT_LEG = (
  "hip_yaw_r_joint",
  "hip_roll_r_joint",
  "hip_pitch_r_joint",
  "knee_pitch_r_joint",
  "ankle_pitch_r_joint",
)
_LEFT_ARM = ("shoulder_pitch_l_joint",)
_RIGHT_ARM = ("shoulder_pitch_r_joint",)
# Joints that flip sign under mirror (yaw + roll axes).
_MIRROR_NEGATE = (
  "hip_yaw_l_joint",
  "hip_yaw_r_joint",
  "hip_roll_l_joint",
  "hip_roll_r_joint",
  "waist_yaw_joint",
)


def _mirror_init(env: "ManagerBasedRlEnv") -> None:
  """Cache symmetry permutation + sign tensors on env."""
  if getattr(env, "_mirror_cached", False):
    return
  _ensure_id_cache(env)
  robot: Entity = env.scene["robot"]
  joint_names = robot.joint_names
  n = len(joint_names)
  name_to_idx = {nm: i for i, nm in enumerate(joint_names)}

  perm = list(range(n))
  for l_n, r_n in zip(_LEFT_LEG, _RIGHT_LEG, strict=True):
    li, ri = name_to_idx[l_n], name_to_idx[r_n]
    perm[li], perm[ri] = ri, li
  for l_n, r_n in zip(_LEFT_ARM, _RIGHT_ARM, strict=True):
    li, ri = name_to_idx[l_n], name_to_idx[r_n]
    perm[li], perm[ri] = ri, li

  signs = torch.ones(n, device=env.device, dtype=torch.float32)
  for nm in _MIRROR_NEGATE:
    if nm in name_to_idx:
      signs[name_to_idx[nm]] = -1.0

  env._mirror_perm = torch.tensor(perm, device=env.device, dtype=torch.long)
  # Apply perm first then signs: mirrored = sign[perm] * x[:, perm]
  env._mirror_signs_post_perm = signs[env._mirror_perm].clone()

  # Actor-obs sub-permutation: excludes passive joints absent from the obs
  # (e.g. ankle_roll added in V10). For models where all joints are actuated
  # this is identical to the full permutation, so it is always computed.
  passive = {nm for nm in joint_names if "ankle_roll" in nm}
  act_names = [nm for nm in joint_names if nm not in passive]
  model_to_act = {name_to_idx[nm]: k for k, nm in enumerate(act_names)}
  sub_perm = [model_to_act[perm[name_to_idx[nm]]] for nm in act_names]
  sub_signs = [float(signs[perm[name_to_idx[nm]]]) for nm in act_names]
  env._mirror_perm_actor = torch.tensor(sub_perm, device=env.device, dtype=torch.long)
  env._mirror_signs_actor = torch.tensor(
    sub_signs, device=env.device, dtype=torch.float32
  )

  env._mirror_cached = True


def _mirror_joint_vector(x: torch.Tensor, env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Apply joint mirror to a (B, n) tensor in mjlab joint order."""
  return x[:, env._mirror_perm] * env._mirror_signs_post_perm


def _mirror_actor_obs(obs: torch.Tensor, env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Mirror a (B, 9 + 3n) actor obs tensor in Ultra layout."""
  n_full = int(env._mirror_perm.numel())
  n = (obs.shape[1] - 9) // 3
  # Use actor sub-perm when obs has fewer joints than the full model
  # (e.g. V10 has 15 model joints but only 13 in the obs).
  if n != n_full and hasattr(env, "_mirror_perm_actor"):
    perm = env._mirror_perm_actor
    signs = env._mirror_signs_actor
  else:
    perm = env._mirror_perm
    signs = env._mirror_signs_post_perm
  ang_vel = obs[:, 0:3].clone()
  grav = obs[:, 3:6].clone()
  cmd = obs[:, 6:9].clone()
  jpos = obs[:, 9 : 9 + n]
  jvel = obs[:, 9 + n : 9 + 2 * n]
  act = obs[:, 9 + 2 * n : 9 + 3 * n]

  ang_vel[:, 0] *= -1.0
  ang_vel[:, 2] *= -1.0
  grav[:, 1] *= -1.0
  cmd[:, 1] *= -1.0
  cmd[:, 2] *= -1.0

  jpos = jpos[:, perm] * signs
  jvel = jvel[:, perm] * signs
  act = act[:, perm] * signs
  return torch.cat([ang_vel, grav, cmd, jpos, jvel, act], dim=-1)


def _mirror_critic_obs(obs: torch.Tensor, env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Mirror Ultra-aligned critic obs.

  Layout:
    base_lin_vel_yaw(3) + ang_vel(3) + grav(3) + cmd(3) + jpos(n) + jvel(n)
    + actions(n) + feet_vel_z(2) + foot_force_z(2) + contact(2)
    + foot_clearance(2).
  """
  # Use the actor sub-perm size for joint blocks: if passive joints are
  # excluded from the critic joint obs (V10+), n_actor < n_full.
  if hasattr(env, "_mirror_perm_actor"):
    n = int(env._mirror_perm_actor.numel())

    def _mirror_jnt(x: torch.Tensor) -> torch.Tensor:
      return x[:, env._mirror_perm_actor] * env._mirror_signs_actor

  else:
    n = int(env._mirror_perm.numel())

    def _mirror_jnt(x: torch.Tensor) -> torch.Tensor:
      return _mirror_joint_vector(x, env)

  out = obs.clone()
  # base_lin_vel: flip y (yaw-frame).
  out[:, 1] *= -1.0
  # ang_vel: flip x, z.
  out[:, 3] *= -1.0
  out[:, 5] *= -1.0
  # grav: flip y.
  out[:, 7] *= -1.0
  # cmd: flip y, yaw.
  out[:, 10] *= -1.0
  out[:, 11] *= -1.0
  # jpos / jvel / actions.
  off = 12
  for _ in range(3):
    block = out[:, off : off + n]
    out[:, off : off + n] = _mirror_jnt(block)
    off += n
  # Privileged left/right pairs: swap each (B, 2) block.
  while off + 2 <= out.shape[1]:
    a = out[:, off].clone()
    b = out[:, off + 1].clone()
    out[:, off] = b
    out[:, off + 1] = a
    off += 2
  return out


def _mirror_actions(actions: torch.Tensor, env: "ManagerBasedRlEnv") -> torch.Tensor:
  if hasattr(env, "_mirror_perm_actor"):
    return actions[:, env._mirror_perm_actor] * env._mirror_signs_actor
  return _mirror_joint_vector(actions, env)


def ultra_data_augmentation_func(
  obs: torch.Tensor | None = None,
  actions: torch.Tensor | None = None,
  env=None,
  obs_type: str = "policy",
):
  """Symmetry hook for ``MultiAmpOnPolicyRunner`` (mjlab Ultra layout).

  Returns a tuple ``(obs_aug, actions_aug)`` doubled along dim 0:
  the first half is the original input, the second half is the mirrored copy.
  Either ``obs`` or ``actions`` may be ``None``; mirrored output for ``None``
  inputs is also ``None``.
  """
  assert env is not None, "ultra_data_augmentation_func requires env."
  unwrapped = getattr(env, "unwrapped", env)
  _mirror_init(unwrapped)

  if obs is not None:
    # Detect history-flattened actor obs by checking divisibility against the
    # expected single-frame size; mirror each frame, preserve overall shape.
    one_step = 9 + 3 * int(
      unwrapped._mirror_perm_actor.numel()
      if hasattr(unwrapped, "_mirror_perm_actor")
      else unwrapped._mirror_perm.numel()
    )  # 48 for Ultra
    if obs_type == "critic":
      critic_step = getattr(env, "num_one_step_critic_obs", obs.shape[1])
      if (
        obs.dim() == 2
        and critic_step is not None
        and obs.shape[1] > critic_step
        and obs.shape[1] % critic_step == 0
      ):
        b = obs.shape[0]
        t = obs.shape[1] // critic_step
        flat = obs.reshape(b * t, critic_step)
        flat = _mirror_critic_obs(flat, unwrapped)
        mirrored = flat.reshape(b, t * critic_step)
      else:
        mirrored = _mirror_critic_obs(obs, unwrapped)
    elif obs.dim() == 3:
      b, t, d = obs.shape
      flat = obs.reshape(b * t, d)
      flat = _mirror_actor_obs(flat, unwrapped)
      mirrored = flat.reshape(b, t, d)
    elif obs.dim() == 2 and obs.shape[1] != one_step and obs.shape[1] % one_step == 0:
      # History-flattened (B, T*one_step).
      b = obs.shape[0]
      t = obs.shape[1] // one_step
      flat = obs.reshape(b * t, one_step)
      flat = _mirror_actor_obs(flat, unwrapped)
      mirrored = flat.reshape(b, t * one_step)
    else:
      mirrored = _mirror_actor_obs(obs, unwrapped)
    obs_out = torch.cat([obs, mirrored], dim=0)
  else:
    obs_out = None

  if actions is not None:
    if obs_type == "critic":
      # In critic mode the runner passes ``next_critic_obs_batch`` as the
      # ``actions`` argument — mirror it as a critic obs, not as actions.
      critic_step = getattr(env, "num_one_step_critic_obs", actions.shape[1])
      if (
        actions.dim() == 2
        and critic_step is not None
        and actions.shape[1] > critic_step
        and actions.shape[1] % critic_step == 0
      ):
        b = actions.shape[0]
        t = actions.shape[1] // critic_step
        flat = actions.reshape(b * t, critic_step)
        flat = _mirror_critic_obs(flat, unwrapped)
        mirrored_a = flat.reshape(b, t * critic_step)
      else:
        mirrored_a = _mirror_critic_obs(actions, unwrapped)
    else:
      mirrored_a = _mirror_actions(actions, unwrapped)
    actions_out = torch.cat([actions, mirrored_a], dim=0)
  else:
    actions_out = None

  return obs_out, actions_out
