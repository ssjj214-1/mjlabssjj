from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  wrap_to_pi,
)

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)
    # Rate-limited command support (ultra_run_lab-style slew). ``cmd_target_b``
    # holds the sampled target; ``vel_command_b`` ramps toward it at
    # ``cfg.max_lin_accel`` when the ramp is enabled. No-op when max_lin_accel<=0.
    self.cmd_target_b = torch.zeros(self.num_envs, 3, device=self.device)
    # Bucket assignment (0=stand, 1=walk, 2=run) for hist10 bucket sampling.
    self.command_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)
    self.is_world_env = torch.zeros_like(self.is_heading_env)
    self.is_forward_env = torch.zeros_like(self.is_heading_env)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.rel_walk_envs > 0.0:
      self._resample_bucket_command(env_ids)
      return

    ramp_on = self.cfg.max_lin_accel > 0.0
    prev_cmd = self.vel_command_b[env_ids].clone() if ramp_on else None
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    # High-speed coverage: bias run-style envs toward the top speed band so the
    # curriculum cap (up to 15 m/s) is actually sampled.
    self._resample_high_speed(env_ids)

    # Randomly assign world-frame envs.
    self.is_world_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_world_envs
    # Copy sampled velocities as world-frame reference for world envs.
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

    # Forward-only envs: positive lin_vel_x, zero lateral and angular.
    self.is_forward_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs
    fwd_ids = env_ids[self.is_forward_env[env_ids]]
    if len(fwd_ids) > 0:
      self.vel_command_b[fwd_ids, 0] = (
        self.vel_command_b[fwd_ids, 0].abs().clamp(min=0.3)
      )
      self.vel_command_b[fwd_ids, 1] = 0.0
      self.vel_command_b[fwd_ids, 2] = 0.0

    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_vel_env_ids = env_ids[init_vel_mask]
    if len(init_vel_env_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
      lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

    # Command ramp (slew): record the just-sampled command as the target. A
    # ``1 - cmd_step_sample_frac`` fraction of resampled envs hold their previous
    # command and ramp toward the new target (the rest step to it immediately).
    if ramp_on:
      self.cmd_target_b[env_ids] = self.vel_command_b[env_ids].clone()
      step_frac = self.cfg.cmd_step_sample_frac
      if step_frac < 1.0 and prev_cmd is not None:
        ramp_mask = torch.rand(len(env_ids), device=self.device) >= step_frac
        if torch.any(ramp_mask):
          self.vel_command_b[env_ids[ramp_mask]] = prev_cmd[ramp_mask]

  def _resample_high_speed(
    self, env_ids: torch.Tensor, force_run: bool = False
  ) -> None:
    """Resample a fraction of run-style envs into the top speed band.

    Run-style envs (freshly sampled ``|lin_vel_x| >= run_high_speed_enter``) are
    forced non-standing; a ``run_high_speed_sample_frac`` fraction of them get a
    new ``lin_vel_x`` drawn uniformly from ``[threshold_frac * vmax, vmax]``.
    No-op when ``run_high_speed_sample_frac <= 0``. With ``force_run`` (bucket
    sampling, where the run bucket already guarantees a high ``lin_vel_x``) all
    ``env_ids`` are treated as run envs without the ``run_high_speed_enter``
    filter, matching hist10.
    """
    frac = self.cfg.run_high_speed_sample_frac
    if frac <= 0.0:
      return
    vmax = float(self.cfg.ranges.lin_vel_x[1])
    if vmax <= 0.0:
      return
    if force_run:
      run_ids = env_ids
    else:
      run_mask = self.vel_command_b[env_ids, 0].abs() >= self.cfg.run_high_speed_enter
      if not torch.any(run_mask):
        return
      run_ids = env_ids[run_mask]
    self.is_standing_env[run_ids] = False
    v_high = self.cfg.run_high_speed_threshold_frac * vmax
    if v_high >= vmax:
      self.vel_command_b[run_ids, 0] = vmax
      return
    rand = torch.rand(len(run_ids), device=self.device)
    high_ids = run_ids[rand < frac]
    if len(high_ids) > 0:
      self.vel_command_b[high_ids, 0] = torch.empty(
        len(high_ids), device=self.device
      ).uniform_(v_high, vmax)

  def _sample_heading_bucket(self, ids: torch.Tensor) -> None:
    if not self.cfg.heading_command:
      return
    assert self.cfg.ranges.heading is not None
    r = torch.empty(len(ids), device=self.device)
    self.heading_target[ids] = r.uniform_(*self.cfg.ranges.heading)
    self.is_heading_env[ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

  def _sample_run_velocity(self, ids: torch.Tensor) -> None:
    """Run bucket: lin_vel_x in [run_lin_vel_x_min, vmax]; vy/yaw from ranges."""
    vmax = float(self.cfg.ranges.lin_vel_x[1])
    vx_lo = float(self.cfg.run_lin_vel_x_min or 0.0)
    vx_hi = max(vmax, vx_lo)
    r = torch.empty(len(ids), device=self.device)
    self.vel_command_b[ids, 0] = r.uniform_(vx_lo, vx_hi)
    self.vel_command_b[ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    self._sample_heading_bucket(ids)

  def _apply_stand_enforce(self, ids: torch.Tensor) -> None:
    """Walk/run bucket: force standing+zero when the *target* speed norm is below
    the stand-enter threshold (mirrors hist10 ``_apply_stand_enforce``)."""
    if len(ids) == 0:
      return
    thr = self.cfg.stand_enter_speed
    low = torch.linalg.norm(self.cmd_target_b[ids, :3], dim=1) < thr
    if not torch.any(low):
      return
    low_ids = ids[low]
    self.is_standing_env[low_ids] = True
    self.vel_command_b[low_ids] = 0.0
    self.cmd_target_b[low_ids] = 0.0

  def _apply_stand_enforce_runtime(self, ids: torch.Tensor) -> None:
    """Runtime stand-enforce (bucket mode): non-heading envs use the (ramping)
    target speed, heading envs the live command (yaw set by the control law)."""
    if len(ids) == 0:
      return
    thr = self.cfg.stand_enter_speed
    heading = self.is_heading_env[ids]
    speed = torch.zeros(len(ids), device=self.device)
    non_heading = ~heading
    if torch.any(non_heading):
      speed[non_heading] = torch.linalg.norm(
        self.cmd_target_b[ids[non_heading], :3], dim=1
      )
    if torch.any(heading):
      speed[heading] = torch.linalg.norm(self.vel_command_b[ids[heading], :3], dim=1)
    low = speed < thr
    if not torch.any(low):
      return
    low_ids = ids[low]
    self.is_standing_env[low_ids] = True
    self.vel_command_b[low_ids] = 0.0
    self.cmd_target_b[low_ids] = 0.0

  def _resample_bucket_command(self, env_ids: torch.Tensor) -> None:
    """hist10 stand/walk/run bucket sampling.

    Each env is assigned to a bucket by a single uniform draw: stand
    (``rel_standing_envs``), walk (``rel_walk_envs``), run (remainder). Walk envs
    draw ``lin_vel_x`` from ``walk_lin_vel_x``; run envs from
    ``[run_lin_vel_x_min, vmax]`` then through high-speed resampling. The command
    ramp (``cmd_step_sample_frac`` / ``max_lin_accel``) is applied to non-stand
    envs exactly as in the single-range path.
    """
    prev_cmd = self.vel_command_b[env_ids].clone()
    n = len(env_ids)
    p_stand = self.cfg.rel_standing_envs
    p_walk = self.cfg.rel_walk_envs
    walk_vx = self.cfg.walk_lin_vel_x or (0.15, 0.95)

    r = torch.rand(n, device=self.device)
    stand_m = r < p_stand
    walk_m = (r >= p_stand) & (r < p_stand + p_walk)
    run_m = r >= p_stand + p_walk
    self.command_mode[env_ids[stand_m]] = 0
    self.command_mode[env_ids[walk_m]] = 1
    self.command_mode[env_ids[run_m]] = 2

    stand_ids = env_ids[stand_m]
    if len(stand_ids) > 0:
      self.is_standing_env[stand_ids] = True
      self.vel_command_b[stand_ids] = 0.0
      self.cmd_target_b[stand_ids] = 0.0

    walk_ids = env_ids[walk_m]
    if len(walk_ids) > 0:
      self.is_standing_env[walk_ids] = False
      rr = torch.empty(len(walk_ids), device=self.device)
      self.vel_command_b[walk_ids, 0] = rr.uniform_(*walk_vx)
      self.vel_command_b[walk_ids, 1] = rr.uniform_(*self.cfg.ranges.lin_vel_y)
      self.vel_command_b[walk_ids, 2] = rr.uniform_(*self.cfg.ranges.ang_vel_z)
      self._sample_heading_bucket(walk_ids)

    run_ids = env_ids[run_m]
    if len(run_ids) > 0:
      self.is_standing_env[run_ids] = False
      self._sample_run_velocity(run_ids)
      self._resample_high_speed(run_ids, force_run=True)

    # World-frame reference (kept for parity with the single-range path).
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

    non_stand_ids = env_ids[~stand_m]
    self.cmd_target_b[env_ids] = self.vel_command_b[env_ids].clone()
    self._apply_stand_enforce(non_stand_ids)

    # Step vs ramp: a fraction of non-stand envs hold their previous command and
    # ramp toward the new target under max_lin_accel.
    step_frac = self.cfg.cmd_step_sample_frac
    if step_frac < 1.0:
      ramp_mask = (torch.rand(n, device=self.device) >= step_frac) & ~stand_m
      if torch.any(ramp_mask):
        self.vel_command_b[env_ids[ramp_mask]] = prev_cmd[ramp_mask]

  def _apply_cmd_accel_limit(self) -> None:
    """Move ``vel_command_b`` toward ``cmd_target_b`` at ``max_lin_accel``.

    Linear xy is always rate-limited; the angular (yaw) command is rate-limited
    only for non-heading envs (heading envs derive yaw from a control law in
    :meth:`_update_command`, so it is already smooth).
    """
    max_dv = self.cfg.max_lin_accel * self._env.step_dt
    for dim in (0, 1):
      delta = self.cmd_target_b[:, dim] - self.vel_command_b[:, dim]
      self.vel_command_b[:, dim] += torch.clamp(delta, -max_dv, max_dv)
    non_heading = ~self.is_heading_env
    if torch.any(non_heading):
      delta_z = self.cmd_target_b[:, 2] - self.vel_command_b[:, 2]
      self.vel_command_b[non_heading, 2] += torch.clamp(
        delta_z[non_heading], -max_dv, max_dv
      )

  def _update_command(self) -> None:
    if self.cfg.max_lin_accel > 0.0:
      self._apply_cmd_accel_limit()
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    # World-frame envs: rotate world-frame linear vel into body frame.
    if self.is_world_env.any():
      w_ids = self.is_world_env.nonzero(as_tuple=False).flatten()
      heading = self.robot.data.heading_w[w_ids]
      cos_h = torch.cos(heading)
      sin_h = torch.sin(heading)
      vx_w = self.vel_command_w[w_ids, 0]
      vy_w = self.vel_command_w[w_ids, 1]
      self.vel_command_b[w_ids, 0] = cos_h * vx_w + sin_h * vy_w
      self.vel_command_b[w_ids, 1] = -sin_h * vx_w + cos_h * vy_w

    if self.cfg.rel_walk_envs > 0.0:
      # Keep the stand bucket pinned at zero and re-check walk/run buckets that
      # may have ramped/heading-controlled below the stand threshold.
      self.is_standing_env[self.command_mode == 0] = True
      active = (
        (self.command_mode == 1) | (self.command_mode == 2)
      ) & ~self.is_standing_env
      if torch.any(active):
        self._apply_stand_enforce_runtime(active.nonzero(as_tuple=False).flatten())

    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0
    self.vel_command_w[standing_env_ids, :] = 0.0
    if self.cfg.max_lin_accel > 0.0:
      self.cmd_target_b[standing_env_ids, :] = 0.0

  # GUI.

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create velocity joystick sliders in the Viser viewer."""
    from viser import Icon

    ranges = self.cfg.ranges

    axes = [
      ("lin_vel_x", ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[1]),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, max_val in axes:
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=max_val,
          step=0.1,
          min=0.1,
          max=max(20.0, float(max_val)),
        )
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    # Store GUI state for compute() override.
    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value
        self.vel_command_w[idx, i] = s.value
        self.cmd_target_b[idx, i] = s.value
      self.is_standing_env[idx] = False
      self.is_world_env[idx] = False
      self.is_heading_env[idx] = False
      self.command_mode[idx] = 2

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      # Helper to transform local to world coordinates.
      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      # Command linear velocity arrow (blue).
      cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
      )
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Command angular velocity arrow (green).
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
      )
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      # Actual linear velocity arrow (cyan).
      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      # Actual angular velocity arrow (light green).
      act_ang_from = act_lin_from
      act_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0
  rel_world_envs: float = 0.0
  """Fraction of environments that use world-frame velocity commands.
  World-frame envs sample linear velocity in world frame and rotate to body
  frame each step, so the command direction stays fixed in the world."""
  rel_forward_envs: float = 0.0
  """Fraction of environments that receive forward-only commands (positive
  lin_vel_x, zero lin_vel_y and ang_vel_z). Increases training coverage for
  straight-line walking, which is important for stair climbing."""
  init_velocity_prob: float = 0.0
  max_lin_accel: float = 0.0
  """If > 0, rate-limit (slew) the velocity command toward the freshly sampled
  target at this acceleration (m/s^2 for linear xy, rad/s^2 for non-heading yaw)
  instead of stepping to it instantly. This makes acceleration gradual so the
  policy tracks a smoothly rising command (and posture adapts to the current
  speed) rather than lunging to close a step error. 0 disables (default;
  original step behavior). Mirrors ultra_run_lab's ``max_lin_accel``."""
  cmd_step_sample_frac: float = 1.0
  """Fraction of envs that, on resample, jump straight to the new command
  (step). The remaining ``1 - frac`` hold their previous command and ramp toward
  the new target under ``max_lin_accel``. 1.0 = all step (original behavior).
  Only used when ``max_lin_accel > 0``. ultra_run_lab uses 0.5."""
  run_high_speed_sample_frac: float = 0.0
  """If > 0, a fraction of *run-style* envs (those whose freshly sampled
  ``|lin_vel_x| >= run_high_speed_enter``) are resampled into the top speed band
  ``[run_high_speed_threshold_frac * vmax, vmax]`` and forced non-standing. This
  guarantees high-speed coverage near the curriculum cap (uniform sampling
  under-represents the top end as the cap widens to 15 m/s). 0 disables
  (default). Mirrors ultra_run_lab's ``run_high_speed_sample_frac``."""
  run_high_speed_threshold_frac: float = 0.7
  """Lower edge of the high-speed band as a fraction of ``vmax``. Only used when
  ``run_high_speed_sample_frac > 0``."""
  run_high_speed_enter: float = 1.05
  """``|lin_vel_x|`` threshold above which an env counts as run-style for the
  high-speed resampling. Should match the style scheduler's ``run_enter``."""
  rel_walk_envs: float = 0.0
  """If > 0, enable ultra_run_lab hist10 stand/walk/run *bucket* sampling. Each
  resampled env is assigned a bucket: stand (prob ``rel_standing_envs``), walk
  (prob ``rel_walk_envs``) or run (remainder). Walk envs draw ``lin_vel_x`` from
  ``walk_lin_vel_x``; run envs draw it from ``[run_lin_vel_x_min, vmax]`` and go
  through the high-speed resampling. 0 disables (default; v4-v8 single-range
  sampling). Mirrors ultra_run_lab's ``rel_walk_envs``."""
  walk_lin_vel_x: tuple[float, float] | None = None
  """Walk-bucket ``lin_vel_x`` range (only used when ``rel_walk_envs > 0``)."""
  run_lin_vel_x_min: float | None = None
  """Run-bucket lower ``lin_vel_x`` bound; run envs draw from
  ``[run_lin_vel_x_min, vmax]`` (only used when ``rel_walk_envs > 0``)."""
  stand_enter_speed: float = 0.08
  """Command speed-norm below which a walk/run bucket env is forced to standing
  (matches hist10 ``style_from_command.stand_enter``). Bucket sampling only."""

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
