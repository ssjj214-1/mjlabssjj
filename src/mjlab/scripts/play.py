"""Script to play RL agent with RSL-RL."""

import os
import sys
import time as _time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from inspect import signature
from pathlib import Path
from threading import Lock
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago


def _parse_wandb_dt(value: str | datetime) -> datetime:
  """Parse a W&B datetime string (or pass through a datetime object)."""
  if isinstance(value, str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
  return value


def _load_runner_checkpoint(runner, path: str, device: str) -> None:
  """Load a checkpoint across runner APIs.

  MjlabOnPolicyRunner exposes `load(path, load_cfg=..., strict=..., map_location=...)`
  while MultiAmpOnPolicyRunner uses a simpler `load(path, load_optimizer=True)`
  signature. This helper selects a compatible call automatically.
  """
  params = signature(runner.load).parameters
  if "load_cfg" in params:
    runner.load(path, load_cfg={"actor": True}, strict=True, map_location=device)
    return

  # Legacy/simple API (e.g. MultiAmpOnPolicyRunner).
  runner.load(path)


class _HIMHistoryPolicy:
  """Adapt a HIM (history-conditioned) inference policy to the viewer's API.

  The AMP+HIM actor expects a flattened observation history of shape
  ``(B, history_length * num_one_step_obs)`` (oldest -> newest), which during
  training is produced by ``UltraAMPHIMVecEnvWrapper``. At play time the viewer
  feeds a single-frame observation TensorDict, so this adapter maintains the
  rolling history buffer (refilling on episode resets) before delegating to the
  underlying policy.
  """

  def __init__(self, actor, env, device: str) -> None:
    self._actor = actor
    self._unwrapped = env.unwrapped
    self._device = device
    self._history_length = int(actor.history_length)
    self._num_one_step_obs = int(actor.num_one_step_obs)
    self._hist: torch.Tensor | None = None

  def _extract_frame(self, obs) -> torch.Tensor:
    actor = obs["actor"] if hasattr(obs, "keys") else obs
    return actor.to(self._device).float()

  def __call__(self, obs):
    frame = self._extract_frame(obs)
    batch = frame.shape[0]
    if self._hist is None or self._hist.shape[0] != batch:
      self._hist = frame[:, None, :].repeat(1, self._history_length, 1)
    else:
      self._hist = torch.roll(self._hist, shifts=-1, dims=1)
      self._hist[:, -1, :] = frame
      # Refill history for envs that just reset so stale frames don't leak.
      ep_len = self._unwrapped.episode_length_buf
      reset_mask = ep_len == 0
      if torch.any(reset_mask):
        self._hist[reset_mask] = frame[reset_mask][:, None, :]
    flat = self._hist.reshape(batch, -1)
    return self._actor.act_inference(flat)


def _maybe_wrap_him_policy(policy, runner, env, device: str):
  """Wrap ``policy`` with history stacking if the runner uses a HIM actor."""
  actor = getattr(getattr(runner, "alg", None), "policy", None)
  if actor is not None and getattr(actor, "use_him", False):
    return _HIMHistoryPolicy(actor, env, device)
  return policy


class _VelocityTeleop:
  """Keyboard velocity teleop + real-time speed printing for the velocity task.

  Replaces the random velocity command with a manually-driven one and prints
  the commanded vs. actual base velocity to the terminal. The command term's
  ``compute`` is wrapped so the manual command is written after every env step
  (and randomization is disabled so it stays stable).
  """

  def __init__(
    self,
    env,
    *,
    init_vx: float = 0.0,
    init_vy: float = 0.0,
    init_wz: float = 0.0,
    vx_step: float = 0.5,
    vy_step: float = 0.25,
    wz_step: float = 0.25,
    vx_range: tuple[float, float] = (-0.5, 20.0),
    vy_range: tuple[float, float] = (-1.0, 1.0),
    wz_range: tuple[float, float] = (-2.0, 2.0),
    print_period_s: float = 0.25,
    deploy_slew_accel: float = 4.0,
  ) -> None:
    self._unwrapped = env.unwrapped
    self._term = self._unwrapped.command_manager.get_term("twist")
    self._robot = self._unwrapped.scene["robot"]
    self._env = self._unwrapped
    self._vx = self._clamp(init_vx, vx_range)
    self._vy = self._clamp(init_vy, vy_range)
    self._wz = self._clamp(init_wz, wz_range)
    self._cmd_vx = 0.0
    self._cmd_vy = 0.0
    self._cmd_wz = 0.0
    self._deploy_slew_accel = deploy_slew_accel
    self._vx_step, self._vy_step, self._wz_step = vx_step, vy_step, wz_step
    self._vx_range, self._vy_range, self._wz_range = vx_range, vy_range, wz_range
    self._print_period_s = print_period_s
    self._last_print = 0.0
    self._lock = Lock()
    self._disable_randomization()
    self._patch_compute()

  def _disable_randomization(self) -> None:
    cfg = self._term.cfg
    cfg.heading_command = False
    cfg.ranges.heading = None
    cfg.rel_standing_envs = 0.0
    cfg.rel_heading_envs = 0.0
    cfg.rel_world_envs = 0.0
    cfg.rel_forward_envs = 0.0
    cfg.rel_walk_envs = 0.0
    cfg.init_velocity_prob = 0.0
    cfg.resampling_time_range = (1.0e9, 1.0e9)
    cfg.max_lin_accel = max(float(cfg.max_lin_accel), self._deploy_slew_accel)

  def _patch_compute(self) -> None:
    term = self._term
    orig_compute = term.compute

    def compute(dt: float) -> None:
      orig_compute(dt)
      self._apply()

    term.compute = compute

  def _apply(self) -> None:
    with self._lock:
      target_vx, target_vy, target_wz = self._vx, self._vy, self._wz
    max_dv = max(float(self._term.cfg.max_lin_accel), 0.0) * self._env.step_dt
    if max_dv > 0.0:
      self._cmd_vx += self._clamp(target_vx - self._cmd_vx, (-max_dv, max_dv))
      self._cmd_vy += self._clamp(target_vy - self._cmd_vy, (-max_dv, max_dv))
      self._cmd_wz += self._clamp(target_wz - self._cmd_wz, (-max_dv, max_dv))
    else:
      self._cmd_vx, self._cmd_vy, self._cmd_wz = target_vx, target_vy, target_wz
    cmd = torch.tensor(
      [self._cmd_vx, self._cmd_vy, self._cmd_wz],
      device=self._term.vel_command_b.device,
      dtype=self._term.vel_command_b.dtype,
    )
    self._term.vel_command_b[:] = cmd
    self._term.vel_command_w[:] = cmd
    if hasattr(self._term, "cmd_target_b"):
      target = torch.tensor(
        [target_vx, target_vy, target_wz],
        device=self._term.vel_command_b.device,
        dtype=self._term.vel_command_b.dtype,
      )
      self._term.cmd_target_b[:] = target
    if hasattr(self._term, "is_standing_env"):
      self._term.is_standing_env[:] = False
    if hasattr(self._term, "command_mode"):
      self._term.command_mode[:] = 2
    self._maybe_print(target_vx, target_vy, target_wz, cmd)

  def _maybe_print(self, vx: float, vy: float, wz: float, cmd: torch.Tensor) -> None:
    now = _time.time()
    if now - self._last_print < self._print_period_s:
      return
    self._last_print = now
    lin = self._robot.data.root_link_lin_vel_b[0]
    ang = self._robot.data.root_link_ang_vel_b[0]
    msg = (
      f"\r[target] vx={vx:+5.2f} vy={vy:+5.2f} wz={wz:+5.2f} | "
      f"[cmd] vx={cmd[0].item():+5.2f} vy={cmd[1].item():+5.2f} "
      f"wz={cmd[2].item():+5.2f} | "
      f"[act] vx={lin[0].item():+5.2f} vy={lin[1].item():+5.2f} "
      f"wz={ang[2].item():+5.2f} m/s,rad/s   "
    )
    print(msg, end="", flush=True)

  def set_velocity(self, vx: float, vy: float, wz: float) -> None:
    """Set absolute velocity command (used by joystick teleop)."""
    with self._lock:
      self._vx = self._clamp(vx, self._vx_range)
      self._vy = self._clamp(vy, self._vy_range)
      self._wz = self._clamp(wz, self._wz_range)

  @property
  def ranges(
    self,
  ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    return self._vx_range, self._vy_range, self._wz_range

  @staticmethod
  def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))

  def handle_key(self, key: int) -> None:
    from mjlab.viewer.native.keys import (
      KEY_C,
      KEY_E,
      KEY_Q,
      KEY_S,
      KEY_W,
      KEY_X,
      KEY_Z,
    )

    with self._lock:
      if key == KEY_W:
        self._vx = self._clamp(self._vx + self._vx_step, self._vx_range)
      elif key == KEY_S:
        self._vx = self._clamp(self._vx - self._vx_step, self._vx_range)
      elif key == KEY_Q:
        self._wz = self._clamp(self._wz + self._wz_step, self._wz_range)
      elif key == KEY_E:
        self._wz = self._clamp(self._wz - self._wz_step, self._wz_range)
      elif key == KEY_Z:
        self._vy = self._clamp(self._vy + self._vy_step, self._vy_range)
      elif key == KEY_C:
        self._vy = self._clamp(self._vy - self._vy_step, self._vy_range)
      elif key == KEY_X:
        self._vx = self._vy = self._wz = 0.0


class _JoystickTeleop:
  """Background-thread Xbox/XInput gamepad reader that drives ``_VelocityTeleop``.

  Reads analog sticks via pygame and writes the velocity command proportionally,
  so the gamepad works with either viewer backend. Degrades to a no-op (and a
  warning) if pygame or a joystick is unavailable. Mapping mirrors the deploy SDK
  (standard SDL XInput layout):
    Left stick  Y -> vx (push up = forward), Left stick X -> vy (push left = +y)
    Right stick X -> wz (push left = +yaw)
    Button Y (3)  -> zero,   Button RB (5) -> full-throttle forward (vx_max)
  """

  def __init__(self, teleop: "_VelocityTeleop", vx_max: float = 12.5) -> None:
    self._teleop = teleop
    self._vx_max = vx_max
    (self._vx_range, self._vy_range, self._wz_range) = teleop.ranges
    self._deadzone = 0.12
    self._vx_step = 0.5
    self._wz_step = 0.25
    self._running = False
    self._ok = False
    self._js = None
    self._connected = False
    self._num_axes = 0
    self._num_buttons = 0
    self._num_hats = 0
    self._prev_hat = (0, 0)
    # Persistent command (held between D-pad presses, like the keyboard).
    self._vx = 0.0
    self._vy = 0.0
    self._wz = 0.0
    self._last_acquire = 0.0
    try:
      os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
      import pygame

      self._pygame = pygame
      pygame.init()
      pygame.joystick.init()
      self._ok = True
      self._acquire()
      if self._connected:
        print(
          f"[INFO] Joystick teleop enabled: {self._js.get_name()} "  # type: ignore[union-attr]
          f"(axes={self._num_axes}, buttons={self._num_buttons}, hats={self._num_hats})\n"
          f"       D-pad up/down = vx +/-{self._vx_step} (one step per press, holds), "
          f"left/right = yaw\n"
          "       L-stick = forward/strafe, R-stick = yaw (proportional while pushed), "
          f"Y = zero, RB = full forward ({vx_max:.1f} m/s)"
        )
      else:
        print(
          "[INFO] Joystick teleop armed but no pad detected yet "
          "(wireless pad asleep? press a button) — will auto-connect."
        )
    except Exception as e:  # noqa: BLE001 - degrade gracefully.
      print(f"[WARN] --joystick-cmd ignored: {type(e).__name__}: {e}")

  def _acquire(self) -> bool:
    try:
      self._pygame.joystick.quit()
      self._pygame.joystick.init()
      if self._pygame.joystick.get_count() == 0:
        self._connected = False
        self._js = None
        return False
      self._js = self._pygame.joystick.Joystick(0)
      self._js.init()
      self._num_axes = self._js.get_numaxes()
      self._num_buttons = self._js.get_numbuttons()
      self._num_hats = self._js.get_numhats()
      self._prev_hat = (0, 0)
      self._connected = True
      return True
    except Exception:  # noqa: BLE001
      self._connected = False
      self._js = None
      return False

  @staticmethod
  def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))

  def _dz(self, v: float) -> float:
    if abs(v) < self._deadzone:
      return 0.0
    sign = 1.0 if v > 0 else -1.0
    return sign * (abs(v) - self._deadzone) / (1.0 - self._deadzone)

  def _loop(self) -> None:
    while self._running:
      if not self._connected:
        now = _time.time()
        if now - self._last_acquire > 1.0:
          self._last_acquire = now
          if self._acquire():
            print(f"[INFO] Joystick connected: {self._js.get_name()}")  # type: ignore[union-attr]
        _time.sleep(0.05)
        continue
      try:
        self._pygame.event.pump()

        def raw(i: int) -> float:
          return self._js.get_axis(i) if 0 <= i < self._num_axes else 0.0  # type: ignore[union-attr]

        sprint = self._num_buttons > 5 and self._js.get_button(5) == 1  # type: ignore[union-attr]
        zero = self._num_buttons > 3 and self._js.get_button(3) == 1  # type: ignore[union-attr]

        # D-pad (hat) edge -> one incremental step, like keyboard W/S/A/D.
        up = down = left = right = False
        if self._num_hats > 0:
          hx, hy = self._js.get_hat(0)  # type: ignore[union-attr]
          phx, phy = self._prev_hat
          up = hy == 1 and phy != 1
          down = hy == -1 and phy != -1
          left = hx == -1 and phx != -1
          right = hx == 1 and phx != 1
          self._prev_hat = (hx, hy)

        r_vx, r_vy, r_wz = raw(1), raw(0), raw(2)
        if zero:
          self._vx = self._vy = self._wz = 0.0
        elif sprint:
          self._vx, self._vy, self._wz = self._vx_max, 0.0, 0.0
        else:
          if up:
            self._vx = self._clamp(self._vx + self._vx_step, self._vx_range)
          if down:
            self._vx = self._clamp(self._vx - self._vx_step, self._vx_range)
          if left:
            self._wz = self._clamp(self._wz + self._wz_step, self._wz_range)
          if right:
            self._wz = self._clamp(self._wz - self._wz_step, self._wz_range)
          # Sticks override their axis only while pushed; centered = hold.
          if abs(r_vx) > self._deadzone:
            self._vx = self._clamp(-self._dz(r_vx) * self._vx_max, self._vx_range)
          if abs(r_vy) > self._deadzone:
            self._vy = self._clamp(-self._dz(r_vy) * self._vy_range[1], self._vy_range)
          if abs(r_wz) > self._deadzone:
            self._wz = self._clamp(-self._dz(r_wz) * self._wz_range[1], self._wz_range)
        self._teleop.set_velocity(self._vx, self._vy, self._wz)
      except Exception:  # noqa: BLE001 - device likely unplugged/asleep.
        self._connected = False
        self._js = None
        self._vx = self._vy = self._wz = 0.0
        self._teleop.set_velocity(0.0, 0.0, 0.0)
        print("[INFO] Joystick disconnected (will auto-reconnect).")
      _time.sleep(0.02)

  def start(self) -> None:
    if not self._ok:
      return
    self._running = True
    from threading import Thread

    Thread(target=self._loop, daemon=True).start()


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  keyboard_cmd: bool = False
  """Drive the velocity command from the keyboard and print speed in real time.

  Disables random command resampling. Native-viewer keys:
    W/S forward +/-, Q/E yaw left/right, Z/C strafe left/right, X zero.
  """
  joystick_cmd: bool = False
  """Drive the velocity command from an Xbox/XInput gamepad (pygame).

  Works with both viewer backends. L-stick = forward/strafe, R-stick = yaw,
  Y = zero, RB = full forward. Can be combined with --keyboard-cmd.
  """
  joy_vx_max: float = 12.5
  """Forward speed (m/s) at full-throttle left stick / RB. Matches the deploy SDK default."""
  init_vx: float = 0.0
  """Initial forward velocity command (m/s) for keyboard teleop. Set e.g. 8.0 to
  launch from standstill straight to 8 m/s and watch the start/acceleration
  posture; W/S still adjust it live. Requires --keyboard-cmd."""
  init_vy: float = 0.0
  """Initial lateral velocity command (m/s) for keyboard teleop."""
  init_wz: float = 0.0
  """Initial yaw-rate command (rad/s) for keyboard teleop."""
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs are written."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    _load_runner_checkpoint(runner, str(resume_path), device)
    policy = runner.get_inference_policy(device=device)
    policy = _maybe_wrap_him_policy(policy, runner, env, device)

  # Build checkpoint manager for hot-swapping checkpoints in the viewer.
  ckpt_manager: CheckpointManager | None = None
  if TRAINED_MODE and resume_path is not None:
    _ckpt_runner = runner  # pyright: ignore[reportPossiblyUnboundVariable]

    def _reload_policy(path: str):
      _load_runner_checkpoint(_ckpt_runner, path, device)
      reloaded = _ckpt_runner.get_inference_policy(device=device)
      return _maybe_wrap_him_policy(reloaded, _ckpt_runner, env, device)

    if cfg.wandb_run_path is None:
      ckpt_dir = resume_path.parent

      def fetch_available_local() -> list[tuple[str, str]]:
        now = _time.time()
        entries: list[tuple[str, str, int]] = []
        for f in sorted(ckpt_dir.glob("*.pt")):
          try:
            step = int(f.stem.split("_")[1])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(int(now - f.stat().st_mtime))
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_local,
        load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)),
      )
    else:
      import wandb

      api = wandb.Api()
      run_path = str(cfg.wandb_run_path)
      wandb_run = api.run(run_path)
      _log_root = log_root_path  # pyright: ignore[reportPossiblyUnboundVariable]

      def fetch_available_wandb() -> list[tuple[str, str]]:
        wandb_run.load()
        now = datetime.now(tz=timezone.utc)
        entries: list[tuple[str, str, int]] = []
        for f in wandb_run.files():
          if not f.name.endswith(".pt"):
            continue
          try:
            step = int(f.name.split("_")[1].split(".")[0])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(
            int((now - _parse_wandb_dt(f.updated_at)).total_seconds())
          )
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_wandb,
        load_checkpoint=lambda name: _reload_policy(
          str(get_wandb_checkpoint_path(_log_root, Path(run_path), name)[0])
        ),
        run_name=_parse_wandb_dt(wandb_run.created_at).strftime("%Y-%m-%d_%H-%M-%S"),
        run_url=wandb_run.url,
        run_status=wandb_run.state,
      )

  # Optional keyboard / joystick velocity teleop + real-time speed printing.
  teleop: _VelocityTeleop | None = None
  if cfg.keyboard_cmd or cfg.joystick_cmd:
    if "twist" in env.unwrapped.command_manager.active_terms:
      teleop = _VelocityTeleop(
        env, init_vx=cfg.init_vx, init_vy=cfg.init_vy, init_wz=cfg.init_wz
      )
      if cfg.keyboard_cmd:
        print(
          "[INFO] Keyboard velocity teleop enabled:\n"
          "       W/S = forward +/-, Q/E = yaw left/right, "
          "Z/C = strafe left/right, X = zero\n"
          f"       initial cmd: vx={cfg.init_vx:+.2f} vy={cfg.init_vy:+.2f} "
          f"wz={cfg.init_wz:+.2f}"
        )
      if cfg.joystick_cmd:
        _JoystickTeleop(teleop, vx_max=cfg.joy_vx_max).start()
    else:
      print("[WARN] --keyboard-cmd/--joystick-cmd ignored: no 'twist' command in task.")

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  key_callback = teleop.handle_key if teleop is not None else None

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy, key_callback=key_callback).run()
  elif resolved_viewer == "viser":
    if teleop is not None:
      print(
        "[WARN] Keyboard teleop key input is only wired for the native viewer; "
        "speed printing still works, but use the Viser joystick GUI to steer."
      )
    ViserPlayViewer(env, policy, checkpoint_manager=ckpt_manager).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  maybe_print_top_level_help("play")

  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
