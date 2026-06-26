"""Recovery reset utilities for Ultra GameYaw AMP/HIM tasks.

This mirrors the recovery part of the older AMP_mjlab pipeline: a fixed subset
of environments delays failure resets, and those environments reset from
fall/get-up motion frames so the same locomotion policy sees recovery states.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationManager
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_REQUIRED_NPZ_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


class DelayedTerminationManager(TerminationManager):
  """Delay non-timeout resets for a subset of environments."""

  def __init__(
    self,
    base: TerminationManager,
    delay_env_mask: torch.Tensor,
    max_delay_steps: int,
  ) -> None:
    self.__dict__.update(base.__dict__)
    self._delay_env_mask = delay_env_mask
    self._delay_counters = torch.zeros_like(delay_env_mask, dtype=torch.long)
    self._max_delay_steps = int(max_delay_steps)

  def compute(self) -> torch.Tensor:
    dones = super().compute()
    if self._max_delay_steps <= 0:
      return dones

    delay_and_terminated = (
      self._delay_env_mask & self._terminated_buf & ~self._truncated_buf
    )
    self._delay_counters[delay_and_terminated] += 1

    not_ready = delay_and_terminated & (self._delay_counters < self._max_delay_steps)
    self._terminated_buf[not_ready] = False

    ready = delay_and_terminated & (self._delay_counters >= self._max_delay_steps)
    self._delay_counters[ready] = 0

    recovered = self._delay_env_mask & ~delay_and_terminated
    self._delay_counters[recovered] = 0
    return self._truncated_buf | self._terminated_buf

  def reset(
    self, env_ids: torch.Tensor | slice | None = None
  ) -> dict[str, torch.Tensor]:
    if env_ids is None:
      env_ids = slice(None)
    self._delay_counters[env_ids] = 0
    return super().reset(env_ids)


class RecoveryMotionResetManager:
  """Cache recovery motion frames and write random frames on reset."""

  _instance: RecoveryMotionResetManager | None = None

  def __init__(self) -> None:
    self.recovery_frames: dict[str, dict[str, torch.Tensor]] = {}

  @classmethod
  def get(cls) -> RecoveryMotionResetManager:
    if cls._instance is None:
      cls._instance = cls()
    return cls._instance

  def init(self, env: ManagerBasedRlEnv, recovery_dir: str) -> None:
    recovery_dir = os.path.abspath(recovery_dir)
    if recovery_dir in self.recovery_frames:
      return
    motions = self._load_dir(recovery_dir, device=str(env.device))
    self.recovery_frames[recovery_dir] = self._concat_frames(motions)
    frame_count = self.recovery_frames[recovery_dir]["root_pos"].shape[0]
    print(
      "[RecoveryMotionResetManager] Loaded "
      f"{len(motions)} recovery clips, {frame_count} frames from {recovery_dir}"
    )

  def reset(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    recovery_dir: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> None:
    recovery_dir = os.path.abspath(recovery_dir)
    if recovery_dir not in self.recovery_frames:
      self.init(env, recovery_dir)

    delay_mask = _get_delay_env_mask(env)
    if delay_mask is None:
      return

    recovery_env_ids = env_ids[delay_mask[env_ids]]
    if recovery_env_ids.numel() == 0:
      return

    self._write_reset_state(
      env,
      env_ids=recovery_env_ids,
      frames=self.recovery_frames[recovery_dir],
      asset_cfg=asset_cfg,
    )

  @staticmethod
  def _load_dir(dir_path: str, device: str) -> list[dict[str, torch.Tensor]]:
    if not os.path.isdir(dir_path):
      raise FileNotFoundError(f"Recovery motion directory not found: {dir_path}")

    motions: list[dict[str, torch.Tensor]] = []
    for filename in sorted(os.listdir(dir_path)):
      if not filename.endswith(".npz"):
        continue
      path = os.path.join(dir_path, filename)
      data = np.load(path)
      missing = [key for key in _REQUIRED_NPZ_KEYS if key not in data.files]
      if missing:
        raise KeyError(f"Recovery motion {path} missing keys: {missing}")
      motions.append(
        {
          "root_pos": torch.tensor(
            data["body_pos_w"][:, 0, :], dtype=torch.float32, device=device
          ),
          "root_quat": torch.tensor(
            data["body_quat_w"][:, 0, :], dtype=torch.float32, device=device
          ),
          "root_lin_vel": torch.tensor(
            data["body_lin_vel_w"][:, 0, :], dtype=torch.float32, device=device
          ),
          "root_ang_vel": torch.tensor(
            data["body_ang_vel_w"][:, 0, :], dtype=torch.float32, device=device
          ),
          "joint_pos": torch.tensor(
            data["joint_pos"], dtype=torch.float32, device=device
          ),
          "joint_vel": torch.tensor(
            data["joint_vel"], dtype=torch.float32, device=device
          ),
        }
      )

    if not motions:
      raise FileNotFoundError(f"No .npz recovery motions found in: {dir_path}")
    return motions

  @staticmethod
  def _concat_frames(
    motions: list[dict[str, torch.Tensor]],
  ) -> dict[str, torch.Tensor]:
    return {
      key: torch.cat([motion[key] for motion in motions], dim=0)
      for key in (
        "root_pos",
        "root_quat",
        "root_lin_vel",
        "root_ang_vel",
        "joint_pos",
        "joint_vel",
      )
    }

  @staticmethod
  def _write_reset_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    frames: dict[str, torch.Tensor],
    asset_cfg: SceneEntityCfg,
  ) -> None:
    num_reset = len(env_ids)
    frame_ids = torch.randint(
      0, frames["root_pos"].shape[0], (num_reset,), device=env.device
    )

    asset: Entity = env.scene[asset_cfg.name]
    root_pos = frames["root_pos"][frame_ids]
    root_quat = frames["root_quat"][frame_ids]
    positions = env.scene.env_origins[env_ids].clone()
    # The motion frame z is measured from the ground (z=0 in the mocap). On
    # terrain, env_origins[:, 2] carries the spawn tile's surface height, so the
    # frame z must be added on top of it -- not overwrite it -- or the robot is
    # placed at absolute mocap height and floats above / sinks into the terrain.
    # On a flat plane env_origins[:, 2] == 0, so this reduces to the frame z.
    positions[:, 2] = positions[:, 2] + root_pos[:, 2]

    asset.write_root_link_pose_to_sim(
      torch.cat([positions, root_quat], dim=-1), env_ids=env_ids
    )
    asset.write_root_link_velocity_to_sim(
      torch.cat(
        [
          frames["root_lin_vel"][frame_ids],
          frames["root_ang_vel"][frame_ids],
        ],
        dim=-1,
      ),
      env_ids=env_ids,
    )

    joint_pos = frames["joint_pos"][frame_ids]
    joint_vel = frames["joint_vel"][frame_ids]
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
      joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)

    soft_limits = asset.data.soft_joint_pos_limits
    assert soft_limits is not None
    joint_limits = soft_limits[env_ids][:, asset_cfg.joint_ids]
    joint_pos = joint_pos[:, asset_cfg.joint_ids].clamp_(
      joint_limits[..., 0], joint_limits[..., 1]
    )

    asset.write_joint_state_to_sim(
      joint_pos,
      joint_vel[:, asset_cfg.joint_ids],
      joint_ids=joint_ids,
      env_ids=env_ids,
    )


def _get_delay_env_mask(env: ManagerBasedRlEnv) -> torch.Tensor | None:
  manager = env.termination_manager
  if isinstance(manager, DelayedTerminationManager):
    return manager._delay_env_mask
  return None


def _get_active_delay_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
  manager = env.termination_manager
  if isinstance(manager, DelayedTerminationManager):
    return manager._delay_env_mask & (manager._delay_counters > 0)
  return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def recovery_active_ratio(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 for envs currently inside delayed recovery, else 0."""

  return _get_active_delay_mask(env).to(torch.float32)


def recovery_delay_progress(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per-env delayed recovery counter normalized by ``max_delay_steps``."""

  manager = env.termination_manager
  if not isinstance(manager, DelayedTerminationManager):
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
  denom = max(float(manager._max_delay_steps), 1.0)
  return manager._delay_counters.to(torch.float32) / denom


class RecoveryTransitionMetric:
  """Detect starts, successes, and max-delay failures of recovery attempts."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    del cfg
    self._prev_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(self, env: ManagerBasedRlEnv, mode: str) -> torch.Tensor:
    active = _get_active_delay_mask(env)
    started = ~self._prev_active & active
    ended = self._prev_active & ~active
    reset = env.reset_buf.bool()
    terminated = env.reset_terminated.bool()

    if mode == "attempt":
      out = started
    elif mode == "success":
      out = ended & ~reset
    elif mode == "failure":
      out = ended & reset & terminated
    else:
      raise ValueError(f"Unsupported recovery transition metric mode: {mode}")

    self._prev_active = active.clone()
    return out.to(torch.float32)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._prev_active[env_ids] = False


class RecoveryEpisodeOutcomeMetric:
  """Latch whether an episode has seen a recovery success or failure."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    del cfg
    self._prev_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._occurred = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(self, env: ManagerBasedRlEnv, mode: str) -> torch.Tensor:
    active = _get_active_delay_mask(env)
    ended = self._prev_active & ~active
    reset = env.reset_buf.bool()
    terminated = env.reset_terminated.bool()

    if mode == "success":
      event = ended & ~reset
    elif mode == "failure":
      event = ended & reset & terminated
    else:
      raise ValueError(f"Unsupported recovery outcome metric mode: {mode}")

    self._occurred |= event
    self._prev_active = active.clone()
    return self._occurred.to(torch.float32)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._prev_active[env_ids] = False
    self._occurred[env_ids] = False


def ultra_recovery_style_update(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  command_name: str = "twist",
  stand_enter: float = 0.08,
  stand_exit: float = 0.12,
  run_enter: float = 1.05,
  run_exit: float = 0.85,
  recovery_style_id: int = 3,
) -> None:
  """Stand/walk/run style update with active delayed-fall envs set to recovery."""

  from . import ultra_mdp

  ultra_mdp.ultra_style_update(
    env=env,
    env_ids=env_ids,
    command_name=command_name,
    stand_enter=stand_enter,
    stand_exit=stand_exit,
    run_enter=run_enter,
    run_exit=run_exit,
  )
  active_delay = _get_active_delay_mask(env)
  if active_delay.any():
    style_ids = cast(torch.Tensor, env.__dict__["style_ids"])
    style_ids[active_delay] = int(recovery_style_id)
    env.__dict__["style_ids"] = style_ids


def recovery_root_height_exp(
  env: ManagerBasedRlEnv,
  std: float = 0.3,
  delay_env_rew_ratio: float = 3.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  terrain_height_sensor: str | None = None,
) -> torch.Tensor:
  """Reward active recovery envs for returning the root to default height.

  With ``terrain_height_sensor`` set (a base down-ray ``TerrainHeightSensor``),
  the height is measured against the ground directly under the robot
  (``root_z - terrain_z``) so the target is correct on terrain. Without it the
  absolute world ``root_z`` is used, which is only valid on a flat plane.
  """

  asset: Entity = env.scene[asset_cfg.name]
  active_delay = _get_active_delay_mask(env)
  if terrain_height_sensor is not None:
    cur_height = env.scene[terrain_height_sensor].data.heights[:, 0]
  else:
    cur_height = asset.data.root_link_pos_w[:, 2]
  height_error = torch.square(asset.data.default_root_state[:, 2] - cur_height)
  reward = torch.exp(-height_error / (std * std)) * float(delay_env_rew_ratio)
  return torch.where(active_delay, reward, torch.zeros_like(reward))


def recovery_body_orientation_exp(
  env: ManagerBasedRlEnv,
  delay_env_rew_ratio: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward active recovery envs for bringing the waist/base back upright."""

  asset: Entity = env.scene[asset_cfg.name]
  active_delay = _get_active_delay_mask(env)
  body_id = 0
  if isinstance(asset_cfg.body_ids, list) and asset_cfg.body_ids:
    body_id = int(asset_cfg.body_ids[0])
  proj = quat_apply_inverse(
    asset.data.body_link_quat_w[:, body_id, :],
    asset.data.gravity_vec_w,
  )
  err = proj[:, 0].square() + proj[:, 1].square()
  reward = torch.exp(-100.0 * err) * float(delay_env_rew_ratio)
  return torch.where(active_delay, reward, torch.zeros_like(reward))


def init_recovery_motion_loader(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  recovery_dir: str,
  delay_reset_env_ratio: float = 0.4,
  max_delay_steps: int = 250,
) -> None:
  """Startup event: load recovery motions and install delayed termination."""

  del env_ids
  RecoveryMotionResetManager.get().init(env, recovery_dir)

  num_delay = int(env.num_envs * float(delay_reset_env_ratio))
  if num_delay <= 0 or max_delay_steps <= 0:
    return

  delay_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  delay_indices = torch.randperm(env.num_envs, device=env.device)[:num_delay]
  delay_mask[delay_indices] = True
  env.termination_manager = DelayedTerminationManager(
    base=env.termination_manager,
    delay_env_mask=delay_mask,
    max_delay_steps=max_delay_steps,
  )
  print(
    "[init_recovery_motion_loader] DelayedTerminationManager installed: "
    f"{num_delay}/{env.num_envs} envs, max_delay_steps={max_delay_steps}"
  )


def reset_from_recovery_motion(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  recovery_dir: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset delayed-recovery envs from random fall/get-up motion frames."""

  RecoveryMotionResetManager.get().reset(
    env=env,
    env_ids=env_ids,
    recovery_dir=recovery_dir,
    asset_cfg=asset_cfg,
  )
