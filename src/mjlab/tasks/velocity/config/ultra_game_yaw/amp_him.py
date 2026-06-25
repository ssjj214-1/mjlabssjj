"""AMP + HIM training integration for the Ultra GameYaw velocity task.

Brings the multi-style AMP (WGAN) + HIM-GRU encoder pipeline from the Ultra
Isaac Lab fork (vendored under ``mjlab.third_party.amp_rsl_rl``) on top of
mjlab's manager-based velocity environment.

What this module provides:

* ``UltraAMPHIMVecEnvWrapper`` — adapter exposing the contract that
  :class:`MultiAmpOnPolicyRunner` expects (history-stacked actor obs,
  ``num_one_step_obs``, ``style_ids``, ``get_amp_obs_for_expert_trans``,
  ``get_amp_observations_pos``, ``reset_env_ids``, etc.).
* ``UltraGameYawAMPHIMRunner`` — thin subclass of
  :class:`MultiAmpOnPolicyRunner` that wraps the underlying
  :class:`RslRlVecEnvWrapper` into the AMP/HIM adapter.
* ``RslRlAmpHimRunnerCfg`` — dataclass mirroring Ultra's
  ``UltraGameYawMotionRunAgentCfg`` so ``mjlab.scripts.train`` can serialise
  the agent config and feed it to the runner unchanged.
* ``ultra_game_yaw_amp_him_runner_cfg`` — factory returning the dataclass
  populated with the same hyper-parameters as the Ultra training run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple

import torch

from mjlab import MJLAB_SRC_PATH
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.third_party.amp_rsl_rl.runners import MultiAmpOnPolicyRunner

# ---------------------------------------------------------------------------
# AMP motion datasets (copied verbatim from the Ultra Isaac Lab task).
# ---------------------------------------------------------------------------

_AMP_DIR: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "ultra_game_yaw" / "amp_motions"
)

ULTRA_AMP_MOTION_FILES: dict[str, list[str]] = {
  "style_0": [str(_AMP_DIR / "stand_ultra_yaw2.txt")],
  "style_1": [str(_AMP_DIR / "walk_ultra_yaw2.txt")],
  "style_2": [
    str(_AMP_DIR / "run_17200_9mps.txt"),
    str(_AMP_DIR / "run_ultrayaw_14mps.txt"),
  ],
}

# AMP frame layout (matches AMPLoader.{JOINT,END}_*_IDX, total 38 columns).
_AMP_JOINT_NAMES: Tuple[str, ...] = (
  "shoulder_pitch_r_joint",
  "shoulder_pitch_l_joint",
  "hip_yaw_r_joint",
  "hip_roll_r_joint",
  "hip_pitch_r_joint",
  "knee_pitch_r_joint",
  "ankle_pitch_r_joint",
  "hip_yaw_l_joint",
  "hip_roll_l_joint",
  "hip_pitch_l_joint",
  "knee_pitch_l_joint",
  "ankle_pitch_l_joint",
  "waist_yaw_joint",
)
_AMP_EE_BODIES: Tuple[str, ...] = (
  "shoulder_pitch_l_link",  # left hand proxy (shoulder + offset)
  "shoulder_pitch_r_link",  # right hand proxy
)
_AMP_FOOT_BODY_CANDIDATES: Tuple[Tuple[str, str], ...] = (
  # AMP motion files encode the pitch-link foot proxy. V10's passive roll links
  # are physical contact feet, but they should not change the AMP target frame.
  ("ankle_pitch_l_link", "ankle_pitch_r_link"),
  ("ankle_roll_l_link", "ankle_roll_r_link"),
)
_HAND_LOCAL_OFFSET = (0.0, 0.0, -0.4)


def _select_amp_foot_bodies(body_names: list[str] | tuple[str, ...]) -> Tuple[str, str]:
  body_name_set = set(body_names)
  for candidate in _AMP_FOOT_BODY_CANDIDATES:
    if all(name in body_name_set for name in candidate):
      return candidate
  raise KeyError(f"No supported AMP foot bodies found in robot bodies: {body_names}")


# ---------------------------------------------------------------------------
# VecEnv adapter exposing AMP + HIM hooks.
# ---------------------------------------------------------------------------


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Apply WXYZ quaternion to vec3 (broadcasted over leading dims)."""
  w, xyz = q[..., 0:1], q[..., 1:4]
  t = 2.0 * torch.cross(xyz, v, dim=-1)
  return v + w * t + torch.cross(xyz, t, dim=-1)


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
  out = q.clone()
  out[..., 1:4] *= -1.0
  return out


class UltraAMPHIMVecEnvWrapper:
  """Wrap a :class:`RslRlVecEnvWrapper` into a vec-env compatible with
  :class:`MultiAmpOnPolicyRunner`.

  This class is **not** a subclass of
  :class:`mjlab.third_party.amp_rsl_rl.env.VecEnv` (that base is empty) — it
  just implements the duck-typed interface the runner uses.
  """

  def __init__(
    self,
    base_env: RslRlVecEnvWrapper,
    history_length: int,
    num_styles: int,
    style_thresholds: dict[str, float],
    critic_history_length: int = 1,
    obs_lin_vel_scale: float = 1.0,
  ) -> None:
    self._base = base_env
    self.unwrapped: ManagerBasedRlEnv = base_env.unwrapped
    self.cfg = self.unwrapped.cfg
    self.device = base_env.device
    self.num_envs = base_env.num_envs
    self.num_actions = base_env.num_actions
    self.step_dt = self.unwrapped.step_dt
    self.max_episode_length = self.unwrapped.max_episode_length

    # History config (also exposed to MultiAmpOnPolicyRunner via num_one_step_obs).
    self.history_length = int(history_length)
    self.critic_history_length = int(critic_history_length)
    self.num_styles = int(num_styles)
    self.style_thresholds = dict(style_thresholds)

    # Detect single-frame obs sizes by computing once.
    obs_dict = self.unwrapped.observation_manager.compute()
    actor_obs = obs_dict["actor"]
    critic_obs = obs_dict["critic"]
    self.num_one_step_obs = int(actor_obs.shape[1])
    self.num_one_step_critic_obs = int(critic_obs.shape[1])
    # In the Ultra-aligned env the first 3 dims of critic obs are the
    # privileged ``base_lin_vel`` (yaw-frame) -- the HIM target.
    crit_terms = list(self.unwrapped.observation_manager.active_terms["critic"])
    self.him_vel_index_in_critic = (
      crit_terms.index("base_lin_vel") if "base_lin_vel" in crit_terms else 0
    )
    # Match the runner's obs scaling lookup for HIM supervision.
    self._obs_lin_vel_scale = float(obs_lin_vel_scale)

    # History buffer: (num_envs, history_length, num_one_step_obs).
    self._actor_history = torch.zeros(
      self.num_envs,
      self.history_length,
      self.num_one_step_obs,
      device=self.device,
      dtype=torch.float32,
    )
    self._critic_history = torch.zeros(
      self.num_envs,
      self.critic_history_length,
      self.num_one_step_critic_obs,
      device=self.device,
      dtype=torch.float32,
    )

    # Style ids are owned by the underlying env (set by the
    # ``ultra_style_update`` step event so reward functions can read them).
    # Initialize the env-side buffer if missing (defensive).
    if not hasattr(self.unwrapped, "style_ids"):
      self.unwrapped.style_ids = torch.zeros(
        self.num_envs, dtype=torch.long, device=self.device
      )
    self.reset_env_ids = torch.zeros(0, dtype=torch.long, device=self.device)
    self.last_episode_age = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    # Cache joint and body indices used by the AMP feature extractor.
    robot = self.unwrapped.scene["robot"]
    self._robot = robot
    joint_names = robot.joint_names
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    missing_j = [n for n in _AMP_JOINT_NAMES if n not in name_to_idx]
    assert not missing_j, f"AMP joints missing in robot: {missing_j}"
    self._amp_joint_ids = torch.tensor(
      [name_to_idx[n] for n in _AMP_JOINT_NAMES],
      device=self.device,
      dtype=torch.long,
    )
    amp_body_names = _AMP_EE_BODIES + _select_amp_foot_bodies(robot.body_names)
    body_ids, _ = robot.find_bodies(list(amp_body_names), preserve_order=True)
    self._amp_body_ids = torch.tensor(body_ids, device=self.device, dtype=torch.long)
    self._hand_local_vec = torch.tensor(
      _HAND_LOCAL_OFFSET, device=self.device, dtype=torch.float32
    ).expand(self.num_envs, 3)

    # AMP loaders (positives) — populated after env construction.
    self.amp_loader_dict: dict[int, Any] = {}

    # Initial fill of the history buffer with the current observation.
    self._actor_history[:] = actor_obs.detach().to(self._actor_history.dtype)[
      :, None, :
    ]
    self._critic_history[:] = critic_obs.detach().to(self._critic_history.dtype)[
      :, None, :
    ]

    # Track latest one-step critic obs (HIM supervision + terminal critic obs).
    self._last_critic_obs = critic_obs.detach().clone()

  @property
  def latest_critic_obs(self) -> torch.Tensor:
    """Latest one-step privileged critic observation (for HIM supervision)."""
    return self._last_critic_obs

  def _stack_critic_obs(self, critic_obs: torch.Tensor) -> torch.Tensor:
    """Return critic obs with temporal stacking (hist10 parity when length > 1)."""
    if self.critic_history_length <= 1:
      return critic_obs
    return self._critic_history.reshape(self.num_envs, -1)

  def _roll_critic_history(
    self, critic_obs: torch.Tensor, reset_ids: torch.Tensor
  ) -> None:
    if self.critic_history_length <= 1:
      return
    self._critic_history = torch.roll(self._critic_history, shifts=-1, dims=1)
    self._critic_history[:, -1, :] = critic_obs
    if reset_ids.numel() > 0:
      self._critic_history[reset_ids] = critic_obs[reset_ids][:, None, :]

  # ------------------------------------------------------------------
  # Episode length buffer must read/write the underlying env so that
  # ``init_at_random_ep_len`` works (runner sets ``self.env.episode_length_buf``).
  # ------------------------------------------------------------------
  @property
  def episode_length_buf(self) -> torch.Tensor:
    return self.unwrapped.episode_length_buf

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor) -> None:
    self.unwrapped.episode_length_buf = value

  # ------------------------------------------------------------------
  # MultiAmpOnPolicyRunner contract.
  # ------------------------------------------------------------------
  def get_observations(self) -> tuple[torch.Tensor, dict]:
    actor_hist = self._actor_history.reshape(self.num_envs, -1)
    critic_out = self._stack_critic_obs(self._last_critic_obs)
    extras = {"observations": {"critic": critic_out}}
    return actor_hist, extras

  def step(
    self, actions: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    obs_td, rewards, dones, extras = self._base.step(actions)
    actor_obs = obs_td["actor"]
    critic_obs = obs_td["critic"]

    # Reset env bookkeeping (must be set BEFORE reading next obs in runner).
    reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
    self.reset_env_ids = reset_ids

    # Termination critic obs — runner indexes ``next_critic_obs[reset_ids] =
    # term``, so we hand back ONLY the rows for envs that just reset, with the
    # pre-reset (terminal-state) one-step critic observation.
    extras = dict(extras)
    obs_extras = dict(extras.get("observations", {}))
    if reset_ids.numel() > 0:
      obs_extras["termination_critic_obs"] = self._last_critic_obs[reset_ids]

    # Roll history: shift left and write the new frame at the end. For envs
    # that just reset, fill the history with the new (post-reset) obs.
    self._actor_history = torch.roll(self._actor_history, shifts=-1, dims=1)
    self._actor_history[:, -1, :] = actor_obs
    if reset_ids.numel() > 0:
      self._actor_history[reset_ids] = actor_obs[reset_ids][:, None, :]
    self._roll_critic_history(critic_obs, reset_ids)

    self._last_critic_obs = critic_obs.detach().clone()
    obs_extras["critic"] = self._stack_critic_obs(critic_obs)
    extras["observations"] = obs_extras
    self.last_episode_age = torch.where(
      dones.bool(),
      torch.zeros_like(self.last_episode_age),
      self.last_episode_age + 1,
    )

    # NOTE: ``env.style_ids`` is updated by the ``ultra_style_update`` step
    # event inside the underlying ManagerBasedRlEnv so reward functions read
    # the latest style. The wrapper exposes it as a property below for the
    # runner / AMP loaders.

    actor_hist = self._actor_history.reshape(self.num_envs, -1)
    return actor_hist, rewards, dones, extras

  def close(self) -> None:
    self._base.close()

  def seed(self, seed: int = -1) -> int:
    return self._base.seed(seed)

  # ------------------------------------------------------------------
  # AMP / style hooks.
  # ------------------------------------------------------------------
  @property
  def style_ids(self) -> torch.Tensor:
    """Per-env style id (stand=0/walk=1/run=2). Owned by the underlying
    env's ``ultra_style_update`` step event."""
    return self.unwrapped.style_ids

  @style_ids.setter
  def style_ids(self, value: torch.Tensor) -> None:
    self.unwrapped.style_ids = value

  def get_amp_obs_for_expert_trans(self) -> torch.Tensor:
    """Return shape ``(B, 38)`` AMP feature vector aligned with the motion files."""
    data = self._robot.data
    joint_pos = data.joint_pos[:, self._amp_joint_ids]
    joint_vel = data.joint_vel[:, self._amp_joint_ids]

    body_pos_w = data.body_link_pos_w[:, self._amp_body_ids, :]  # (B, 4, 3)
    body_quat_w = data.body_link_quat_w[:, self._amp_body_ids, :]  # (B, 4, 4)
    root_pos_w = data.root_link_pos_w  # (B, 3)
    root_quat_w = data.root_link_quat_w  # (B, 4)
    root_pos = root_pos_w[:, None, :]
    root_quat_inv = _quat_conj(root_quat_w)

    # Hand proxies = shoulder body + (R_shoulder * local_z_neg).
    lhand_w = body_pos_w[:, 0, :] + _quat_apply(
      body_quat_w[:, 0, :], self._hand_local_vec
    )
    rhand_w = body_pos_w[:, 1, :] + _quat_apply(
      body_quat_w[:, 1, :], self._hand_local_vec
    )
    lfoot_w = body_pos_w[:, 2, :]
    rfoot_w = body_pos_w[:, 3, :]

    rel = torch.stack([lhand_w, rhand_w, lfoot_w, rfoot_w], dim=1) - root_pos
    root_quat_inv_b = root_quat_inv[:, None, :].expand(self.num_envs, 4, 4)
    rel_b = _quat_apply(root_quat_inv_b, rel)
    return torch.cat([joint_pos, joint_vel, rel_b.reshape(self.num_envs, -1)], dim=-1)

  def get_amp_observations_pos(self) -> torch.Tensor:
    """Sample positives from per-style AMP loaders (for WGAN)."""
    if not self.amp_loader_dict:
      raise RuntimeError(
        "amp_loader_dict not populated; the runner should set it after construction."
      )
    times = (self.episode_length_buf * self.step_dt).cpu().numpy()
    amp_dim = next(iter(self.amp_loader_dict.values())).trajectories[0].shape[1]
    expert = torch.zeros(
      self.num_envs, amp_dim, device=self.device, dtype=torch.float32
    )
    style_np = self.style_ids.cpu().numpy()
    for style_id, loader in self.amp_loader_dict.items():
      mask = style_np == style_id
      n = int(mask.sum())
      if n == 0:
        continue
      traj_idxs = loader.weighted_traj_idx_sample_batch(n)
      frames = loader.get_frame_at_time_batch(traj_idxs, times[mask])
      expert[torch.from_numpy(mask).to(self.device)] = frames.to(
        dtype=torch.float32, device=self.device
      )
    return expert


# ---------------------------------------------------------------------------
# Custom runner.
# ---------------------------------------------------------------------------


class UltraGameYawAMPHIMRunner(MultiAmpOnPolicyRunner):
  """MultiAmpOnPolicyRunner that wraps a mjlab :class:`RslRlVecEnvWrapper`."""

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    history_length = int(train_cfg.pop("actor_obs_history_length", 10))
    critic_history_length = int(train_cfg.pop("critic_obs_history_length", 1))
    style_cfg = train_cfg.pop(
      "style_thresholds",
      {
        "stand_enter": 0.08,
        "stand_exit": 0.12,
        "run_enter": 1.05,
        "run_exit": 0.85,
      },
    )
    num_styles = int(train_cfg.pop("num_styles", 3))
    train_cfg.pop("clip_actions", None)
    train_cfg.pop("upload_model", None)
    train_cfg.pop("wandb_tags", None)
    # NOTE: keep "wandb_project" in train_cfg so WandbSummaryWriter can read it.
    train_cfg.pop("neptune_project", None)
    train_cfg.pop("load_run", None)
    train_cfg.pop("load_checkpoint", None)
    train_cfg.pop("resume", None)

    amp_env = UltraAMPHIMVecEnvWrapper(
      env,
      history_length=history_length,
      critic_history_length=critic_history_length,
      num_styles=num_styles,
      style_thresholds=style_cfg,
    )

    super().__init__(amp_env, train_cfg, log_dir, device)
    # The base runner already constructed AMPLoaders into self.amp_data_dict;
    # publish them on the env so ``get_amp_observations_pos`` can sample.
    amp_env.amp_loader_dict = {
      int(k.split("_", 1)[1]): v for k, v in self.amp_data_dict.items()
    }

  def save(self, path: str, infos=None) -> None:
    """Save checkpoints locally but skip uploading them to W&B.

    The base runner uploads every checkpoint to W&B when ``logger_type`` is
    ``wandb``. We keep local checkpoints and all scalar/metric logging (which
    goes through ``writer.add_scalar`` independent of ``logger_type``) but
    avoid the per-checkpoint upload by masking ``logger_type`` during save.
    """
    saved_logger_type = self.logger_type
    self.logger_type = "tensorboard"
    try:
      super().save(path, infos)
    finally:
      self.logger_type = saved_logger_type


# ---------------------------------------------------------------------------
# Runner config dataclass.
# ---------------------------------------------------------------------------


def _default_amp_motion_files_dict() -> dict[str, list[str]]:
  return {k: list(v) for k, v in ULTRA_AMP_MOTION_FILES.items()}


def _default_amp_reward_coef() -> dict[str, float]:
  return {"style_0": 1.0, "style_1": 1.0, "style_2": 0.6}


def _default_policy_cfg() -> dict[str, Any]:
  return {
    "class_name": "ActorCritic",
    "init_noise_std": 1.0,
    "noise_std_type": "scalar",
    "actor_hidden_dims": [512, 256, 128],
    "critic_hidden_dims": [512, 256, 128],
    "activation": "elu",
    "actor_recent_frames": 5,
    "him_encoder_type": "gru",
    "him_gru_hidden_dim": 64,
    "him_gru_num_layers": 1,
    "him_gru_head_hidden_dims": [64],
  }


def _default_algorithm_cfg() -> dict[str, Any]:
  from .ultra_mdp import ultra_data_augmentation_func

  return {
    "class_name": "MULTIAMPPPO",
    "value_loss_coef": 1.0,
    "use_clipped_value_loss": True,
    "clip_param": 0.2,
    "entropy_coef": 0.005,
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
    "learning_rate": 1.0e-3,
    "schedule": "adaptive",
    "gamma": 0.99,
    "lam": 0.95,
    "desired_kl": 0.01,
    "max_grad_norm": 1.0,
    "normalize_advantage_per_mini_batch": False,
    # HoST-style smoothness loss (default OFF via lower_bound=0.0; V5 overrides
    # these). Keeps the keys present so they're serialized to params/wandb.
    "smoothness_upper_bound": 1.0,
    "smoothness_lower_bound": 0.0,
    "value_smoothness_coef": 0.1,
    "rnd_cfg": None,
    "symmetry_cfg": {
      "use_data_augmentation": True,
      "use_mirror_loss": True,
      "mirror_loss_coeff": 1.0,
      "data_augmentation_func": ultra_data_augmentation_func,
    },
  }


@dataclass
class RslRlAmpHimRunnerCfg(RslRlBaseRunnerCfg):
  """Runner config for the Ultra GameYaw AMP+HIM training task.

  Mirrors ``UltraGameYawMotionRunAgentCfg`` from the Ultra Isaac Lab task.
  Extra fields beyond :class:`RslRlBaseRunnerCfg` are read by
  :class:`UltraGameYawAMPHIMRunner` from the ``asdict`` payload.
  """

  empirical_normalization: bool = False
  use_wgan_discriminator: bool = True
  amp_motion_files_dict: dict[str, list[str]] = field(
    default_factory=_default_amp_motion_files_dict
  )
  amp_reward_coef_dict: dict[str, float] = field(
    default_factory=_default_amp_reward_coef
  )
  amp_num_preload_transitions: int = 200_000
  amp_task_reward_lerp: float = 0.7
  amp_discr_hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
  min_normalized_std: list[float] = field(default_factory=lambda: [0.05] * 13)
  policy: dict[str, Any] = field(default_factory=_default_policy_cfg)
  algorithm: dict[str, Any] = field(default_factory=_default_algorithm_cfg)
  num_styles: int = 3
  actor_obs_history_length: int = 10
  critic_obs_history_length: int = 1
  """Privileged critic temporal stack depth. hist10 uses 10; older mjlab
  AMP+HIM tasks keep 1 (single-frame critic) for backward compatibility."""
  style_thresholds: dict[str, float] = field(
    default_factory=lambda: {
      "stand_enter": 0.08,
      "stand_exit": 0.12,
      "run_enter": 1.05,
      "run_exit": 0.85,
    }
  )


def ultra_game_yaw_amp_him_runner_cfg() -> RslRlAmpHimRunnerCfg:
  """Factory matching ``UltraGameYawMotionRunAgentCfg`` defaults."""
  return RslRlAmpHimRunnerCfg(
    seed=42,
    num_steps_per_env=32,
    max_iterations=20_000,
    save_interval=200,
    experiment_name="ultra_game_yaw_amp_him",
    run_name="",
    # `wandb` logs to Weights & Biases AND writes local TensorBoard tfevents
    # (WandbSummaryWriter subclasses torch's SummaryWriter), so both are on.
    logger="wandb",
    wandb_project="ultra_game_yaw_amp_him",
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
  )


# ---------------------------------------------------------------------------
# Parallel "accel" variant: adds a 4th AMP style fed by the stand-to-run expert
# clip to shape the start/acceleration phase. Reuses everything else (rewards,
# HIM, symmetry) from the baseline runner. The baseline factory above is left
# untouched so both tasks train in parallel.
# ---------------------------------------------------------------------------

# Stand-to-run launch clip, converted from the GMR retarget via
# ``scripts/convert_gmr_pkl_to_amp_txt.py``.
ULTRA_ACCEL_MOTION_FILE: str = str(_AMP_DIR / "accel_stand_to_run.txt")


def ultra_game_yaw_amp_him_accel_runner_cfg() -> RslRlAmpHimRunnerCfg:
  """Runner cfg for the accel variant: baseline + a 4th (launch) AMP style."""
  cfg = ultra_game_yaw_amp_him_runner_cfg()
  cfg.num_styles = 4
  cfg.amp_motion_files_dict = {
    **_default_amp_motion_files_dict(),
    "style_3": [ULTRA_ACCEL_MOTION_FILE],
  }
  # Launch style weighted a bit below stand/walk but above steady run, so the
  # accel discriminator pulls firmly during the (short) launch window.
  cfg.amp_reward_coef_dict = {
    **_default_amp_reward_coef(),
    "style_3": 0.8,
  }
  cfg.experiment_name = "ultra_game_yaw_amp_him_accel"
  cfg.wandb_project = "ultra_game_yaw_amp_him_accel"
  return cfg
