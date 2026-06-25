# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch

from mjlab.third_party import amp_rsl_rl as rsl_rl
from mjlab.third_party.amp_rsl_rl.algorithms import MULTIAMPPPO
from mjlab.third_party.amp_rsl_rl.env import VecEnv
from mjlab.third_party.amp_rsl_rl.modules import (
  ActorCritic,
  ActorCriticRecurrent,
  Discriminator,
  EmpiricalNormalization,
  StudentTeacher,
  StudentTeacherRecurrent,
  WGANDiscriminator,
)
from mjlab.third_party.amp_rsl_rl.utils import (
  AMPLoader,
  Normalizer,
  store_code_state,
)


def resolve_lin_vel_obs_scale(env: VecEnv) -> float:
  """Critic yaw-frame linear velocity scale (matches env obs normalization)."""
  unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
  if hasattr(unwrapped, "obs_scales"):
    scale = getattr(unwrapped.obs_scales, "lin_vel", None)
    if scale is not None:
      return float(scale)
  cfg = getattr(unwrapped, "cfg", None)
  if cfg is not None:
    norm = getattr(cfg, "normalization", None)
    if norm is not None:
      obs_scales = getattr(norm, "obs_scales", None)
      if obs_scales is not None:
        scale = getattr(obs_scales, "lin_vel", None)
        if scale is not None:
          return float(scale)
  return 1.0


def replace_reset_amp_obs_with_terminal(
  next_amp_obs: torch.Tensor,
  reset_env_ids: torch.Tensor,
  terminal_amp_obs: torch.Tensor,
) -> torch.Tensor:
  """Return next AMP obs with reset rows replaced by terminal AMP obs.

  mjlab auto-resets before computing post-step observations, so reset rows in
  ``next_amp_obs`` belong to the next episode. The previous rollout AMP obs is
  the best available terminal-state approximation, matching AMP_mjlab's G1
  runner behavior.
  """

  next_amp_obs_with_term = torch.clone(next_amp_obs)
  if reset_env_ids.numel() > 0:
    next_amp_obs_with_term[reset_env_ids] = terminal_amp_obs[reset_env_ids]
  return next_amp_obs_with_term


class MultiAmpOnPolicyRunner:
  """On-policy runner for training and evaluation."""

  def __init__(
    self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"
  ):
    self.cfg = train_cfg
    self.alg_cfg = train_cfg["algorithm"]
    self.policy_cfg = train_cfg["policy"]
    self.device = device
    self.env = env

    # check if multi-gpu is enabled
    self._configure_multi_gpu()

    # resolve training type depending on the algorithm
    if self.alg_cfg["class_name"] in ["PPO", "AMPPPO", "MULTIAMPPPO"]:
      self.training_type = "rl"
    elif self.alg_cfg["class_name"] == "Distillation":
      self.training_type = "distillation"
    else:
      raise ValueError(
        f"Training type not found for algorithm {self.alg_cfg['class_name']}."
      )

    # resolve dimensions of observations
    obs, extras = self.env.get_observations()
    num_obs = obs.shape[1]
    self.num_one_step_obs = getattr(self.env, "num_one_step_obs", None)
    self.num_one_step_critic_obs = getattr(self.env, "num_one_step_critic_obs", None)
    if self.num_one_step_obs is not None:
      history_length = int(num_obs // self.num_one_step_obs)
    else:
      history_length = None

    # resolve type of privileged observations
    if self.training_type == "rl":
      if "critic" in extras["observations"]:
        self.privileged_obs_type = (
          "critic"  # actor-critic reinforcement learnig, e.g., PPO
        )
      else:
        self.privileged_obs_type = None
    if self.training_type == "distillation":
      if "teacher" in extras["observations"]:
        self.privileged_obs_type = "teacher"  # policy distillation
      else:
        self.privileged_obs_type = None

    # resolve dimensions of privileged observations
    if self.privileged_obs_type is not None:
      num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
    else:
      num_privileged_obs = num_obs

    # evaluate the policy class
    policy_class = eval(self.policy_cfg.pop("class_name"))
    if self.num_one_step_obs is not None:
      vel_index = getattr(
        self.env, "him_vel_index_in_critic", self.num_one_step_obs + 1
      )
      vel_scale = resolve_lin_vel_obs_scale(self.env)
      policy: (
        ActorCritic | ActorCriticRecurrent | StudentTeacher | StudentTeacherRecurrent
      ) = policy_class(
        num_obs,
        num_privileged_obs,
        self.env.num_actions,
        **{
          **self.policy_cfg,
          "num_one_step_obs": self.num_one_step_obs,
          "num_one_step_critic_obs": self.num_one_step_critic_obs,
          "history_length": history_length,
          "vel_index_in_critic": vel_index,
          "vel_scale_for_actor": vel_scale,
        },
      ).to(self.device)
    else:
      policy: (
        ActorCritic | ActorCriticRecurrent | StudentTeacher | StudentTeacherRecurrent
      ) = policy_class(
        num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg
      ).to(self.device)

    # resolve dimension of rnd gated state
    if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
      # check if rnd gated state is present
      rnd_state = extras["observations"].get("rnd_state")
      if rnd_state is None:
        raise ValueError(
          "Observations for the key 'rnd_state' not found in infos['observations']."
        )
      # get dimension of rnd gated state
      num_rnd_state = rnd_state.shape[1]
      # add rnd gated state to config
      self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
      # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
      self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

    # if using symmetry then pass the environment config object
    if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
      # this is used by the symmetry function for handling different observation terms
      self.alg_cfg["symmetry_cfg"]["_env"] = env

    # init multi amp loader
    self.amp_data_dict = {}
    self.amp_normalizer_dict = {}
    self.discriminator_dict = {}

    motion_files_dict = train_cfg["amp_motion_files_dict"]
    reward_coef_dict = train_cfg["amp_reward_coef_dict"]
    self.use_wgan = train_cfg.get("use_wgan_discriminator", False)

    for style_id in range(self.env.num_styles):
      style_key = f"style_{style_id}"
      amp_data = AMPLoader(
        device,
        time_between_frames=self.env.step_dt,
        preload_transitions=True,
        num_preload_transitions=train_cfg["amp_num_preload_transitions"],
        motion_files=motion_files_dict[style_key],
      )
      amp_normalizer = Normalizer(amp_data.observation_dim)
      if self.use_wgan:
        discriminator = WGANDiscriminator(
          amp_data.observation_dim * 2,
          reward_coef_dict[style_key],
          train_cfg["amp_discr_hidden_dims"],
          device,
          train_cfg["amp_task_reward_lerp"],
        ).to(self.device)
      else:
        discriminator = Discriminator(
          amp_data.observation_dim * 2,
          reward_coef_dict[style_key],
          train_cfg["amp_discr_hidden_dims"],
          device,
          train_cfg["amp_task_reward_lerp"],
        ).to(self.device)

      self.amp_data_dict[style_key] = amp_data
      self.amp_normalizer_dict[style_key] = amp_normalizer
      self.discriminator_dict[style_key] = discriminator

    min_std = torch.tensor(
      train_cfg["min_normalized_std"],
      device=self.device,
      dtype=torch.float32,
      requires_grad=False,
    )
    if min_std.numel() != self.env.num_actions:
      raise ValueError(
        "min_normalized_std must have one value per action: "
        f"got {min_std.numel()} values for {self.env.num_actions} actions"
      )

    # initialize algorithm
    alg_class = eval(self.alg_cfg.pop("class_name"))
    self.alg: MULTIAMPPPO = alg_class(
      policy,
      self.discriminator_dict,
      self.amp_data_dict,
      self.amp_normalizer_dict,
      device=self.device,
      min_std=min_std,
      **self.alg_cfg,
      multi_gpu_cfg=self.multi_gpu_cfg,
    )

    # store training configuration
    self.num_steps_per_env = self.cfg["num_steps_per_env"]
    self.save_interval = self.cfg["save_interval"]
    self.empirical_normalization = self.cfg["empirical_normalization"]
    if self.empirical_normalization:
      self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(
        self.device
      )
      self.privileged_obs_normalizer = EmpiricalNormalization(
        shape=[num_privileged_obs], until=1.0e8
      ).to(self.device)
    else:
      self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
      self.privileged_obs_normalizer = torch.nn.Identity().to(
        self.device
      )  # no normalization

    # init storage and model
    self.alg.init_storage(
      self.training_type,
      self.env.num_envs,
      self.num_steps_per_env,
      [num_obs],
      [num_privileged_obs],
      [self.env.num_actions],
      next_critic_obs_shape=(
        [self.num_one_step_critic_obs]
        if self.num_one_step_critic_obs is not None
        else None
      ),
    )

    # Decide whether to disable logging
    # We only log from the process with rank 0 (main process)
    self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
    # Logging
    self.log_dir = log_dir
    self.logger_type = self.cfg.get("logger", "tensorboard").lower()
    self.writer = None
    self.tot_timesteps = 0
    self.tot_time = 0
    self.current_learning_iteration = 0
    self.git_status_repos = [rsl_rl.__file__]

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
    # initialize writer
    if self.log_dir is not None and self.writer is None and not self.disable_logs:
      # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
      self.logger_type = self.cfg.get("logger", "tensorboard")
      self.logger_type = self.logger_type.lower()

      if self.logger_type == "neptune":
        from mjlab.third_party.amp_rsl_rl.utils.neptune_utils import (
          NeptuneSummaryWriter,
        )

        self.writer = NeptuneSummaryWriter(
          log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
        )
        self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
      elif self.logger_type == "wandb":
        from mjlab.third_party.amp_rsl_rl.utils.wandb_utils import WandbSummaryWriter

        self.writer = WandbSummaryWriter(
          log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
        )
        self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
      elif self.logger_type == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
      else:
        raise ValueError(
          "Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'."
        )

    # check if teacher is loaded
    if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
      raise ValueError(
        "Teacher model parameters not loaded. Please load a teacher model to distill."
      )

    # randomize initial episode lengths (for exploration)
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )
      if hasattr(self.env, "last_episode_age"):
        self.env.last_episode_age.fill_(int(self.env.max_episode_length))

    # start learning
    obs, extras = self.env.get_observations()
    privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
    amp_obs = self.env.get_amp_obs_for_expert_trans()
    amp_obs_pos = self.env.get_amp_observations_pos() if self.use_wgan else None
    obs, privileged_obs, amp_obs = (
      obs.to(self.device),
      privileged_obs.to(self.device),
      amp_obs.to(self.device),
    )
    if self.use_wgan:
      amp_obs_pos = amp_obs_pos.to(self.device)  # type: ignore
    self.train_mode()  # switch to train mode (for dropout for example)

    # Book keeping
    ep_infos = []
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(
      self.env.num_envs, dtype=torch.float, device=self.device
    )
    cur_episode_length = torch.zeros(
      self.env.num_envs, dtype=torch.float, device=self.device
    )

    # AMP reward decomposition logging: the per-style discriminators share the
    # same task<->style lerp, so read it once. total = (1-lerp)*disc + lerp*task.
    amp_lerp = float(next(iter(self.alg.discriminator_dict.values())).task_reward_lerp)

    # create buffers for logging extrinsic and intrinsic rewards
    if self.alg.rnd:
      erewbuffer = deque(maxlen=100)
      irewbuffer = deque(maxlen=100)
      cur_ereward_sum = torch.zeros(
        self.env.num_envs, dtype=torch.float, device=self.device
      )
      cur_ireward_sum = torch.zeros(
        self.env.num_envs, dtype=torch.float, device=self.device
      )

    # Ensure all parameters are in-synced (all ranks must reach barrier before broadcast)
    if self.is_distributed:
      from mjlab.third_party.amp_rsl_rl.utils import distributed_barrier

      distributed_barrier("pre_learn_broadcast")
      print(f"[rank {self.gpu_global_rank}] Synchronizing parameters...", flush=True)
      self.alg.broadcast_parameters()
      distributed_barrier("post_learn_broadcast")

    # Start training
    start_iter = self.current_learning_iteration
    tot_iter = start_iter + num_learning_iterations
    for it in range(start_iter, tot_iter):
      start = time.time()
      # Per-iteration AMP reward accounting (GPU-side; one sync at log time).
      amp_task_sum = torch.zeros((), device=self.device)
      amp_total_sum = torch.zeros((), device=self.device)
      amp_disc_sum = torch.zeros(self.env.num_styles, device=self.device)
      amp_disc_cnt = torch.zeros(self.env.num_styles, device=self.device)
      amp_env_steps = 0
      # Rollout
      with torch.inference_mode():
        for _ in range(self.num_steps_per_env):
          # Sample actions
          actions = self.alg.act(obs, privileged_obs, amp_obs)
          # Step the environment
          obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
          next_amp_obs = self.env.get_amp_obs_for_expert_trans()
          next_amp_obs_pos = (
            self.env.get_amp_observations_pos() if self.use_wgan else None
          )
          # Move to device
          obs, rewards, dones, next_amp_obs = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
            next_amp_obs.to(self.device),
          )
          if self.use_wgan:
            next_amp_obs_pos = next_amp_obs_pos.to(self.device)  # type: ignore
          # perform normalization
          raw_privileged_obs = None
          if self.privileged_obs_type is not None:
            raw_privileged_obs = infos["observations"][self.privileged_obs_type].to(
              self.device
            )
          obs = self.obs_normalizer(obs)
          if self.privileged_obs_type is not None:
            privileged_obs = self.privileged_obs_normalizer(raw_privileged_obs)
          else:
            privileged_obs = obs

          next_critic_obs = None
          if self.num_one_step_critic_obs is not None:
            # Estimator update uses raw (un-normalized) one-step next critic obs.
            latest = getattr(self.env, "latest_critic_obs", None)
            if latest is not None:
              next_critic_obs = latest.clone().detach()
            else:
              src = (
                raw_privileged_obs if raw_privileged_obs is not None else privileged_obs
              )
              if src.shape[1] > self.num_one_step_critic_obs:
                next_critic_obs = (
                  src[:, -self.num_one_step_critic_obs :].clone().detach()
                )
              else:
                next_critic_obs = (
                  src[:, : self.num_one_step_critic_obs].clone().detach()
                )
            reset_env_ids = self.env.reset_env_ids
            if len(reset_env_ids) > 0 and "termination_critic_obs" in infos.get(
              "observations", {}
            ):
              term = infos["observations"]["termination_critic_obs"].to(self.device)
              next_critic_obs[reset_env_ids] = term.clone().detach()

          # Account for terminal state transitions. mjlab auto-resets before
          # computing observations, so reset rows in next_amp_obs are post-reset.
          reset_env_ids = self.env.reset_env_ids
          next_amp_obs_with_term = replace_reset_amp_obs_with_terminal(
            next_amp_obs=next_amp_obs,
            reset_env_ids=reset_env_ids,
            terminal_amp_obs=amp_obs,
          )

          # Raw task reward (env reward manager output, pre-AMP). dt-scaled.
          task_reward_step = rewards.detach()
          amp_task_sum += task_reward_step.sum()
          amp_env_steps += int(task_reward_step.numel())

          total_rewards = torch.zeros_like(rewards)
          for style_id in range(self.env.num_styles):
            idxs = (self.env.style_ids == style_id).nonzero(as_tuple=True)[0]
            if len(idxs) == 0:
              continue
            style_key = f"style_{style_id}"
            style_amp_obs = amp_obs[idxs]
            style_amp_obs_pos = amp_obs_pos[idxs] if self.use_wgan else None  # type: ignore
            style_next_amp_obs = next_amp_obs_with_term[idxs]
            style_next_amp_obs_pos = next_amp_obs_pos[idxs] if self.use_wgan else None  # type: ignore
            task_rewards = rewards[idxs]
            if self.use_wgan:
              total_reward = self.alg.discriminator_dict[
                style_key
              ].predict_amp_reward_WGAN(
                style_amp_obs,
                style_next_amp_obs,
                style_amp_obs_pos,  # type: ignore[arg-type]
                style_next_amp_obs_pos,  # type: ignore[arg-type]
                task_rewards,
                normalizer=self.alg.amp_normalizer_dict[style_key],
              )[0]
            else:
              total_reward = self.alg.discriminator_dict[style_key].predict_amp_reward(
                style_amp_obs,
                style_next_amp_obs,
                task_rewards,
                normalizer=self.alg.amp_normalizer_dict[style_key],
              )[0]
            total_rewards[idxs] = total_reward
            # Back out the pure discriminator (style) reward for logging:
            # total = (1-lerp)*disc + lerp*task  =>  disc = (total - lerp*task)/(1-lerp).
            if amp_lerp < 1.0:
              disc_sub = (total_reward - amp_lerp * task_rewards) / (1.0 - amp_lerp)
            else:
              disc_sub = torch.zeros_like(total_reward)
            amp_disc_sum[style_id] += disc_sub.sum()
            amp_disc_cnt[style_id] += float(len(idxs))

          rewards = total_rewards
          amp_total_sum += rewards.sum()

          amp_obs = torch.clone(next_amp_obs)
          if self.use_wgan:
            amp_obs_pos = torch.clone(next_amp_obs_pos)  # type: ignore
          self.alg.process_env_step(
            rewards,
            dones,
            infos,
            next_amp_obs_with_term,
            next_critic_obs,
            style_ids=self.env.style_ids.to(self.device),
          )

          # Extract intrinsic rewards (only for logging)
          intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

          # book keeping
          if self.log_dir is not None:
            if "episode" in infos:
              ep_infos.append(infos["episode"])
            elif "log" in infos:
              ep_infos.append(infos["log"])
            # Update rewards
            if self.alg.rnd:
              cur_ereward_sum += rewards
              cur_ireward_sum += intrinsic_rewards  # type: ignore
              cur_reward_sum += rewards + intrinsic_rewards
            else:
              cur_reward_sum += rewards
            # Update episode length
            cur_episode_length += 1
            # Clear data for completed episodes
            # -- common
            new_ids = (dones > 0).nonzero(as_tuple=False)
            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0
            # -- intrinsic and extrinsic rewards
            if self.alg.rnd:
              erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
              irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
              cur_ereward_sum[new_ids] = 0
              cur_ireward_sum[new_ids] = 0

        stop = time.time()
        collection_time = stop - start
        start = stop

        # compute returns
        if self.training_type == "rl":
          self.alg.compute_returns(privileged_obs)

      # update policy
      loss_dict = self.alg.update()

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it
      # log info (all ranks participate for distributed reductions; only rank0 writes)
      if self.log_dir is not None:
        self.log(locals())
        # Save model (rank0 only)
        if not self.disable_logs and it % self.save_interval == 0:
          self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

      # Clear episode infos
      ep_infos.clear()
      # Save code state
      if it == start_iter and not self.disable_logs:
        # obtain all the diff files
        git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
        # if possible store them to wandb
        if self.logger_type in ["wandb", "neptune"] and git_file_paths:
          for path in git_file_paths:
            self.writer.save_file(path)

    # Save the final model after training
    if self.log_dir is not None and not self.disable_logs:
      self.save(
        os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt")
      )

  def log(self, locs: dict, width: int = 80, pad: int = 35):
    # Compute the collection size
    collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
    # Update total time-steps and time
    self.tot_timesteps += collection_size
    self.tot_time += locs["collection_time"] + locs["learn_time"]
    iteration_time = locs["collection_time"] + locs["learn_time"]

    # -- Episode info
    # Avoid object collectives and per-key reductions here. They are not
    # needed for training and can hang long multi-GPU runs. Rank 0 logs its
    # local episode samples only.
    ep_string = ""
    keys: list[str] = []
    if not self.disable_logs:
      local_keys: set[str] = set()
      for ep_info in locs.get("ep_infos", []):
        local_keys.update(ep_info.keys())
      keys = sorted(local_keys)

    for key in keys:
      infotensor = torch.tensor([], device=self.device)
      for ep_info in locs.get("ep_infos", []):
        # handle scalar and zero dimensional tensor infos
        if key not in ep_info:
          continue
        if not isinstance(ep_info[key], torch.Tensor):
          ep_info[key] = torch.Tensor([ep_info[key]])
        if len(ep_info[key].shape) == 0:
          ep_info[key] = ep_info[key].unsqueeze(0)
        infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
      local_n = int(infotensor.numel())
      value = (
        torch.mean(infotensor) if local_n > 0 else torch.tensor(0.0, device=self.device)
      )
      if "/" in key:
        self.writer.add_scalar(key, value, locs["it"])
        ep_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
      else:
        self.writer.add_scalar("Episode/" + key, value, locs["it"])
        ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

    mean_std = self.alg.policy.action_std.mean()
    fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

    # -- Losses
    if not self.disable_logs:
      for key, value in locs["loss_dict"].items():
        self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
      self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

    # -- Policy
    if not self.disable_logs:
      self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

    # -- Performance
    if not self.disable_logs:
      self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
      self.writer.add_scalar(
        "Perf/collection time", locs["collection_time"], locs["it"]
      )
      self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

    # -- Training
    # Keep logging rank-local. Logging collectives are not required for
    # optimization and can desynchronize when one rank has different episode
    # bookkeeping at long horizons.
    local_rew_n = len(locs["rewbuffer"])
    if local_rew_n > 0:
      local_rew_mean = torch.tensor(
        statistics.mean(locs["rewbuffer"]), device=self.device
      )
      local_len_mean = torch.tensor(
        statistics.mean(locs["lenbuffer"]), device=self.device
      )
    else:
      local_rew_mean = torch.tensor(0.0, device=self.device)
      local_len_mean = torch.tensor(0.0, device=self.device)

    if not self.disable_logs and (local_rew_n > 0 or self.is_distributed):
      if self.alg.rnd:
        erew_mean = (
          torch.tensor(statistics.mean(locs["erewbuffer"]), device=self.device)
          if len(locs["erewbuffer"]) > 0
          else torch.tensor(0.0, device=self.device)
        )
        irew_mean = (
          torch.tensor(statistics.mean(locs["irewbuffer"]), device=self.device)
          if len(locs["irewbuffer"]) > 0
          else torch.tensor(0.0, device=self.device)
        )
        self.writer.add_scalar("Rnd/mean_extrinsic_reward", erew_mean, locs["it"])
        self.writer.add_scalar("Rnd/mean_intrinsic_reward", irew_mean, locs["it"])
        self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
      self.writer.add_scalar("Train/mean_reward", local_rew_mean, locs["it"])
      self.writer.add_scalar("Train/mean_episode_length", local_len_mean, locs["it"])
      if self.logger_type != "wandb":
        self.writer.add_scalar("Train/mean_reward/time", local_rew_mean, self.tot_time)
        self.writer.add_scalar(
          "Train/mean_episode_length/time", local_len_mean, self.tot_time
        )

    # -- AMP reward decomposition (task vs discriminator/style, per-step means).
    # Answers "how much is RL vs the AMP discriminator actually contributing".
    amp_env_steps = locs.get("amp_env_steps", 0)
    if not self.disable_logs and amp_env_steps > 0:
      n = float(amp_env_steps)
      lerp = locs["amp_lerp"]
      disc_sum = locs["amp_disc_sum"]
      disc_cnt = locs["amp_disc_cnt"]
      task_mean = (locs["amp_task_sum"] / n).item()
      total_mean = (locs["amp_total_sum"] / n).item()
      disc_mean_all = (disc_sum.sum() / disc_cnt.sum().clamp_min(1.0)).item()
      # Contributions that literally add up to total_mean.
      task_contrib = lerp * task_mean
      style_contrib = total_mean - task_contrib
      self.writer.add_scalar("AMP/task_reward_mean", task_mean, locs["it"])
      self.writer.add_scalar("AMP/disc_reward_mean", disc_mean_all, locs["it"])
      self.writer.add_scalar("AMP/total_reward_mean", total_mean, locs["it"])
      self.writer.add_scalar("AMP/task_contrib", task_contrib, locs["it"])
      self.writer.add_scalar("AMP/style_contrib", style_contrib, locs["it"])
      if abs(total_mean) > 1e-9:
        self.writer.add_scalar(
          "AMP/task_contrib_frac", task_contrib / total_mean, locs["it"]
        )
        self.writer.add_scalar(
          "AMP/style_contrib_frac", style_contrib / total_mean, locs["it"]
        )
      for sid in range(self.env.num_styles):
        c = disc_cnt[sid].item()
        self.writer.add_scalar(f"AMP/style_{sid}_env_frac", c / n, locs["it"])
        if c > 0:
          self.writer.add_scalar(
            f"AMP/style_{sid}_disc_reward",
            (disc_sum[sid] / disc_cnt[sid]).item(),
            locs["it"],
          )

    str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

    if local_rew_n > 0:
      log_string = (
        f"""{"#" * width}\n"""
        f"""{str.center(width, " ")}\n\n"""
        f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {
          locs["collection_time"]:.3f}s, learning {locs["learn_time"]:.3f}s)\n"""
        f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
      )
      # -- Losses
      for key, value in locs["loss_dict"].items():
        log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""
      # -- Rewards
      if self.alg.rnd:
        log_string += (
          f"""{"Mean extrinsic reward:":>{pad}} {statistics.mean(locs["erewbuffer"]):.2f}\n"""
          f"""{"Mean intrinsic reward:":>{pad}} {statistics.mean(locs["irewbuffer"]):.2f}\n"""
        )
      log_string += f"""{"Mean reward:":>{pad}} {local_rew_mean.item():.2f}\n"""
      # -- AMP task/style/total per-step decomposition
      if locs.get("amp_env_steps", 0) > 0:
        _n = float(locs["amp_env_steps"])
        _task = (locs["amp_task_sum"] / _n).item()
        _disc = (
          locs["amp_disc_sum"].sum() / locs["amp_disc_cnt"].sum().clamp_min(1.0)
        ).item()
        _tot = (locs["amp_total_sum"] / _n).item()
        log_string += (
          f"""{"Per-step task/disc/total:":>{pad}} """
          f"""{_task:.3f} / {_disc:.3f} / {_tot:.3f}\n"""
        )
      # -- episode info
      log_string += f"""{"Mean episode length:":>{pad}} {local_len_mean.item():.2f}\n"""
    else:
      log_string = (
        f"""{"#" * width}\n"""
        f"""{str.center(width, " ")}\n\n"""
        f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {
          locs["collection_time"]:.3f}s, learning {locs["learn_time"]:.3f}s)\n"""
        f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
      )
      for key, value in locs["loss_dict"].items():
        log_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""

    log_string += ep_string
    log_string += (
      f"""{"-" * width}\n"""
      f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
      f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
      f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
      f"""{"ETA:":>{pad}} {
        time.strftime(
          "%H:%M:%S",
          time.gmtime(
            self.tot_time
            / (locs["it"] - locs["start_iter"] + 1)
            * (locs["start_iter"] + locs["num_learning_iterations"] - locs["it"])
          ),
        )
      }\n"""
    )
    if not self.disable_logs:
      print(log_string)

  def save(self, path: str, infos=None):
    # -- Save model
    saved_dict = {
      "model_state_dict": self.alg.policy.state_dict(),
      "optimizer_state_dict": self.alg.optimizer.state_dict(),
      "discriminator_state_dict": {
        k: d.state_dict() for k, d in self.alg.discriminator_dict.items()
      },
      "iter": self.current_learning_iteration,
      "infos": infos,
    }
    # -- Save RND model if used
    if self.alg.rnd:
      saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
      saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
    # -- Save observation normalizer if used
    if self.empirical_normalization:
      saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
      saved_dict["privileged_obs_norm_state_dict"] = (
        self.privileged_obs_normalizer.state_dict()
      )
    if hasattr(self.alg.policy, "estimator") and self.alg.policy.estimator is not None:
      saved_dict["estimator_optimizer_state_dict"] = (
        self.alg.policy.estimator.optimizer.state_dict()
      )

    # save model
    torch.save(saved_dict, path)

    # upload model to external logging service
    if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
      self.writer.save_model(path, self.current_learning_iteration)

  def load(
    self,
    path: str,
    load_optimizer: bool = True,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | torch.device | None = None,
  ):
    """Load checkpoint with backward-compatible signature.

    Supports both:
    - legacy AMP API: ``load(path, load_optimizer=True)``
    - mjlab runner API: ``load(path, load_cfg=..., strict=..., map_location=...)``

    ``load_cfg`` and ``strict`` are accepted for API compatibility and ignored
    because this runner loads full policy/discriminator states.
    """
    del load_cfg, strict
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
    # -- Load model
    resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
    for k, state in loaded_dict["discriminator_state_dict"].items():
      self.alg.discriminator_dict[k].load_state_dict(state)
    # -- Load RND model if used
    if self.alg.rnd:
      self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
    # -- Load observation normalizer if used
    if self.empirical_normalization:
      if resumed_training:
        # if a previous training is resumed, the actor/student normalizer is loaded for the actor/student
        # and the critic/teacher normalizer is loaded for the critic/teacher
        self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        self.privileged_obs_normalizer.load_state_dict(
          loaded_dict["privileged_obs_norm_state_dict"]
        )
      else:
        # if the training is not resumed but a model is loaded, this run must be distillation training following
        # an rl training. Thus the actor normalizer is loaded for the teacher model. The student's normalizer
        # is not loaded, as the observation space could differ from the previous rl training.
        self.privileged_obs_normalizer.load_state_dict(
          loaded_dict["obs_norm_state_dict"]
        )
    if hasattr(self.alg.policy, "estimator") and self.alg.policy.estimator is not None:
      if "estimator_optimizer_state_dict" in loaded_dict:
        self.alg.policy.estimator.optimizer.load_state_dict(
          loaded_dict["estimator_optimizer_state_dict"]
        )
    # -- load optimizer if used
    if load_optimizer and resumed_training:
      # -- algorithm optimizer
      self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
      # -- RND optimizer if used
      if self.alg.rnd:
        self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
    # -- load current learning iteration
    if resumed_training:
      self.current_learning_iteration = loaded_dict["iter"]
    return loaded_dict["infos"]

  def get_inference_policy(self, device=None):
    self.eval_mode()  # switch to evaluation mode (dropout for example)
    if device is not None:
      self.alg.policy.to(device)
    policy = self.alg.policy.act_inference
    if self.cfg["empirical_normalization"]:
      if device is not None:
        self.obs_normalizer.to(device)
      policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
    return policy

  def train_mode(self):
    # -- PPO
    self.alg.policy.train()
    for d in self.alg.discriminator_dict.values():
      d.train()
    # -- RND
    if self.alg.rnd:
      self.alg.rnd.train()
    # -- Normalization
    if self.empirical_normalization:
      self.obs_normalizer.train()
      self.privileged_obs_normalizer.train()

  def eval_mode(self):
    # -- PPO
    self.alg.policy.eval()
    for d in self.alg.discriminator_dict.values():
      d.eval()
    # -- RND
    if self.alg.rnd:
      self.alg.rnd.eval()
    # -- Normalization
    if self.empirical_normalization:
      self.obs_normalizer.eval()
      self.privileged_obs_normalizer.eval()

  def add_git_repo_to_log(self, repo_file_path):
    self.git_status_repos.append(repo_file_path)

  """
    Helper functions.
    """

  def _configure_multi_gpu(self):
    """Configure multi-gpu training."""
    # check if distributed training is enabled
    self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
    self.is_distributed = self.gpu_world_size > 1

    # if not distributed training, set local and global rank to 0 and return
    if not self.is_distributed:
      self.gpu_local_rank = 0
      self.gpu_global_rank = 0
      self.multi_gpu_cfg = None
      return

    # get rank and world size
    self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
    self.gpu_global_rank = int(os.getenv("RANK", "0"))

    # make a configuration dictionary
    self.multi_gpu_cfg = {
      "global_rank": self.gpu_global_rank,  # rank of the main process
      "local_rank": self.gpu_local_rank,  # rank of the current process
      "world_size": self.gpu_world_size,  # total number of processes
    }

    # check if user has device specified for local rank
    if self.device != f"cuda:{self.gpu_local_rank}":
      raise ValueError(
        f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
      )
    # validate multi-gpu configuration
    if self.gpu_local_rank >= self.gpu_world_size:
      raise ValueError(
        f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
      )
    if self.gpu_global_rank >= self.gpu_world_size:
      raise ValueError(
        f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
      )

    from mjlab.third_party.amp_rsl_rl.utils import init_distributed_training

    if not torch.distributed.is_initialized():
      init_distributed_training(
        self.gpu_local_rank, self.gpu_global_rank, self.gpu_world_size
      )
