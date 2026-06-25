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

import torch
import torch.nn as nn
import torch.optim as optim

from mjlab.third_party.amp_rsl_rl.modules import ActorCritic
from mjlab.third_party.amp_rsl_rl.modules.rnd import RandomNetworkDistillation
from mjlab.third_party.amp_rsl_rl.storage import ReplayBuffer, RolloutStorage
from mjlab.third_party.amp_rsl_rl.utils import string_to_callable


class MULTIAMPPPO:
  """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

  policy: ActorCritic
  """The actor critic module."""

  def __init__(
    self,
    policy,
    discriminator_dict: dict,  # dict: {style_key: Discriminator}
    amp_data_dict: dict,  # dict: {style_key: AMPLoader}
    amp_normalizer_dict: dict,  # dict: {style_key: Normalizer}
    amp_replay_buffer_size=100000,
    min_std=None,
    num_learning_epochs=1,
    num_mini_batches=1,
    clip_param=0.2,
    gamma=0.998,
    lam=0.95,
    value_loss_coef=1.0,
    entropy_coef=0.0,
    learning_rate=1e-3,
    max_grad_norm=1.0,
    use_clipped_value_loss=True,
    schedule="fixed",
    desired_kl=0.01,
    device="cpu",
    normalize_advantage_per_mini_batch=False,
    # HoST-style smoothness regularization (ports HoST's rsl_rl PPO "smooth
    # loss"). Perturbs the observation along the consecutive-step direction with
    # a random magnitude and penalizes the resulting change in BOTH the policy
    # mean and the value, so it regularizes the actor (real-robot jitter) and the
    # critic together. Coefficients follow HoST's bound formula:
    #     epsilon            = lower / (upper - lower)
    #     policy_smooth_coef = upper * epsilon
    #     value_smooth_coef  = value_smoothness_coef * policy_smooth_coef
    # Default ``smoothness_lower_bound=0.0`` -> coef 0 == OFF, so existing tasks
    # are unchanged (matches HoST's own default-off behaviour).
    smoothness_upper_bound: float = 1.0,
    smoothness_lower_bound: float = 0.0,
    value_smoothness_coef: float = 0.1,
    # RND parameters
    rnd_cfg: dict | None = None,
    # Symmetry parameters
    symmetry_cfg: dict | None = None,
    # Distributed training parameters
    multi_gpu_cfg: dict | None = None,
  ):
    # device-related parameters
    self.device = device
    self.is_multi_gpu = multi_gpu_cfg is not None
    # Multi-GPU parameters
    if multi_gpu_cfg is not None:
      self.gpu_global_rank = multi_gpu_cfg["global_rank"]
      self.gpu_world_size = multi_gpu_cfg["world_size"]
    else:
      self.gpu_global_rank = 0
      self.gpu_world_size = 1

    # RND components
    if rnd_cfg is not None:
      # Create RND module
      self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
      # Create RND optimizer
      params = self.rnd.predictor.parameters()
      self.rnd_optimizer = optim.Adam(params, lr=rnd_cfg.get("learning_rate", 1e-3))
    else:
      self.rnd = None
      self.rnd_optimizer = None

    # Symmetry components
    if symmetry_cfg is not None:
      # Check if symmetry is enabled
      use_symmetry = (
        symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
      )
      # Print that we are not using symmetry
      if not use_symmetry:
        print("Symmetry not used for learning. We will use it for logging instead.")
      # If function is a string then resolve it to a function
      if isinstance(symmetry_cfg["data_augmentation_func"], str):
        symmetry_cfg["data_augmentation_func"] = string_to_callable(
          symmetry_cfg["data_augmentation_func"]
        )
      # Check valid configuration
      if symmetry_cfg["use_data_augmentation"] and not callable(
        symmetry_cfg["data_augmentation_func"]
      ):
        raise ValueError(
          "Data augmentation enabled but the function is not callable:"
          f" {symmetry_cfg['data_augmentation_func']}"
        )
      # Store symmetry configuration
      self.symmetry = symmetry_cfg
    else:
      self.symmetry = None

    # multi style discriminator components
    self.amploss_coef = 1.0
    self.min_std = None if min_std is None else min_std.to(device=self.device)

    self.amp_data_dict = amp_data_dict
    self.amp_normalizer_dict = amp_normalizer_dict
    self.discriminator_dict = discriminator_dict
    for d in self.discriminator_dict.values():
      d.to(self.device)
    self.use_wgan = all(
      hasattr(disc, "predict_amp_reward_WGAN")
      for disc in self.discriminator_dict.values()
    )
    self.amp_storage_dict = {}
    for style_key, disc in self.discriminator_dict.items():
      input_dim = disc.input_dim // 2
      self.amp_storage_dict[style_key] = ReplayBuffer(
        input_dim, amp_replay_buffer_size, self.device
      )

    self.amp_transition = RolloutStorage.Transition()

    # PPO components
    self.policy = policy
    self.policy.to(self.device)
    self._clamp_policy_std()
    # Create optimizer
    params = [{"params": self.policy.parameters(), "name": "policy"}]
    for k, disc in self.discriminator_dict.items():
      params.append(
        {
          "params": disc.trunk.parameters(),
          "weight_decay": 10e-4,
          "name": f"amp_trunk_{k}",
        }
      )
      params.append(
        {
          "params": disc.amp_linear.parameters(),
          "weight_decay": 10e-2,
          "name": f"amp_head_{k}",
        }
      )
    self.optimizer = optim.Adam(params, lr=learning_rate)
    # Create rollout storage
    self.storage: RolloutStorage = None  # type: ignore
    self.transition = RolloutStorage.Transition()

    # PPO parameters
    self.clip_param = clip_param
    self.num_learning_epochs = num_learning_epochs
    self.num_mini_batches = num_mini_batches
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef
    self.gamma = gamma
    self.lam = lam
    self.max_grad_norm = max_grad_norm
    self.use_clipped_value_loss = use_clipped_value_loss
    self.desired_kl = desired_kl
    self.schedule = schedule
    self.learning_rate = learning_rate
    self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    # HoST-style smoothness coefficients (see ctor docstring above).
    self.smoothness_upper_bound = float(smoothness_upper_bound)
    self.smoothness_lower_bound = float(smoothness_lower_bound)
    self.value_smoothness_coef = float(value_smoothness_coef)
    self.smooth_on = (
      self.smoothness_lower_bound > 0.0
      and self.smoothness_upper_bound > self.smoothness_lower_bound
    )
    if self.smooth_on:
      _eps = self.smoothness_lower_bound / (
        self.smoothness_upper_bound - self.smoothness_lower_bound
      )
      self.policy_smooth_coef = self.smoothness_upper_bound * _eps
      self.value_smooth_coef = self.value_smoothness_coef * self.policy_smooth_coef
    else:
      self.policy_smooth_coef = 0.0
      self.value_smooth_coef = 0.0

  def init_storage(
    self,
    training_type,
    num_envs,
    num_transitions_per_env,
    actor_obs_shape,
    critic_obs_shape,
    actions_shape,
    next_critic_obs_shape=None,
  ):
    # create memory for RND as well :)
    if self.rnd:
      rnd_state_shape = [self.rnd.num_states]
    else:
      rnd_state_shape = None
    # create rollout storage
    self.storage = RolloutStorage(
      training_type,
      num_envs,
      num_transitions_per_env,
      actor_obs_shape,
      critic_obs_shape,
      actions_shape,
      rnd_state_shape,
      self.device,
      next_privileged_obs_shape=next_critic_obs_shape,
    )

  def act(self, obs, critic_obs, amp_obs):
    if self.policy.is_recurrent:
      self.transition.hidden_states = self.policy.get_hidden_states()
    # compute the actions and values
    self.transition.actions = self.policy.act(obs).detach()
    self.transition.values = self.policy.evaluate(critic_obs).detach()
    self.transition.actions_log_prob = self.policy.get_actions_log_prob(
      self.transition.actions
    ).detach()
    self.transition.action_mean = self.policy.action_mean.detach()
    self.transition.action_sigma = self.policy.action_std.detach()
    # need to record obs and critic_obs before env.step()
    self.transition.observations = obs
    self.transition.privileged_observations = critic_obs
    self.amp_transition.observations = amp_obs
    return self.transition.actions

  def process_env_step(
    self, rewards, dones, infos, amp_obs, next_critic_obs=None, style_ids=None
  ):
    # Record the rewards and dones
    # Note: we clone here because later on we bootstrap the rewards based on timeouts
    if next_critic_obs is not None:
      self.transition.next_critic_observations = next_critic_obs.clone()
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones

    # Compute the intrinsic rewards and add to extrinsic rewards
    if self.rnd:
      # Obtain curiosity gates / observations from infos
      rnd_state = infos["observations"]["rnd_state"]
      # Compute the intrinsic rewards
      # note: rnd_state is the gated_state after normalization if normalization is used
      self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
      # Add intrinsic rewards to extrinsic rewards
      self.transition.rewards += self.intrinsic_rewards
      # Record the curiosity gates
      self.transition.rnd_state = rnd_state.clone()

    # Bootstrapping on time outs
    if "time_outs" in infos:
      self.transition.rewards += self.gamma * torch.squeeze(
        self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
      )

    # record the transition
    if style_ids is not None:
      unique_style_ids = torch.unique(style_ids)
      for sid in unique_style_ids:
        sid_int = int(sid.item())
        style_key = f"style_{sid_int}"
        if style_key not in self.amp_storage_dict:
          continue

        mask = style_ids == sid
        idxs = mask.nonzero(as_tuple=True)[0]
        if idxs.numel() == 0:
          continue

        amp_prev_obs_style = self.amp_transition.observations[idxs]
        amp_next_obs_style = amp_obs[idxs]

        # insert into that style's replay buffer
        self.amp_storage_dict[style_key].insert(amp_prev_obs_style, amp_next_obs_style)
    self.storage.add_transitions(self.transition)
    self.transition.clear()
    self.amp_transition.clear()
    self.policy.reset(dones)

  def compute_returns(self, last_critic_obs):
    # compute value for the last step
    last_values = self.policy.evaluate(last_critic_obs).detach()
    self.storage.compute_returns(
      last_values,
      self.gamma,
      self.lam,
      normalize_advantage=not self.normalize_advantage_per_mini_batch,
    )

  def update(self):  # noqa: C901
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_entropy = 0
    mean_amp_loss = 0
    mean_grad_pen_loss = 0
    mean_policy_pred = 0
    mean_expert_pred = 0
    mean_estimation_loss = 0
    mean_swap_loss = 0
    # -- RND loss
    if self.rnd:
      mean_rnd_loss = 0
    else:
      mean_rnd_loss = None
    # -- Symmetry loss
    if self.symmetry:
      mean_symmetry_loss = 0
    else:
      mean_symmetry_loss = None
    # -- HoST-style smoothness losses (accumulators always float; only emitted
    #    to the loss dict when smoothness is enabled).
    mean_smooth_policy = 0.0
    mean_smooth_value = 0.0
    mean_action_smoothness = 0.0  # diagnostic: realized ||mu(s_t) - mu(s_{t+1})||

    # Precompute consecutive-step (s_t, s_{t+1}) pairs for the smoothness loss,
    # mirroring HoST's storage generator which builds next_obs from obs[1:] and
    # ``cont`` from ``1 - dones[:-1]``. Derived directly from the rollout buffer
    # (no extra storage/runner plumbing). We keep actor and critic pairs since
    # mjlab uses asymmetric obs (HoST feeds the same obs to actor and critic).
    sm_obs_cur = sm_obs_nxt = sm_cont = None
    sm_crit_cur = sm_crit_nxt = None
    sm_n_pairs = 0
    if self.smooth_on:
      with torch.no_grad():
        obs_seq = self.storage.observations  # [T, N, actor_dim]
        cont_seq = 1.0 - self.storage.dones.float()  # [T, N, 1]
        a_dim = obs_seq.shape[-1]
        sm_obs_cur = obs_seq[:-1].reshape(-1, a_dim)
        sm_obs_nxt = obs_seq[1:].reshape(-1, a_dim)
        sm_cont = cont_seq[:-1].reshape(-1, 1)
        sm_n_pairs = sm_obs_cur.shape[0]
        if self.storage.privileged_observations is not None:
          crit_seq = self.storage.privileged_observations  # [T, N, critic_dim]
          c_dim = crit_seq.shape[-1]
          sm_crit_cur = crit_seq[:-1].reshape(-1, c_dim)
          sm_crit_nxt = crit_seq[1:].reshape(-1, c_dim)

    # generator for mini batches
    if self.policy.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    batch_size_per_iter = (
      self.storage.num_envs
      * self.storage.num_transitions_per_env
      // self.num_mini_batches
    )
    amp_policy_generator_dict = {}
    amp_expert_generator_dict = {}

    # A style's policy replay buffer can be empty when no env was assigned that
    # style during the rollout (e.g. a stand-heavy / pure-standing curriculum
    # stage where only the stand style appears). Sampling from an empty buffer
    # crashes (np.random.choice on 0 samples), so only run the discriminator for
    # styles that actually collected policy transitions this iteration.
    active_style_keys = [
      k for k, s in self.amp_storage_dict.items() if s.num_samples > 0
    ]

    for style_key in active_style_keys:
      amp_policy_generator_dict[style_key] = self.amp_storage_dict[
        style_key
      ].feed_forward_generator(
        self.num_learning_epochs * self.num_mini_batches, batch_size_per_iter
      )
      amp_expert_generator_dict[style_key] = self.amp_data_dict[
        style_key
      ].feed_forward_generator(
        self.num_learning_epochs * self.num_mini_batches, batch_size_per_iter
      )

    # iterate over batches
    for sample in generator:
      (
        obs_batch,
        critic_obs_batch,
        next_critic_obs_batch,
        actions_batch,
        target_values_batch,
        advantages_batch,
        returns_batch,
        old_actions_log_prob_batch,
        old_mu_batch,
        old_sigma_batch,
        hid_states_batch,
        masks_batch,
        rnd_state_batch,
      ) = sample

      # number of augmentations per sample
      # we start with 1 and increase it if we use symmetry augmentation
      num_aug = 1
      # original batch size
      original_batch_size = obs_batch.shape[0]

      # check if we should normalize advantages per mini batch
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          advantages_batch = (advantages_batch - advantages_batch.mean()) / (
            advantages_batch.std() + 1e-8
          )

      # Perform symmetric augmentation
      if self.symmetry and self.symmetry["use_data_augmentation"]:
        # augmentation using symmetry
        data_augmentation_func = self.symmetry["data_augmentation_func"]
        # returned shape: [batch_size * num_aug, ...]
        obs_batch, actions_batch = data_augmentation_func(
          obs=obs_batch,
          actions=actions_batch,
          env=self.symmetry["_env"],
          obs_type="policy",
        )
        if next_critic_obs_batch is not None:
          critic_obs_batch, next_critic_obs_batch = data_augmentation_func(
            obs=critic_obs_batch,
            actions=next_critic_obs_batch,
            env=self.symmetry["_env"],
            obs_type="critic",
          )
        else:
          critic_obs_batch, _ = data_augmentation_func(
            obs=critic_obs_batch,
            actions=None,
            env=self.symmetry["_env"],
            obs_type="critic",
          )
        # compute number of augmentations per sample
        num_aug = int(obs_batch.shape[0] / original_batch_size)
        # repeat the rest of the batch
        # -- actor
        old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
        # -- critic
        target_values_batch = target_values_batch.repeat(num_aug, 1)
        advantages_batch = advantages_batch.repeat(num_aug, 1)
        returns_batch = returns_batch.repeat(num_aug, 1)

      # Recompute actions log prob and entropy for current batch of transitions
      # Note: we need to do this because we updated the policy with the new parameters
      # -- actor
      self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
      actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
      # -- critic
      value_batch = self.policy.evaluate(
        critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
      )
      # -- entropy
      # we only keep the entropy of the first augmentation (the original one)
      mu_batch = self.policy.action_mean[:original_batch_size]
      sigma_batch = self.policy.action_std[:original_batch_size]
      entropy_batch = self.policy.entropy[:original_batch_size]

      # KL
      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = torch.sum(
            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
            / (2.0 * torch.square(sigma_batch))
            - 0.5,
            axis=-1,
          )
          kl_mean = torch.mean(kl)

          # Reduce the KL divergence across all GPUs
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size

          # Update the learning rate
          # Perform this adaptation only on the main process
          # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
          #       then the learning rate should be the same across all GPUs.
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)

          # Update the learning rate for all GPUs
          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()

          # Update the learning rate for all parameter groups
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      if (
        hasattr(self.policy, "estimator")
        and self.policy.estimator is not None
        and next_critic_obs_batch is not None
      ):
        estimation_loss, swap_loss = self.policy.estimator.update(
          obs_batch, next_critic_obs_batch, lr=self.learning_rate
        )
        mean_estimation_loss += estimation_loss
        mean_swap_loss += swap_loss

      # Surrogate loss
      ratio = torch.exp(
        actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
      )
      surrogate = -torch.squeeze(advantages_batch) * ratio
      surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      # Value function loss
      if self.use_clipped_value_loss:
        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (returns_batch - value_batch).pow(2).mean()

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy_batch.mean()
      )

      # Symmetry loss
      if self.symmetry:
        # obtain the symmetric actions
        # if we did augmentation before then we don't need to augment again
        if not self.symmetry["use_data_augmentation"]:
          data_augmentation_func = self.symmetry["data_augmentation_func"]
          obs_batch, _ = data_augmentation_func(
            obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
          )
          # compute number of augmentations per sample
          num_aug = int(obs_batch.shape[0] / original_batch_size)

        # actions predicted by the actor for symmetrically-augmented observations
        mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

        # compute the symmetrically augmented actions
        # note: we are assuming the first augmentation is the original one.
        #   We do not use the action_batch from earlier since that action was sampled from the distribution.
        #   However, the symmetry loss is computed using the mean of the distribution.
        action_mean_orig = mean_actions_batch[:original_batch_size]
        _, actions_mean_symm_batch = data_augmentation_func(
          obs=None,
          actions=action_mean_orig,
          env=self.symmetry["_env"],
          obs_type="policy",
        )

        # compute the loss (we skip the first augmentation as it is the original one)
        mse_loss = torch.nn.MSELoss()
        symmetry_loss = mse_loss(
          mean_actions_batch[original_batch_size:],
          actions_mean_symm_batch.detach()[original_batch_size:],
        )
        # add the loss to the total loss
        if self.symmetry["use_mirror_loss"]:
          loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
        else:
          symmetry_loss = symmetry_loss.detach()

      # Random Network Distillation loss
      if self.rnd:
        # predict the embedding and the target
        predicted_embedding = self.rnd.predictor(rnd_state_batch)
        target_embedding = self.rnd.target(rnd_state_batch).detach()
        # compute the loss as the mean squared error
        mseloss = torch.nn.MSELoss()
        rnd_loss = mseloss(predicted_embedding, target_embedding)

      # Discriminator loss.
      amp_loss_total = 0.0
      grad_pen_total = 0.0
      policy_pred_mean = 0.0
      expert_pred_mean = 0.0

      for style_key in active_style_keys:
        disc = self.discriminator_dict[style_key]

        policy_state, policy_next_state = next(amp_policy_generator_dict[style_key])
        expert_state, expert_next_state = next(amp_expert_generator_dict[style_key])

        amp_norm = self.amp_normalizer_dict.get(style_key)
        if amp_norm is not None:
          with torch.no_grad():
            policy_state = amp_norm.normalize_torch(policy_state, self.device)
            policy_next_state = amp_norm.normalize_torch(policy_next_state, self.device)
            expert_state = amp_norm.normalize_torch(expert_state, self.device)
            expert_next_state = amp_norm.normalize_torch(expert_next_state, self.device)

        # WGAN or GAN
        if self.use_wgan:
          policy_pair = torch.cat([policy_state, policy_next_state], dim=-1)
          expert_pair = torch.cat([expert_state, expert_next_state], dim=-1)

          policy_d = disc(policy_pair)
          expert_d = disc(expert_pair)

          amp_loss = -(torch.mean(expert_d) - torch.mean(policy_d))
          grad_pen = disc.compute_WGAN_grad_pen(expert_pair, policy_pair, lambda_=10)
        else:
          policy_d = disc(torch.cat([policy_state, policy_next_state], dim=-1))
          expert_d = disc(torch.cat([expert_state, expert_next_state], dim=-1))

          expert_loss = torch.nn.MSELoss()(expert_d, torch.ones_like(expert_d))
          policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones_like(policy_d))
          amp_loss = 0.5 * (expert_loss + policy_loss)
          grad_pen = disc.compute_grad_pen(expert_state, expert_next_state, lambda_=10)

        amp_loss_total += amp_loss
        grad_pen_total += grad_pen
        policy_pred_mean += policy_d.mean().item()
        expert_pred_mean += expert_d.mean().item()

        if amp_norm is not None:
          amp_norm.update(policy_state.detach().cpu().numpy())
          amp_norm.update(expert_state.detach().cpu().numpy())

      loss += self.amploss_coef * amp_loss_total + self.amploss_coef * grad_pen_total

      # HoST-style smoothness loss (only when enabled). Perturb the observation
      # along the (s_{t+1} - s_t) direction by a random per-sample magnitude in
      # ``cont * [-1, 1]`` (cont == 0 at episode boundaries -> no perturbation),
      # then penalize the change in the policy mean and the value:
      #     L = c_pi * ||mu(s) - mu(s_mix)||^2 + c_V * ||V(s) - V(s_mix)||^2
      smooth_policy_val = 0.0
      smooth_value_val = 0.0
      action_smoothness_val = 0.0
      if (
        self.smooth_on
        and sm_n_pairs > 0
        and sm_obs_cur is not None
        and sm_obs_nxt is not None
        and sm_cont is not None
      ):
        sample_n = min(original_batch_size, sm_n_pairs)
        sel = torch.randint(0, sm_n_pairs, (sample_n,), device=self.device)
        obs_cur = sm_obs_cur[sel]
        obs_nxt = sm_obs_nxt[sel]
        cont = sm_cont[sel]
        # Per-sample scalar mix weight, broadcast across obs dims.
        mix_w = cont * (torch.rand_like(cont) - 0.5) * 2.0
        mix_obs = obs_cur + mix_w * (obs_nxt - obs_cur)
        mu_cur = self.policy.act_inference(obs_cur)
        mu_mix = self.policy.act_inference(mix_obs)
        policy_smooth_loss = torch.square(torch.norm(mu_cur - mu_mix, dim=-1)).mean()
        loss = loss + self.policy_smooth_coef * policy_smooth_loss
        smooth_policy_val = policy_smooth_loss.item()
        # Diagnostic only: realized consecutive-step action change.
        with torch.no_grad():
          mu_nxt = self.policy.act_inference(obs_nxt)
          action_smoothness_val = torch.norm(mu_cur - mu_nxt, dim=-1).mean().item()
        # Value smoothness (critic). Uses the same per-sample mix weight on the
        # critic-obs pair, since mjlab's critic obs differs from the actor obs.
        if (
          self.value_smooth_coef > 0.0
          and sm_crit_cur is not None
          and sm_crit_nxt is not None
        ):
          crit_cur = sm_crit_cur[sel]
          crit_nxt = sm_crit_nxt[sel]
          mix_crit = crit_cur + mix_w * (crit_nxt - crit_cur)
          v_cur = self.policy.evaluate(crit_cur)
          v_mix = self.policy.evaluate(mix_crit)
          value_smooth_loss = torch.square(torch.norm(v_cur - v_mix, dim=-1)).mean()
          loss = loss + self.value_smooth_coef * value_smooth_loss
          smooth_value_val = value_smooth_loss.item()

      # Compute the gradients
      # -- For PPO
      self.optimizer.zero_grad()
      loss.backward()
      # -- For RND
      if self.rnd:
        self.rnd_optimizer.zero_grad()  # type: ignore
        rnd_loss.backward()

      # Collect gradients from all GPUs
      if self.is_multi_gpu:
        self.reduce_parameters()

      # Apply the gradients
      # -- For PPO
      grad_norm = nn.utils.clip_grad_norm_(
        self.policy.parameters(), self.max_grad_norm
      )
      # A single non-finite gradient (NaN/Inf, e.g. from an extreme physics
      # state on rough terrain) would otherwise permanently corrupt the policy
      # parameters -- including the std parameter, which then makes
      # ``Normal`` sampling raise "normal expects all elements of std >= 0.0".
      # Skip the step instead of poisoning the network.
      if torch.isfinite(grad_norm):
        self.optimizer.step()
      else:
        self.optimizer.zero_grad(set_to_none=True)
      self._clamp_policy_std()
      # -- For RND
      if self.rnd_optimizer:
        self.rnd_optimizer.step()

      # Store the losses
      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy_batch.mean().item()
      num_styles = max(len(active_style_keys), 1)
      amp_loss_item = (
        amp_loss_total.item()
        if isinstance(amp_loss_total, torch.Tensor)
        else float(amp_loss_total)
      )
      grad_pen_item = (
        grad_pen_total.item()
        if isinstance(grad_pen_total, torch.Tensor)
        else float(grad_pen_total)
      )
      mean_amp_loss += amp_loss_item / num_styles
      mean_grad_pen_loss += grad_pen_item / num_styles
      mean_policy_pred += policy_pred_mean / num_styles
      mean_expert_pred += expert_pred_mean / num_styles
      # -- RND loss
      if mean_rnd_loss is not None:
        mean_rnd_loss += rnd_loss.item()
      # -- Symmetry loss
      if mean_symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()
      # -- Smoothness losses
      if self.smooth_on:
        mean_smooth_policy += smooth_policy_val
        mean_smooth_value += smooth_value_val
        mean_action_smoothness += action_smoothness_val

    # -- For PPO
    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    # -- For RND
    if mean_rnd_loss is not None:
      mean_rnd_loss /= num_updates
    # -- For Symmetry
    if mean_symmetry_loss is not None:
      mean_symmetry_loss /= num_updates
    # -- For smoothness
    if self.smooth_on:
      mean_smooth_policy /= num_updates
      mean_smooth_value /= num_updates
      mean_action_smoothness /= num_updates
    # -- Clear the storage
    mean_amp_loss /= num_updates
    mean_grad_pen_loss /= num_updates
    mean_policy_pred /= num_updates
    mean_expert_pred /= num_updates
    if hasattr(self.policy, "estimator") and self.policy.estimator is not None:
      mean_estimation_loss /= num_updates
      mean_swap_loss /= num_updates
    self.storage.clear()

    # construct the loss dictionary
    loss_dict = {
      "value_function": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
      "amp": mean_amp_loss,
      "amp_grad_pen": mean_grad_pen_loss,
      "amp_policy_pred": mean_policy_pred,
      "amp_expert_pred": mean_expert_pred,
    }
    if hasattr(self.policy, "estimator") and self.policy.estimator is not None:
      loss_dict["estimation"] = mean_estimation_loss
      loss_dict["swap"] = mean_swap_loss
    if self.rnd:
      loss_dict["rnd"] = mean_rnd_loss
    if self.symmetry:
      loss_dict["symmetry"] = mean_symmetry_loss
    if self.smooth_on:
      loss_dict["smooth_policy"] = mean_smooth_policy
      loss_dict["smooth_value"] = mean_smooth_value
      loss_dict["action_smoothness"] = mean_action_smoothness

    return loss_dict

  """
    Helper functions
    """

  def _clamp_policy_std(self):
    """Keep policy std parameters valid for torch Normal sampling."""
    if self.min_std is None:
      return

    with torch.no_grad():
      if hasattr(self.policy, "std"):
        std = self.policy.std
        min_std = self.min_std
        if min_std.numel() == 1 and std.numel() != 1:
          min_std = min_std.expand_as(std)
        elif min_std.shape != std.shape:
          min_std = min_std.reshape_as(std)
        # ``clamp_`` leaves NaN untouched, so a non-finite std would survive and
        # crash the next ``Normal`` sample. Replace non-finite entries with the
        # floor before clamping.
        torch.nan_to_num_(std, nan=0.0, posinf=0.0, neginf=0.0)
        std.clamp_(min=min_std)
      elif hasattr(self.policy, "log_std"):
        log_std = self.policy.log_std
        min_std = self.min_std.clamp_min(torch.finfo(log_std.dtype).tiny)
        if min_std.numel() == 1 and log_std.numel() != 1:
          min_std = min_std.expand_as(log_std)
        elif min_std.shape != log_std.shape:
          min_std = min_std.reshape_as(log_std)
        torch.nan_to_num_(log_std, nan=0.0, posinf=0.0, neginf=0.0)
        log_std.clamp_(min=torch.log(min_std))

  def broadcast_parameters(self):
    """Broadcast model parameters to all GPUs."""
    from mjlab.third_party.amp_rsl_rl.utils import broadcast_module_parameters

    broadcast_module_parameters(self.policy, src=0)
    for discriminator in self.discriminator_dict.values():
      broadcast_module_parameters(discriminator, src=0)
    if self.rnd:
      broadcast_module_parameters(self.rnd.predictor, src=0)

  def reduce_parameters(self):
    """Collect gradients from all GPUs and average them.

    This function is called after the backward pass to synchronize the gradients across all GPUs.
    """
    # Create a tensor to store the gradients. All ranks must reduce tensors
    # with identical sizes, so parameters without local gradients contribute
    # zeros instead of being skipped.
    all_params = list(self.policy.parameters())
    for discriminator in self.discriminator_dict.values():
      all_params += list(discriminator.parameters())
    if self.rnd:
      all_params += list(self.rnd.parameters())
    grad_present = torch.tensor(
      [param.grad is not None for param in all_params],
      device=self.device,
      dtype=torch.float32,
    )
    grads = [
      param.grad.view(-1)
      if param.grad is not None
      else torch.zeros_like(param, memory_format=torch.preserve_format).view(-1)
      for param in all_params
    ]
    all_grads = torch.cat(grads)

    # Average the gradients across all GPUs
    torch.distributed.all_reduce(grad_present, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
    all_grads /= self.gpu_world_size

    # Update the gradients for all parameters with the reduced gradients
    offset = 0
    for param_idx, param in enumerate(all_params):
      numel = param.numel()
      reduced = all_grads[offset : offset + numel].view_as(param)
      if grad_present[param_idx].item() > 0:
        if param.grad is None:
          param.grad = reduced.clone()
        else:
          param.grad.data.copy_(reduced)
      # update the offset for the next parameter
      offset += numel
