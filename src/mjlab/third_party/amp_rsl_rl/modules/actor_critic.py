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
from torch.distributions import Normal

from mjlab.third_party.amp_rsl_rl.utils import resolve_nn_activation

from .him_estimator import HIMEstimator


class ActorCritic(nn.Module):
  is_recurrent = False

  def __init__(
    self,
    num_actor_obs,
    num_critic_obs,
    num_actions,
    num_one_step_obs=None,
    num_one_step_critic_obs=None,
    history_length=None,
    vel_index_in_critic=None,
    num_latent=32,
    vel_scale_for_actor: float = 1.0,
    actor_recent_frames: int = 1,
    him_encoder_type: str = "mlp",
    him_gru_hidden_dim: int = 128,
    him_gru_num_layers: int = 1,
    him_gru_head_hidden_dims=(64,),
    actor_hidden_dims=(256, 256, 256),
    critic_hidden_dims=(256, 256, 256),
    activation="elu",
    init_noise_std=1.0,
    noise_std_type: str = "scalar",
    **kwargs,
  ):
    if kwargs:
      print(
        "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
        + str([key for key in kwargs.keys()])
      )
    super().__init__()
    activation = resolve_nn_activation(activation)

    use_him = num_one_step_obs is not None and history_length is not None
    self.use_him = use_him
    self.num_latent = int(num_latent)
    self.vel_scale_for_actor = float(vel_scale_for_actor)
    self.actor_recent_frames = int(actor_recent_frames)

    if use_him:
      self.history_length = int(history_length)
      self.num_one_step_obs = int(num_one_step_obs)
      self.num_one_step_critic_obs = int(num_one_step_critic_obs)
      n = max(0, min(self.actor_recent_frames, self.history_length))
      self.actor_recent_frames = n
      mlp_input_dim_a = self.num_one_step_obs * n + 3 + self.num_latent
      self.estimator = HIMEstimator(
        temporal_steps=self.history_length,
        num_one_step_obs=self.num_one_step_obs,
        num_one_step_critic_obs=self.num_one_step_critic_obs,
        vel_index_in_critic=int(vel_index_in_critic),
        vel_scale_in_critic=self.vel_scale_for_actor,
        enc_hidden_dims=[128, 64, self.num_latent],
        encoder_type=him_encoder_type,
        gru_hidden_dim=him_gru_hidden_dim,
        gru_num_layers=him_gru_num_layers,
        gru_head_hidden_dims=him_gru_head_hidden_dims,
      )
    else:
      self.history_length = None
      self.num_one_step_obs = None
      self.num_one_step_critic_obs = None
      self.estimator = None
      mlp_input_dim_a = num_actor_obs

    mlp_input_dim_c = num_critic_obs

    actor_layers = []
    actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
    actor_layers.append(activation)
    for layer_index in range(len(actor_hidden_dims)):
      if layer_index == len(actor_hidden_dims) - 1:
        actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
      else:
        actor_layers.append(
          nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1])
        )
        actor_layers.append(activation)
    self.actor = nn.Sequential(*actor_layers)

    critic_layers = []
    critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
    critic_layers.append(activation)
    for layer_index in range(len(critic_hidden_dims)):
      if layer_index == len(critic_hidden_dims) - 1:
        critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
      else:
        critic_layers.append(
          nn.Linear(
            critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]
          )
        )
        critic_layers.append(activation)
    self.critic = nn.Sequential(*critic_layers)

    print(f"Actor MLP: {self.actor}")
    print(f"Critic MLP: {self.critic}")
    if self.estimator is not None:
      print(f"HIM Estimator encoder: {self.estimator.encoder}")

    self.noise_std_type = noise_std_type
    if self.noise_std_type == "scalar":
      self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
    elif self.noise_std_type == "log":
      self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
    else:
      raise ValueError(
        f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
      )

    self.distribution = None
    Normal.set_default_validate_args(False)

  @staticmethod
  def init_weights(sequential, scales):
    [
      torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
      for idx, module in enumerate(
        mod for mod in sequential if isinstance(mod, nn.Linear)
      )
    ]

  def reset(self, dones=None):
    pass

  def forward(self):
    raise NotImplementedError

  @property
  def action_mean(self):
    return self.distribution.mean

  @property
  def action_std(self):
    return self.distribution.stddev

  @property
  def entropy(self):
    return self.distribution.entropy().sum(dim=-1)

  def _actor_input(self, observations):
    if self.use_him:
      with torch.no_grad():
        vel, latent = self.estimator(observations)
      vel = vel * self.vel_scale_for_actor
      n = int(self.actor_recent_frames)
      if n <= 0:
        recent = observations[:, 0:0]
      else:
        recent = observations[:, -(n * self.num_one_step_obs) :]
      return torch.cat([recent, vel, latent], dim=-1)
    return observations

  def update_distribution(self, observations):
    actor_in = self._actor_input(observations)
    mean = self.actor(actor_in)
    if self.noise_std_type == "scalar":
      std = self.std.expand_as(mean)
    elif self.noise_std_type == "log":
      std = torch.exp(self.log_std).expand_as(mean)
    else:
      raise ValueError(
        f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
      )
    self.distribution = Normal(mean, std)

  def act(self, observations, **kwargs):
    self.update_distribution(observations)
    return self.distribution.sample()

  def get_actions_log_prob(self, actions):
    return self.distribution.log_prob(actions).sum(dim=-1)

  def act_inference(self, observations):
    actor_in = self._actor_input(observations)
    return self.actor(actor_in)

  def evaluate(self, critic_observations, **kwargs):
    return self.critic(critic_observations)

  def load_state_dict(self, state_dict, strict=True):
    super().load_state_dict(state_dict, strict=strict)
    return True
