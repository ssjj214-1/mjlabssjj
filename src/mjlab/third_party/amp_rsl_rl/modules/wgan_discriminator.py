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

import torch
import torch.nn as nn
from torch import autograd


class WGANDiscriminator(nn.Module):
  """
  WGAN Discriminator neural network for adversarial motion priors (AMP) reward prediction.
  This is used for running tasks (e.g., ultra_run) instead of the standard GAN discriminator.

  Args:
      input_dim (int): Dimension of the input feature vector (concatenated state and next state).
      amp_reward_coef (float): Coefficient to scale the AMP reward.
      hidden_layer_sizes (list[int]): Sizes of hidden layers in the MLP trunk.
      device (torch.device): Device to run the model on (CPU or GPU).
      task_reward_lerp (float, optional): Interpolation factor between AMP reward and task reward.
          Defaults to 0.0 (only AMP reward).

  Attributes:
      trunk (nn.Sequential): MLP layers processing input features.
      amp_linear (nn.Linear): Final linear layer producing discriminator output.
      task_reward_lerp (float): Interpolation factor for combining rewards.
      disc_pos_mean (float): Moving average of positive discriminator output (for WGAN reward).
  """

  def __init__(
    self, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0
  ):
    super().__init__()

    self.device = device
    self.input_dim = input_dim

    self.amp_reward_coef = amp_reward_coef
    amp_layers = []
    curr_in_dim = input_dim
    for hidden_dim in hidden_layer_sizes:
      amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
      amp_layers.append(nn.ReLU())
      curr_in_dim = hidden_dim
    self.trunk = nn.Sequential(*amp_layers).to(device)
    self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

    self.trunk.train()
    self.amp_linear.train()

    self.task_reward_lerp = task_reward_lerp
    self.disc_pos_mean = 0.0  # Moving average for WGAN reward calculation

  def forward(self, x):
    """
    Forward pass through the discriminator network.

    Args:
        x (torch.Tensor): Input tensor with shape (batch_size, input_dim).

    Returns:
        torch.Tensor: Discriminator output logits with shape (batch_size, 1).
    """
    h = self.trunk(x)
    d = self.amp_linear(h)
    return d

  def compute_grad_pen(self, expert_state, expert_next_state, lambda_=10):
    """
    Compute gradient penalty for the expert data (standard GAN version).
    This method is kept for compatibility but WGAN uses compute_WGAN_grad_pen instead.

    Args:
        expert_state (torch.Tensor): Batch of expert states.
        expert_next_state (torch.Tensor): Batch of expert next states.
        lambda_ (float, optional): Gradient penalty coefficient. Defaults to 10.

    Returns:
        torch.Tensor: Scalar gradient penalty loss.
    """
    expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
    expert_data.requires_grad = True

    disc = self.amp_linear(self.trunk(expert_data))
    ones = torch.ones(disc.size(), device=disc.device)
    grad = autograd.grad(
      outputs=disc,
      inputs=expert_data,
      grad_outputs=ones,
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]

    # Enforce that the grad norm approaches 0.
    grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
    return grad_pen

  def compute_WGAN_grad_pen(self, expert_state, policy_state, lambda_=10):
    """
    Compute WGAN gradient penalty using interpolation between expert and policy states.

    Args:
        expert_state (torch.Tensor): Batch of expert state pairs (concatenated state and next_state).
        policy_state (torch.Tensor): Batch of policy state pairs (concatenated state and next_state).
        lambda_ (float, optional): Gradient penalty coefficient. Defaults to 10.

    Returns:
        torch.Tensor: Scalar gradient penalty loss.
    """
    alpha = torch.rand(expert_state.size(0), 1).to(self.device)  # B, 1
    interpolates = alpha * expert_state + (1 - alpha) * policy_state
    interpolates.requires_grad_(True)
    disc_interpolates = self.forward(interpolates)
    gradients = torch.autograd.grad(
      outputs=disc_interpolates,
      inputs=interpolates,
      grad_outputs=torch.ones_like(disc_interpolates),
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)

    # Enforce that the grad norm approaches 1 (WGAN requirement).
    grad_pen = lambda_ * (gradients.norm(2, dim=1) - 1).pow(2).mean()
    return grad_pen

  def predict_amp_reward(self, state, next_state, task_reward, normalizer=None):
    """
    Predict the AMP reward given current and next states (standard GAN version).
    This method is kept for compatibility but WGAN uses predict_amp_reward_WGAN instead.

    Args:
        state (torch.Tensor): Current state tensor.
        next_state (torch.Tensor): Next state tensor.
        task_reward (torch.Tensor): Task-specific reward tensor.
        normalizer (optional): Normalizer object to normalize input states before prediction.

    Returns:
        tuple:
            - reward (torch.Tensor): Predicted AMP reward (optionally interpolated) with shape (batch_size,).
            - d (torch.Tensor): Raw discriminator output logits with shape (batch_size, 1).
    """
    with torch.no_grad():
      self.eval()
      if normalizer is not None:
        state = normalizer.normalize_torch(state, self.device)
        next_state = normalizer.normalize_torch(next_state, self.device)

      d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
      reward = self.amp_reward_coef * torch.clamp(
        1 - (1 / 4) * torch.square(d - 1), min=0
      )
      if self.task_reward_lerp > 0:
        reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
      self.train()
    return reward.squeeze(), d

  def predict_amp_reward_WGAN(
    self, state, next_state, amp_pos, next_amp_pos, task_reward, normalizer=None
  ):
    """
    Predict the AMP reward using WGAN formulation.

    Args:
        state (torch.Tensor): Current policy state tensor.
        next_state (torch.Tensor): Next policy state tensor.
        amp_pos (torch.Tensor): Current expert/positive state tensor.
        next_amp_pos (torch.Tensor): Next expert/positive state tensor.
        task_reward (torch.Tensor): Task-specific reward tensor.
        normalizer (optional): Normalizer object to normalize input states before prediction.

    Returns:
        tuple:
            - reward (torch.Tensor): Predicted AMP reward (optionally interpolated) with shape (batch_size,).
            - output_neg (torch.Tensor): Raw discriminator output for policy states with shape (batch_size, 1).
    """
    # Reward is used during rollout collection; keep it gradient-free to avoid graph retention.
    with torch.no_grad():
      self.eval()
      if normalizer is not None:
        state = normalizer.normalize_torch(state, self.device)
        next_state = normalizer.normalize_torch(next_state, self.device)
        amp_pos = normalizer.normalize_torch(amp_pos, self.device)
        next_amp_pos = normalizer.normalize_torch(next_amp_pos, self.device)

      # Ensure dtype consistency (avoid float64 from upstream numpy conversions)
      target_dtype = self.amp_linear.weight.dtype
      state = state.to(target_dtype)
      next_state = next_state.to(target_dtype)
      amp_pos = amp_pos.to(target_dtype)
      next_amp_pos = next_amp_pos.to(target_dtype)

      output_neg = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
      output_pos = self.amp_linear(
        self.trunk(torch.cat([amp_pos, next_amp_pos], dim=-1))
      )

      # Moving average of positive discriminator output
      self.disc_pos_mean = (
        0.99 * getattr(self, "disc_pos_mean", 0.0)
        + 0.01 * torch.mean(output_pos).item()
      )
      # score = D(x) - E[D(x_pos)], should be <= 0
      score = (output_neg - self.disc_pos_mean).clamp(max=0.0)
      # reward in (0, 1], scaled by configured AMP coefficient
      reward = self.amp_reward_coef * torch.exp(score / 10)
      if self.task_reward_lerp > 0:
        reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
      self.train()
      return reward.squeeze(), output_neg

  def _lerp_reward(self, disc_r, task_r):
    """
    Linearly interpolate between discriminator reward and task reward.

    Args:
        disc_r (torch.Tensor): Discriminator reward.
        task_r (torch.Tensor): Task reward.

    Returns:
        torch.Tensor: Interpolated reward.
    """
    r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
    return r
