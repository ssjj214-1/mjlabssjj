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

import datetime as _datetime
import importlib
import os
import pathlib
from typing import Callable, Tuple

import git
import numpy as np
import torch


class RunningMeanStd:
  def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
    """
    Calculates the running mean and std of a data stream
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    :param epsilon: helps with arithmetic issues
    :param shape: the shape of the data stream's output
    """
    self.mean = np.zeros(shape, np.float64)
    self.var = np.ones(shape, np.float64)
    self.count = epsilon

  def update(self, arr: np.ndarray) -> None:
    batch_mean = np.mean(arr, axis=0)
    batch_var = np.var(arr, axis=0)
    batch_count = arr.shape[0]
    self.update_from_moments(batch_mean, batch_var, batch_count)

  def update_from_moments(
    self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
  ) -> None:
    delta = batch_mean - self.mean
    tot_count = self.count + batch_count

    new_mean = self.mean + delta * batch_count / tot_count
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    m_2 = (
      m_a
      + m_b
      + np.square(delta) * self.count * batch_count / (self.count + batch_count)
    )
    new_var = m_2 / (self.count + batch_count)

    new_count = batch_count + self.count

    self.mean = new_mean
    self.var = new_var
    self.count = new_count


class Normalizer(RunningMeanStd):
  def __init__(self, input_dim, epsilon=1e-4, clip_obs=10.0):
    super().__init__(shape=input_dim)
    self.epsilon = epsilon
    self.clip_obs = clip_obs

  def normalize(self, input):
    return np.clip(
      (input - self.mean) / np.sqrt(self.var + self.epsilon),
      -self.clip_obs,
      self.clip_obs,
    )

  def normalize_torch(self, input, device):
    mean_torch = torch.tensor(self.mean, device=device, dtype=torch.float32)
    std_torch = torch.sqrt(
      torch.tensor(self.var + self.epsilon, device=device, dtype=torch.float32)
    )
    return torch.clamp((input - mean_torch) / std_torch, -self.clip_obs, self.clip_obs)

  def update_normalizer(self, rollouts, expert_loader):
    policy_data_generator = rollouts.feed_forward_generator_amp(
      None, mini_batch_size=expert_loader.batch_size
    )
    expert_data_generator = expert_loader.dataset.feed_forward_generator_amp(
      expert_loader.batch_size
    )

    for expert_batch, policy_batch in zip(expert_data_generator, policy_data_generator):
      self.update(torch.vstack(tuple(policy_batch) + tuple(expert_batch)).cpu().numpy())


def _dist_is_initialized() -> bool:
  return torch.distributed.is_available() and torch.distributed.is_initialized()


def init_distributed_training(
  local_rank: int, global_rank: int, world_size: int
) -> None:
  """Bind the process to its GPU, then initialize the NCCL process group.

  ``cuda.set_device`` must run before ``init_process_group`` so NCCL knows the
  rank-to-GPU mapping (avoids hangs at barrier/broadcast).
  """
  if _dist_is_initialized():
    return
  torch.cuda.set_device(local_rank)
  init_kwargs = {
    "backend": "nccl",
    "rank": global_rank,
    "world_size": world_size,
  }
  try:
    torch.distributed.init_process_group(
      **init_kwargs, device_id=torch.device("cuda", local_rank)
    )
  except (TypeError, ValueError):
    torch.distributed.init_process_group(**init_kwargs)


def distributed_barrier(tag: str = "", timeout_s: float = 1800.0) -> None:
  """Block until all ranks reach the same point (avoids collective desync).

  Uses ``barrier()`` (NCCL-safe). ``monitored_barrier`` only supports GLOO.
  ``timeout_s`` is informational only for NCCL; use logs to see which rank is slow.
  """
  del timeout_s  # NCCL barrier has no timeout; kept for call-site documentation
  if not _dist_is_initialized():
    return
  rank = torch.distributed.get_rank()
  if tag:
    print(f"[dist][rank {rank}] barrier: {tag}", flush=True)
  torch.distributed.barrier()


def broadcast_log_dir_name(log_root: str, run_name: str, global_rank: int) -> str:
  """Rank 0 picks the run folder name; all ranks receive the same string."""
  if not _dist_is_initialized():
    stamp = _datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{stamp}_{run_name}" if run_name else stamp
    return os.path.join(log_root, name)
  if global_rank == 0:
    stamp = _datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{stamp}_{run_name}" if run_name else stamp
  else:
    name = None
  names = [name]
  torch.distributed.broadcast_object_list(names, src=0)
  return os.path.join(log_root, names[0])


def broadcast_module_parameters(module: torch.nn.Module, src: int = 0) -> None:
  """Broadcast module parameters in-place via NCCL (faster than pickle state_dict)."""
  if not _dist_is_initialized():
    return
  for param in module.parameters():
    torch.distributed.broadcast(param.data, src=src)


def distributed_sync_empirical_normalization(norm_module: torch.nn.Module) -> None:
  """Synchronize EmpiricalNormalization statistics across all ranks.

  This makes the running mean/var consistent across multi-GPU training ranks.
  Expects a module with buffers: ``_mean`` (1, D), ``_var`` (1, D), ``count`` (long scalar).
  """
  if not _dist_is_initialized():
    return
  if not (
    hasattr(norm_module, "_mean")
    and hasattr(norm_module, "_var")
    and hasattr(norm_module, "count")
  ):
    return

  mean = norm_module._mean
  var = norm_module._var
  count = norm_module.count
  if mean is None or var is None or count is None:
    return

  # use float64 for numerical stability
  mean64 = mean.detach().to(dtype=torch.float64)
  var64 = var.detach().to(dtype=torch.float64)
  cnt64 = count.detach().to(dtype=torch.float64)

  # local moments
  sum1 = mean64 * cnt64
  sum2 = (var64 + mean64.square()) * cnt64

  # reduce across ranks
  torch.distributed.all_reduce(cnt64, op=torch.distributed.ReduceOp.SUM)
  torch.distributed.all_reduce(sum1, op=torch.distributed.ReduceOp.SUM)
  torch.distributed.all_reduce(sum2, op=torch.distributed.ReduceOp.SUM)

  # avoid div-by-zero during very early steps
  eps = torch.tensor(1.0, device=cnt64.device, dtype=cnt64.dtype)
  denom = torch.maximum(cnt64, eps)
  global_mean = sum1 / denom
  global_ex2 = sum2 / denom
  global_var = torch.clamp(global_ex2 - global_mean.square(), min=0.0)

  mean.copy_(global_mean.to(dtype=mean.dtype))
  var.copy_(global_var.to(dtype=var.dtype))
  if hasattr(norm_module, "_std"):
    std = norm_module._std
    std.copy_(torch.sqrt(global_var).to(dtype=std.dtype))
  count.copy_(cnt64.to(dtype=count.dtype))


def distributed_sync_running_mean_std(
  norm: RunningMeanStd, device: str | torch.device = "cuda"
) -> None:
  """Synchronize RunningMeanStd/Normalizer statistics across all ranks.

  This is used for AMP normalizers (numpy stats) so that each rank normalizes
  observations with the same running mean/var.
  """
  if not _dist_is_initialized():
    return
  if norm is None:
    return

  # Convert numpy stats to torch tensors on GPU (or provided device)
  mean = torch.as_tensor(norm.mean, dtype=torch.float64, device=device)
  var = torch.as_tensor(norm.var, dtype=torch.float64, device=device)
  cnt = torch.as_tensor(float(norm.count), dtype=torch.float64, device=device)

  sum1 = mean * cnt
  sum2 = (var + mean.square()) * cnt

  torch.distributed.all_reduce(cnt, op=torch.distributed.ReduceOp.SUM)
  torch.distributed.all_reduce(sum1, op=torch.distributed.ReduceOp.SUM)
  torch.distributed.all_reduce(sum2, op=torch.distributed.ReduceOp.SUM)

  denom = torch.maximum(cnt, torch.tensor(1.0, device=cnt.device, dtype=cnt.dtype))
  global_mean = sum1 / denom
  global_ex2 = sum2 / denom
  global_var = torch.clamp(global_ex2 - global_mean.square(), min=0.0)

  norm.mean = global_mean.detach().cpu().numpy()
  norm.var = global_var.detach().cpu().numpy()
  norm.count = float(cnt.detach().cpu().item())


def resolve_nn_activation(act_name: str) -> torch.nn.Module:
  if act_name == "elu":
    return torch.nn.ELU()
  elif act_name == "selu":
    return torch.nn.SELU()
  elif act_name == "relu":
    return torch.nn.ReLU()
  elif act_name == "crelu":
    return torch.nn.CELU()
  elif act_name == "lrelu":
    return torch.nn.LeakyReLU()
  elif act_name == "tanh":
    return torch.nn.Tanh()
  elif act_name == "sigmoid":
    return torch.nn.Sigmoid()
  elif act_name == "identity":
    return torch.nn.Identity()
  else:
    raise ValueError(f"Invalid activation function '{act_name}'.")


def split_and_pad_trajectories(tensor, dones):
  """Splits trajectories at done indices. Then concatenates them and pads with zeros up to the length og the longest trajectory.
  Returns masks corresponding to valid parts of the trajectories
  Example:
      Input: [ [a1, a2, a3, a4 | a5, a6],
               [b1, b2 | b3, b4, b5 | b6]
              ]

      Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
               [a5, a6, 0, 0],   |    [True, True, False, False],
               [b1, b2, 0, 0],   |    [True, True, False, False],
               [b3, b4, b5, 0],  |    [True, True, True, False],
               [b6, 0, 0, 0]     |    [True, False, False, False],
              ]                  | ]

  Assumes that the inputy has the following dimension order: [time, number of envs, additional dimensions]
  """
  dones = dones.clone()
  dones[-1] = 1
  # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
  flat_dones = dones.transpose(1, 0).reshape(-1, 1)

  # Get length of trajectory by counting the number of successive not done elements
  done_indices = torch.cat(
    (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0])
  )
  trajectory_lengths = done_indices[1:] - done_indices[:-1]
  trajectory_lengths_list = trajectory_lengths.tolist()
  # Extract the individual trajectories
  trajectories = torch.split(
    tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list
  )
  # add at least one full length trajectory
  trajectories = trajectories + (
    torch.zeros(tensor.shape[0], *tensor.shape[2:], device=tensor.device),
  )
  # pad the trajectories to the length of the longest trajectory
  padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
  # remove the added tensor
  padded_trajectories = padded_trajectories[:, :-1]

  trajectory_masks = trajectory_lengths > torch.arange(
    0, tensor.shape[0], device=tensor.device
  ).unsqueeze(1)
  return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
  """Does the inverse operation of  split_and_pad_trajectories()"""
  # Need to transpose before and after the masking to have proper reshaping
  return (
    trajectories.transpose(1, 0)[masks.transpose(1, 0)]
    .view(-1, trajectories.shape[0], trajectories.shape[-1])
    .transpose(1, 0)
  )


def store_code_state(logdir, repositories) -> list:
  git_log_dir = os.path.join(logdir, "git")
  os.makedirs(git_log_dir, exist_ok=True)
  file_paths = []
  for repository_file_path in repositories:
    try:
      repo = git.Repo(repository_file_path, search_parent_directories=True)
      t = repo.head.commit.tree
    except Exception:
      print(f"Could not find git repository in {repository_file_path}. Skipping.")
      # skip if not a git repository
      continue
    # get the name of the repository
    repo_name = pathlib.Path(repo.working_dir).name
    diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")
    # check if the diff file already exists
    if os.path.isfile(diff_file_name):
      continue
    # write the diff file
    print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
    with open(diff_file_name, "x", encoding="utf-8") as f:
      content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)}"
      f.write(content)
    # add the file path to the list of files to be uploaded
    file_paths.append(diff_file_name)
  return file_paths


def string_to_callable(name: str) -> Callable:
  """Resolves the module and function names to return the function.

  Args:
      name (str): The function name. The format should be 'module:attribute_name'.

  Raises:
      ValueError: When the resolved attribute is not a function.
      ValueError: When unable to resolve the attribute.

  Returns:
      Callable: The function loaded from the module.
  """
  try:
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    callable_object = getattr(mod, attr_name)
    # check if attribute is callable
    if callable(callable_object):
      return callable_object
    else:
      raise ValueError(f"The imported object is not callable: '{name}'")
  except AttributeError as e:
    msg = (
      "We could not interpret the entry as a callable object. The format of input should be"
      f" 'module:attribute_name'\nWhile processing input '{name}', received the error:\n {e}."
    )
    raise ValueError(msg)
