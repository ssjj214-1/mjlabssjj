"""Tests for the velocity-task terrain difficulty curriculum upward gate."""

from unittest.mock import Mock

import torch

from mjlab.tasks.velocity.mdp.curriculums import terrain_levels_vel


def _make_env(distances: torch.Tensor, cmd_speed: float, ep_len_s: float = 20.0):
  """Build a mock env where each env_id walked ``distances[i]`` metres.

  All envs share the same commanded forward speed ``cmd_speed`` (m/s) so the
  commanded distance is ``cmd_speed * ep_len_s``.
  """
  n = distances.shape[0]
  env = Mock()
  env.max_episode_length_s = ep_len_s

  # env origins at 0; root link positioned so ||pos - origin|| == distance.
  env.scene.env_origins = torch.zeros(n, 3)
  asset = Mock()
  pos = torch.zeros(n, 3)
  pos[:, 0] = distances
  asset.data.root_link_pos_w = pos
  env.scene.__getitem__ = Mock(return_value=asset)

  terrain = Mock()
  terrain.cfg.terrain_generator.size = (8.0, 8.0)
  terrain.cfg.terrain_generator.sub_terrains = {}
  terrain.terrain_levels = torch.zeros(n, dtype=torch.long)
  terrain.terrain_origins = torch.zeros(1, 1, 3)
  terrain.terrain_types = torch.zeros(n, dtype=torch.long)
  env.scene.terrain = terrain

  # Command: forward-only at cmd_speed.
  cmd = torch.zeros(n, 3)
  cmd[:, 0] = cmd_speed
  env.command_manager.get_command.return_value = cmd

  return env, terrain


def test_legacy_gate_promotes_on_half_tile():
  # Without move_up_distance_frac, anything past size/2 (4 m) is promoted, even
  # though a 15 m/s command covers far more -- the fast-runner failure mode.
  distances = torch.tensor([3.0, 5.0])  # below / above 4 m
  env, terrain = _make_env(distances, cmd_speed=15.0)
  terrain_levels_vel(env, torch.tensor([0, 1]), command_name="twist")
  move_up = terrain.update_env_origins.call_args[0][1]
  assert move_up.tolist() == [False, True]


def test_commanded_distance_gate_requires_keeping_up():
  # With the fractional gate, a runner must cover >=80% of commanded distance
  # (15 m/s * 20 s * 0.8 = 240 m) to be promoted. Walking 5 m is nowhere close,
  # so terrain does NOT harden -- this is what lets speed lead difficulty.
  distances = torch.tensor([5.0, 250.0])  # well below / above 240 m
  env, terrain = _make_env(distances, cmd_speed=15.0)
  terrain_levels_vel(
    env, torch.tensor([0, 1]), command_name="twist", move_up_distance_frac=0.8
  )
  move_up = terrain.update_env_origins.call_args[0][1]
  assert move_up.tolist() == [False, True]


def test_dist_frac_logged():
  distances = torch.tensor([150.0])  # half of 15 m/s * 20 s = 300 m
  env, _ = _make_env(distances, cmd_speed=15.0)
  result = terrain_levels_vel(
    env, torch.tensor([0]), command_name="twist", move_up_distance_frac=0.8
  )
  assert "dist_frac" in result
  assert abs(result["dist_frac"].item() - 0.5) < 1e-3
