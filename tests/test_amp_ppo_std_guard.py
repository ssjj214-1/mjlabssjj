"""Regression tests for the policy-std robustness guard in MULTIAMPPPO.

A non-finite or negative ``std`` parameter makes ``torch.distributions.Normal``
sampling raise ``"normal expects all elements of std >= 0.0"``. The training run
on rough terrain hit this when an extreme physics state produced a NaN gradient.
``_clamp_policy_std`` must sanitize the parameter back to a valid range.
"""

import torch

from mjlab.third_party.amp_rsl_rl.algorithms.multi_amp_ppo import MULTIAMPPPO


class _DummyPolicy:
  def __init__(self, std_values: torch.Tensor) -> None:
    self.std = torch.nn.Parameter(std_values.clone())


def _make_algo(min_std: float, num_actions: int) -> MULTIAMPPPO:
  # Bypass the heavy __init__; only _clamp_policy_std state is needed.
  algo = MULTIAMPPPO.__new__(MULTIAMPPPO)
  algo.min_std = torch.full((num_actions,), min_std)
  return algo


def test_clamp_policy_std_recovers_from_nan_and_negative():
  num_actions = 13
  bad = torch.tensor(
    [float("nan"), float("inf"), float("-inf"), -1.0, 0.0] + [0.3] * (num_actions - 5)
  )
  algo = _make_algo(min_std=0.05, num_actions=num_actions)
  algo.policy = _DummyPolicy(bad)

  algo._clamp_policy_std()

  std = algo.policy.std.data
  assert torch.isfinite(std).all(), "std still has non-finite entries"
  assert (std >= 0.05 - 1e-9).all(), "std fell below the configured floor"
  # The whole point: sampling must no longer raise.
  torch.distributions.Normal(torch.zeros(num_actions), std).sample()


def test_clamp_policy_std_preserves_healthy_values():
  num_actions = 13
  healthy = torch.full((num_actions,), 0.4)
  algo = _make_algo(min_std=0.05, num_actions=num_actions)
  algo.policy = _DummyPolicy(healthy)

  algo._clamp_policy_std()

  assert torch.allclose(algo.policy.std.data, healthy), (
    "healthy std values should be left untouched"
  )


def test_nonfinite_grad_norm_is_detected():
  # Mirrors the optimizer-step guard: a non-finite total grad norm must be
  # detectable so the step can be skipped instead of poisoning the weights.
  p = torch.nn.Parameter(torch.zeros(4))
  p.grad = torch.tensor([float("nan"), 1.0, 2.0, 3.0])
  grad_norm = torch.nn.utils.clip_grad_norm_([p], max_norm=1.0)
  assert not torch.isfinite(grad_norm)
