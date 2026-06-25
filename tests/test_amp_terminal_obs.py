"""Tests for AMP terminal observation handling."""

import torch

from mjlab.third_party.amp_rsl_rl.runners.multi_amp_on_policy_runner import (
  replace_reset_amp_obs_with_terminal,
)


def test_replace_reset_amp_obs_with_terminal_uses_pre_reset_rows() -> None:
  """Reset env rows should use terminal/pre-reset AMP obs, not post-reset obs."""

  next_amp_obs = torch.tensor(
    [
      [10.0, 11.0],
      [20.0, 21.0],
      [30.0, 31.0],
    ]
  )
  terminal_amp_obs = torch.tensor(
    [
      [1.0, 2.0],
      [3.0, 4.0],
      [5.0, 6.0],
    ]
  )
  reset_env_ids = torch.tensor([0, 2])

  actual = replace_reset_amp_obs_with_terminal(
    next_amp_obs=next_amp_obs,
    reset_env_ids=reset_env_ids,
    terminal_amp_obs=terminal_amp_obs,
  )

  expected = torch.tensor(
    [
      [1.0, 2.0],
      [20.0, 21.0],
      [5.0, 6.0],
    ]
  )
  torch.testing.assert_close(actual, expected)
  torch.testing.assert_close(
    next_amp_obs,
    torch.tensor(
      [
        [10.0, 11.0],
        [20.0, 21.0],
        [30.0, 31.0],
      ]
    ),
  )
