"""Ultra GameYaw v5: V4 standstill fixes + HIM + HoST-style smoothness loss.

Motivation. A colleague's HIM policy does NOT jitter on the real robot at the
*baseline* (Isaac) PD gains — they did not have to lower kp/kd. The difference
is that they add a network-level **smoothness loss + regularization**, instead
of relying only on the reward-side ``action_rate_l2`` penalty. v5 ports the
exact mechanism used in the HoST repo (``HoST/rsl_rl/.../algorithms/ppo.py``).

HoST's "smooth loss" (this is what v5 now uses):

* Build a perturbed observation by interpolating along the consecutive-step
  direction with a RANDOM per-sample magnitude::

      mix_w  = cont * (rand - 0.5) * 2          # in cont * [-1, 1]
      s_mix  = s_t + mix_w * (s_{t+1} - s_t)

  where ``cont = 1 - done`` (so no perturbation across episode resets).
* Penalize the resulting change in the policy mean AND the value::

      L = c_pi * ||mu(s_t) - mu(s_mix)||^2 + c_V * ||V(s_t) - V(s_mix)||^2

* Coefficients follow HoST's bound formula::

      epsilon = lower / (upper - lower)
      c_pi    = upper * epsilon
      c_V     = value_smoothness_coef * c_pi

There is NO separate action-L2 term in HoST's PPO — the smoothness loss (policy
+ value) IS the regularization. This is different from the reward-side
``action_rate_l2``: it constrains the policy/value FUNCTIONS directly (strong
gradient) and so kills the obs-noise -> high-frequency-action jitter that shows
up on the real robot, without lowering kp/kd.

What v5 is. v5 = **v4 env + HoST smoothness loss**. It inherits everything from
v4 (so it also fixes the standstill drift, not just the high-frequency jitter):

* ORIGINAL Isaac PD (NOT v2/v3's soft PD) and ORIGINAL expert trajectories;
* the Isaac-style PROGRESSIVE velocity curriculum;
* the stand-stability terms (``feet_slide`` on the stand style + ``stand_base_vel``);
* the Isaac-aligned heading command (0.6 / 0.5).

The ONLY thing v5 adds on top of v4 is turning on the HoST smoothness loss in
the PPO update (``multi_amp_ppo``).

Deploy note: the smoothness loss shapes the policy at train time only. Nothing
extra is needed at deploy — the exported ONNX is just a smoother actor, no change
to deploy PD. Heading coupling is inherited from v4: if a v5 policy is deployed,
set the deploy ``heading_kp`` (LocoMode.yaml) to 0.5 to match
``heading_control_stiffness``.
"""

from __future__ import annotations

from mjlab.tasks.velocity.config.ultra_game_yaw.amp_him import (
  RslRlAmpHimRunnerCfg,
  ultra_game_yaw_amp_him_runner_cfg,
)
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs_v4 import (
  ultra_game_yaw_v4_env_cfg,
)

# HoST smoothness coefficients. These mirror HoST's G1 ground config
# (``HoST/legged_gym/.../g1/g1_config_ground.py`` -> class algorithm). Tune by
# watching the deploy/sim action chatter vs tracking lag: raise the lower bound
# to smooth harder, lower it (toward 0 == OFF) to track more aggressively.
_SMOOTHNESS_UPPER_BOUND = 1.0
_SMOOTHNESS_LOWER_BOUND = 0.1
_VALUE_SMOOTHNESS_COEF = 0.1


def ultra_game_yaw_v5_env_cfg(play: bool = False):
  """v5 env == v4 env (standstill fixes + progressive curriculum). The
  smoothness loss lives in the runner/algorithm cfg, not the env."""
  return ultra_game_yaw_v4_env_cfg(play=play)


def ultra_game_yaw_amp_him_v5_runner_cfg() -> RslRlAmpHimRunnerCfg:
  """Baseline AMP-HIM runner + HoST-style smoothness loss turned on."""
  cfg = ultra_game_yaw_amp_him_runner_cfg()
  cfg.experiment_name = "ultra_game_yaw_amp_him_v5"
  cfg.wandb_project = "ultra_game_yaw_amp_him_v5"
  # Turn on the HoST smoothness loss (defaults in the alg cfg use
  # lower_bound=0.0 == off).
  cfg.algorithm = {
    **cfg.algorithm,
    "smoothness_upper_bound": _SMOOTHNESS_UPPER_BOUND,
    "smoothness_lower_bound": _SMOOTHNESS_LOWER_BOUND,
    "value_smoothness_coef": _VALUE_SMOOTHNESS_COEF,
  }
  return cfg
