"""Ultra GameYaw v17: V9 without the HIM velocity estimator (and no terrain).

v17 answers the sim2real-asymmetry hypothesis: the HIM GRU velocity estimator
might be the component that behaves differently on hardware (its estimate can
drift/bias left-right in ways the flat-plane sim never exercises), so we retrain
the exact V9 policy *without* it.

What v17 keeps from V9 (unchanged):

* Environment: the full V9 env — hist10 reward set, hist10 performance-gated
  velocity curriculum (cap ramp with retreat), hist10 command sampling
  (stand/walk/run buckets, ``max_lin_accel=3.0``, heading hold), per-joint PD,
  ``action_scale``, sim dt/decimation, domain randomization and command slew.
* Terrain: V9 is already the **flat**-plane task (no rough/gravel terrain), so
  "no terrain" needs no change — v17 stays on the flat plane.
* AMP: the same multi-style WGAN AMP stack, expert clips, MULTIAMPPPO update,
  symmetry mirror loss and 10-frame actor/critic observation history.

What v17 changes vs V9 (the only difference):

* ``use_him=False`` on the runner. The AMP env wrapper stops exposing
  ``num_one_step_obs`` to :class:`MultiAmpOnPolicyRunner`, so instead of the
  HIM path (GRU encoder -> estimated base velocity + latent -> small recent
  actor window) the runner builds a **plain MLP actor over the full 10-frame
  stacked observation** (this is exactly the hist10 architecture, which never
  had HIM). No estimator network is created and all estimator supervision /
  swap loss is skipped.

Deployment note: a no-HIM policy has a different ONNX I/O contract than the
HIM policies (no estimated-velocity output, actor input is the flat stacked
observation). Deploying v17 to ultra2026_rl_sdk will need a matching export +
loader; the existing HIM ONNX/UltraYawPolicy path does not apply as-is.

Trains independently from the other versions (separate experiment/project).
"""

from __future__ import annotations

from mjlab.tasks.velocity.config.ultra_game_yaw.amp_him import RslRlAmpHimRunnerCfg
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs_v9 import (
  ultra_game_yaw_amp_him_v9_runner_cfg,
  ultra_game_yaw_v9_env_cfg,
)


def ultra_game_yaw_v17_env_cfg(play: bool = False):
  """v17 env == v9 env verbatim (flat plane, hist10 rewards + curriculum).

  Nothing about the environment changes when HIM is removed: the observation
  groups, rewards, curriculum and terrain are identical. The HIM removal is a
  runner-side change only (see :func:`ultra_game_yaw_amp_him_v17_runner_cfg`).
  """
  return ultra_game_yaw_v9_env_cfg(play=play)


def ultra_game_yaw_amp_him_v17_runner_cfg() -> RslRlAmpHimRunnerCfg:
  """v17 runner: V9 runner with the HIM estimator disabled."""
  cfg = ultra_game_yaw_amp_him_v9_runner_cfg()
  cfg.use_him = False
  cfg.experiment_name = "ultra_game_yaw_amp_him_v17"
  cfg.wandb_project = "ultra_game_yaw_amp_him_v17"
  return cfg
