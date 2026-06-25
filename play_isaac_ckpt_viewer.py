"""Play an external (Isaac Sim / ultra_run_lab) checkpoint in the mjlab viewer.

mjlab's ``scripts/play.py`` uses a strict ``load_state_dict`` (AMP-HIM runner),
which crashes on an Isaac checkpoint because its critic / estimator.target layers
have a different shape (privileged-obs dim differs between the two stacks). This
script loads only the inference-relevant weights (actor + HIM encoder) via
``strict=False`` filtering, then launches the native MuJoCo viewer with the same
keyboard teleop + real-time speed printing as ``scripts/play.py``.

Keys (native viewer): W/S forward +/-, Q/E yaw left/right, Z/C strafe, X zero.

Usage:
  uv run python play_isaac_ckpt_viewer.py \\
    --ckpt /home/ps/ultra_run_lab/logs/ultra_game_yaw_run/2026-06-02_18-36-40/model_19999.pt
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts.play import _maybe_wrap_him_policy, _VelocityTeleop
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

# Baseline AMP-HIM task: same reward/obs layout AND same PD as the Isaac task.
TASK = "Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM"

# Joint order the Isaac/ultra_run_lab policy was trained in (obs joint slots +
# action output). mjlab orders joints differently (per-leg DFS), so an Isaac
# checkpoint must have its joint slots permuted or it flails instantly. This is
# the SAME list baked into tools/export_mjlab_onnx.py.
ISAAC_JOINT_NAMES = [
  "hip_yaw_l_joint",
  "hip_yaw_r_joint",
  "waist_yaw_joint",
  "hip_roll_l_joint",
  "hip_roll_r_joint",
  "shoulder_pitch_l_joint",
  "shoulder_pitch_r_joint",
  "hip_pitch_l_joint",
  "hip_pitch_r_joint",
  "knee_pitch_l_joint",
  "knee_pitch_r_joint",
  "ankle_pitch_l_joint",
  "ankle_pitch_r_joint",
]


class _IsaacOrderPolicy:
  """Make an Isaac-order actor work inside a mjlab-order env.

  The per-frame obs layout is ``[ang_vel(3), gravity(3), command(3),
  joint_pos(n), joint_vel(n), last_action(n)]`` (joint block starts at 9). We
  permute the three joint blocks mjlab->Isaac before the (history) policy, and
  permute the policy's action output Isaac->mjlab before it reaches the env.
  """

  def __init__(self, inner, mjlab_to_isaac, isaac_to_mjlab, n_j, jp_start=9):
    self._inner = inner
    self._m2i = mjlab_to_isaac
    self._i2m = isaac_to_mjlab
    self._n = int(n_j)
    self._jp = int(jp_start)

  def _perm_frame(self, frame):
    jp, jv = self._jp, self._jp + self._n
    act = jv + self._n
    head = frame[:, :jp]
    jpos = frame[:, jp:jv].index_select(1, self._m2i)
    jvel = frame[:, jv:act].index_select(1, self._m2i)
    lact = frame[:, act : act + self._n].index_select(1, self._m2i)
    return torch.cat([head, jpos, jvel, lact], dim=1)

  def __call__(self, obs):
    frame = obs["actor"] if hasattr(obs, "keys") else obs
    permuted = self._perm_frame(frame)
    out = self._inner({"actor": permuted})
    return out.index_select(1, self._i2m)


def _build_perms(mjlab_names, device):
  i2m = torch.tensor(
    [ISAAC_JOINT_NAMES.index(n) for n in mjlab_names], dtype=torch.long, device=device
  )
  m2i = torch.tensor(
    [mjlab_names.index(n) for n in ISAAC_JOINT_NAMES], dtype=torch.long, device=device
  )
  return i2m, m2i


def load_ckpt_inference(runner, path: str, device: str) -> list[str]:
  """Load only matching-shape weights (actor + HIM encoder). Returns skipped keys."""
  loaded = torch.load(path, map_location=device, weights_only=False)
  sd = loaded["model_state_dict"]
  model_sd = runner.alg.policy.state_dict()
  filtered = {}
  skipped: list[str] = []
  for k, v in sd.items():
    if k not in model_sd:
      skipped.append(f"{k} (unexpected)")
      continue
    if v.shape != model_sd[k].shape:
      skipped.append(f"{k}: ckpt{tuple(v.shape)} vs model{tuple(model_sd[k].shape)}")
      continue
    filtered[k] = v
  runner.alg.policy.load_state_dict(filtered, strict=False)
  if "discriminator_state_dict" in loaded:
    for k, state in loaded["discriminator_state_dict"].items():
      if k in runner.alg.discriminator_dict:
        try:
          runner.alg.discriminator_dict[k].load_state_dict(state)
        except RuntimeError:
          pass
  return skipped


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True, help="Path to external .pt checkpoint")
  ap.add_argument("--num_envs", type=int, default=1)
  ap.add_argument("--init_vx", type=float, default=0.0)
  ap.add_argument("--init_vy", type=float, default=0.0)
  ap.add_argument("--init_wz", type=float, default=0.0)
  ap.add_argument("--device", default=None)
  ap.add_argument("--viewer", choices=["auto", "native", "viser"], default="auto")
  ap.add_argument("--no_terminations", action="store_true")
  ap.add_argument(
    "--no_perm",
    action="store_true",
    help="Skip ISAAC<->mjlab joint reorder (will flail; for comparison only).",
  )
  args = ap.parse_args()

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = args.num_envs
  if args.no_terminations:
    env_cfg.terminations = {}
    print("[INFO] terminations disabled")
  agent_cfg = load_rl_cfg(TASK)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  skipped = load_ckpt_inference(runner, args.ckpt, device)
  print(f"[INFO] loaded actor + HIM encoder from: {args.ckpt}")
  if skipped:
    print(f"[INFO] skipped {len(skipped)} mismatched/unused keys (critic/estimator):")
    for k in skipped:
      print(f"  - {k}")

  policy = runner.get_inference_policy(device=device)
  policy = _maybe_wrap_him_policy(policy, runner, env, device)

  if not args.no_perm:
    robot = env.unwrapped.scene["robot"]
    mjlab_names = list(robot.joint_names)
    if sorted(mjlab_names) != sorted(ISAAC_JOINT_NAMES):
      raise RuntimeError(
        f"joint name sets differ; cannot build permutation.\n"
        f"  mjlab-only={set(mjlab_names) - set(ISAAC_JOINT_NAMES)}\n"
        f"  isaac-only={set(ISAAC_JOINT_NAMES) - set(mjlab_names)}"
      )
    i2m, m2i = _build_perms(mjlab_names, device)
    n_j = len(mjlab_names)
    policy = _IsaacOrderPolicy(policy, m2i, i2m, n_j)
    print(
      "[INFO] ISAAC<->mjlab joint reorder ENABLED "
      "(obs mjlab->ISAAC in, actions ISAAC->mjlab out)"
    )
  else:
    print("[WARN] --no_perm: feeding raw mjlab-order obs to Isaac actor (will flail)")

  teleop = None
  if "twist" in env.unwrapped.command_manager.active_terms:
    teleop = _VelocityTeleop(
      env, init_vx=args.init_vx, init_vy=args.init_vy, init_wz=args.init_wz
    )
    print(
      "[INFO] Keyboard velocity teleop enabled:\n"
      "       W/S = forward +/-, Q/E = yaw left/right, "
      "Z/C = strafe left/right, X = zero\n"
      f"       initial cmd: vx={args.init_vx:+.2f} vy={args.init_vy:+.2f} "
      f"wz={args.init_wz:+.2f}"
    )

  if args.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved = "native" if has_display else "viser"
  else:
    resolved = args.viewer

  key_callback = teleop.handle_key if teleop is not None else None

  if resolved == "native":
    NativeMujocoViewer(env, policy, key_callback=key_callback).run()
  else:
    if teleop is not None:
      print(
        "[WARN] keyboard keys only wired for native viewer; speed printing still "
        "works, use the Viser joystick GUI to steer."
      )
    ViserPlayViewer(env, policy).run()

  env.close()


if __name__ == "__main__":
  main()
