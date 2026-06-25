"""Headless probe: play an external (e.g. Isaac Sim / ultra_run_lab) checkpoint in mjlab.

Loads only inference-relevant weights (actor + HIM encoder); critic / estimator.target
may differ in shape between Isaac and mjlab and are skipped via strict=False.

Usage:
  uv run python play_isaac_ckpt_probe.py \\
    --ckpt /home/ps/ultra_run_lab/logs/ultra_game_yaw_run/2026-06-02_18-36-40/model_19999.pt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts.play import _maybe_wrap_him_policy
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

# Baseline AMP-HIM task (same reward/obs layout as ultra_run_lab Isaac task).
TASK = "Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM"


def load_ckpt_inference(runner, path: str, device: str) -> list[str]:
  """Load checkpoint for play; return list of shape-mismatched keys (skipped)."""
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
  # Discriminator not needed for inference; load best-effort.
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
  ap.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
  ap.add_argument("--steps", type=int, default=500)
  ap.add_argument("--num_envs", type=int, default=4)
  ap.add_argument("--warmup", type=int, default=80, help="Steps before measuring drift")
  args = ap.parse_args()

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  ckpt = Path(args.ckpt).resolve()
  if not ckpt.exists():
    raise FileNotFoundError(ckpt)

  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg = load_rl_cfg(TASK)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  skipped = load_ckpt_inference(runner, str(ckpt), device)
  if skipped:
    print(f"[INFO] skipped / mismatched keys ({len(skipped)}):")
    for k in skipped:
      print(f"  - {k}")

  policy = runner.get_inference_policy(device=device)
  policy = _maybe_wrap_him_policy(policy, runner, env, device)

  robot = env.unwrapped.scene["robot"]
  twist = env.unwrapped.command_manager.get_term("twist")
  foot_ids, _ = robot.find_bodies(["ankle_pitch_l_link", "ankle_pitch_r_link"])

  def force_zero_cmd() -> None:
    twist.vel_command_b[:, 0] = 0.0
    twist.vel_command_b[:, 1] = 0.0
    twist.vel_command_b[:, 2] = 0.0
    twist.vel_command_w[:, 0] = 0.0
    twist.vel_command_w[:, 1] = 0.0
    twist.vel_command_w[:, 2] = 0.0
    twist.heading_command = False

  obs, _ = env.reset()
  force_zero_cmd()

  vx_log, vy_log, wz_log = [], [], []
  jvel_log, footz_log, footvz_log = [], [], []
  basexy0 = None
  basexy_last = None
  reset_count = 0

  with torch.inference_mode():
    for t in range(args.steps):
      actions = policy(obs)
      obs, _, dones, _ = env.step(actions)
      force_zero_cmd()
      lin = robot.data.root_link_lin_vel_b[:, :2].detach().cpu().numpy()
      ang = robot.data.root_link_ang_vel_b[:, 2].detach().cpu().numpy()
      jvel = robot.data.joint_vel.detach()
      footz = robot.data.body_link_pos_w[:, foot_ids, 2].detach().cpu().numpy()
      footvz = robot.data.body_link_lin_vel_w[:, foot_ids, 2].detach().cpu().numpy()
      basexy = robot.data.root_link_pos_w[:, :2].detach().cpu().numpy()
      if t >= args.warmup:
        vx_log.append(lin[:, 0])
        vy_log.append(lin[:, 1])
        wz_log.append(ang)
        jvel_log.append(jvel.square().mean(dim=1).sqrt().cpu().numpy())
        footz_log.append(footz)
        footvz_log.append(footvz)
        if basexy0 is None:
          basexy0 = basexy.copy()
        basexy_last = basexy.copy()
      reset_count += int(dones.sum().item())

  vx = np.stack(vx_log)
  vy = np.stack(vy_log)
  wz = np.stack(wz_log)
  jvel = np.stack(jvel_log)
  footz = np.stack(footz_log)
  footvz = np.stack(footvz_log)

  print(
    f"\n=== zero-cmd standstill probe ===\n"
    f"task={TASK}\n"
    f"ckpt={ckpt}\n"
    f"steps={args.steps} warmup={args.warmup} num_envs={args.num_envs} "
    f"resets={reset_count}\n"
  )
  print("-- base velocity (net COM motion, ~0 even when micro-stepping) --")
  print(f"{'metric':8s} {'mean':>9s} {'std':>9s} {'|max|':>9s}")
  for name, arr in [("vx", vx), ("vy", vy), ("wz", wz)]:
    flat = arr.reshape(-1)
    print(f"{name:8s} {flat.mean():9.4f} {flat.std():9.4f} {np.abs(flat).max():9.4f}")

  print("\n-- JITTER / STEPPING metrics (the real tell) --")
  print(f"joint_vel_rms   mean={jvel.mean():.4f}  max={jvel.max():.4f} rad/s")
  fz_dev = footz - np.median(footz, axis=0, keepdims=True)
  print(f"foot_z_std      L={footz[:, :, 0].std():.5f}  R={footz[:, :, 1].std():.5f} m")
  print(
    f"foot_lift_max   L={fz_dev[:, :, 0].max():.5f}  R={fz_dev[:, :, 1].max():.5f} m"
    "  (>0.01 = clearly stepping)"
  )
  print(
    f"foot_vz_absmean L={np.abs(footvz[:, :, 0]).mean():.4f}  "
    f"R={np.abs(footvz[:, :, 1]).mean():.4f} m/s"
  )
  if basexy0 is not None and basexy_last is not None:
    drift = np.linalg.norm(basexy_last - basexy0, axis=1)
    print(
      f"base_xy_drift   mean={drift.mean():.4f} m over "
      f"{(args.steps - args.warmup) * 0.01:.1f}s"
    )

  env.close()


if __name__ == "__main__":
  main()
