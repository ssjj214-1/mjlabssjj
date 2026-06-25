"""Headless rollout: log per-joint actuator torque when the policy runs.

Mimics the real-robot "enter LOCOMODE at standstill" scenario (zero velocity
command) and records qfrc_actuator per joint, so we can see whether the policy
drives any joint (esp. waist) toward / past its effort limit in *simulation*.

Usage:
  uv run python sim_torque_probe.py --ckpt logs/.../model_2600.pt --steps 600
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts.play import _load_runner_checkpoint, _maybe_wrap_him_policy
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V2"
DEFAULT_CKPT = (
  "logs/rsl_rl/ultra_game_yaw_amp_him_v2/"
  "2026-06-09_10-39-54_motoralign-lowkd/model_2600.pt"
)
# effort_limit (N*m) from ultra_constants.py, matched by joint-name substring.
EFFORT = {
  "hip_yaw": 250.0,
  "hip_roll": 657.0,
  "hip_pitch": 657.0,
  "knee": 657.0,
  "ankle": 400.0,
  "waist": 250.0,
  "shoulder": 140.0,
}


def effort_of(name: str) -> float:
  for k, v in EFFORT.items():
    if k in name:
      return v
  return float("nan")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", default=DEFAULT_CKPT)
  ap.add_argument("--steps", type=int, default=600)
  ap.add_argument("--num_envs", type=int, default=16)
  ap.add_argument(
    "--cmd",
    choices=["zero", "walk"],
    default="zero",
    help="zero=standstill (mimic takeover); walk=vx 1.0",
  )
  args = ap.parse_args()

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = args.num_envs
  # keep terminations so a divergence resets (we report resets too)
  agent_cfg = load_rl_cfg(TASK)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  _load_runner_checkpoint(runner, str(Path(args.ckpt).resolve()), device)
  policy = runner.get_inference_policy(device=device)
  policy = _maybe_wrap_him_policy(policy, runner, env, device)

  robot = env.unwrapped.scene["robot"]
  jnames = list(robot.joint_names)
  twist = env.unwrapped.command_manager.get_term("twist")
  vx = 1.0 if args.cmd == "walk" else 0.0

  def force_cmd():
    twist.vel_command_b[:, 0] = vx
    twist.vel_command_b[:, 1] = 0.0
    twist.vel_command_b[:, 2] = 0.0
    twist.vel_command_w[:, 0] = vx
    twist.vel_command_w[:, 1] = 0.0
    twist.vel_command_w[:, 2] = 0.0

  obs, _ = env.reset()
  force_cmd()
  tau_log = []
  reset_count = 0
  with torch.inference_mode():
    for _t in range(args.steps):
      actions = policy(obs)
      obs, _, dones, _ = env.step(actions)
      force_cmd()
      tau = robot.data.qfrc_actuator.detach().cpu().numpy()  # (num_envs, nJ)
      tau_log.append(tau)
      reset_count += int(dones.sum().item())

  tau_log = np.stack(tau_log, axis=0)  # (T, num_envs, nJ)
  T = tau_log.shape[0]

  print(
    f"\n=== cmd={args.cmd}  ckpt={Path(args.ckpt).name}  "
    f"steps={T}  num_envs={args.num_envs}  resets={reset_count} ==="
  )
  print(
    f"{'joint':22s} {'|tau|mean':>10s} {'|tau|std':>9s} "
    f"{'|tau|max':>9s} {'effort':>7s} {'%limit':>7s}"
  )
  flat = np.abs(tau_log).reshape(-1, len(jnames))
  for i, nm in enumerate(jnames):
    a = flat[:, i]
    lim = effort_of(nm)
    pct = 100 * a.max() / lim if lim == lim else float("nan")
    print(
      f"{nm:22s} {a.mean():10.2f} {a.std():9.2f} {a.max():9.2f} {lim:7.0f} {pct:6.0f}%"
    )

  # transient: first 40 steps (the "enter network" instant), env 0
  print(
    "\n=== takeover transient: max |tau| in first 40 steps (per env, worst env) ==="
  )
  trans = np.abs(tau_log[:40]).max(axis=0)  # (num_envs, nJ)
  worst = trans.max(axis=0)  # (nJ,)
  for i, nm in enumerate(jnames):
    lim = effort_of(nm)
    pct = 100 * worst[i] / lim if lim == lim else float("nan")
    flag = "  <-- 接近/超限!" if pct > 80 else ""
    print(f"  {nm:22s} max|tau|={worst[i]:8.2f}  ({pct:5.0f}% of {lim:.0f}){flag}")

  np.save("/home/ps/GMR/sim_tau_log.npy", tau_log)
  with open("/home/ps/GMR/sim_tau_jnames.txt", "w") as f:
    f.write("\n".join(jnames))
  print("\nsaved /home/ps/GMR/sim_tau_log.npy")
  env.close()


if __name__ == "__main__":
  main()
