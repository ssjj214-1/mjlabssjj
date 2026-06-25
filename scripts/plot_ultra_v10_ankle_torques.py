"""Plot Ultra V10 ankle torque curves from a trained checkpoint."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts.play import _load_runner_checkpoint, _maybe_wrap_him_policy
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class Config:
  checkpoint: str = (
    "logs/rsl_rl/ultra_game_yaw_amp_him_v10/"
    "2026-06-17_19-22-35_passive_roll/model_8400.pt"
  )
  task: str = "Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V10"
  steps: int = 2000
  warmup_steps: int = 100
  device: str | None = None
  output_dir: str | None = None
  vx: float = 1.0
  vy: float = 0.0
  wz: float = 0.0
  no_terminations: bool = True


def _set_constant_twist(
  env: ManagerBasedRlEnv, vx: float, vy: float, wz: float
) -> None:
  term = env.command_manager.get_term("twist")
  cfg = term.cfg
  cfg.heading_command = False
  cfg.ranges.heading = None
  cfg.rel_standing_envs = 0.0
  cfg.rel_heading_envs = 0.0
  cfg.rel_world_envs = 0.0
  cfg.rel_forward_envs = 0.0
  cfg.init_velocity_prob = 0.0
  cfg.resampling_time_range = (1.0e9, 1.0e9)

  original_compute = term.compute

  def compute(dt: float) -> None:
    original_compute(dt)
    term.vel_command_b[:, 0] = vx
    term.vel_command_b[:, 1] = vy
    term.vel_command_b[:, 2] = wz
    term.vel_command_w[:, 0] = vx
    term.vel_command_w[:, 1] = vy
    term.vel_command_w[:, 2] = wz

  term.compute = compute  # type: ignore[method-assign]
  term.compute(0.0)


def _index_by_name(names: tuple[str, ...], targets: tuple[str, ...]) -> dict[str, int]:
  lookup = {name: i for i, name in enumerate(names)}
  return {name: lookup[name] for name in targets if name in lookup}


def _make_policy(task: str, checkpoint: Path, env: RslRlVecEnvWrapper, device: str):
  agent_cfg = load_rl_cfg(task)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  _load_runner_checkpoint(runner, str(checkpoint), device)
  policy = runner.get_inference_policy(device=device)
  return _maybe_wrap_him_policy(policy, runner, env, device)


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
  if not rows:
    raise RuntimeError("No rows collected.")
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def _plot(path: Path, rows: list[dict[str, float]], columns: list[str]) -> None:
  t = [row["time_s"] for row in rows]
  fig, axes = plt.subplots(len(columns), 1, figsize=(14, 2.6 * len(columns)))
  if len(columns) == 1:
    axes = [axes]
  for ax, col in zip(axes, columns, strict=True):
    ax.plot(t, [row[col] for row in rows], linewidth=1.0)
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.25)
  axes[-1].set_xlabel("time [s]")
  fig.tight_layout()
  fig.savefig(path, dpi=160)
  plt.close(fig)


def _add_base_velocity(
  row: dict[str, float], robot, vx: float, vy: float, wz: float
) -> None:
  lin = robot.data.root_link_lin_vel_b[0]
  ang = robot.data.root_link_ang_vel_b[0]
  row["cmd.vx"] = vx
  row["cmd.vy"] = vy
  row["cmd.wz"] = wz
  row["base.vx"] = float(lin[0].item())
  row["base.vy"] = float(lin[1].item())
  row["base.wz"] = float(ang[2].item())


def _add_joint_state(row: dict[str, float], robot, joint_ids: dict[str, int]) -> None:
  data = robot.data
  for name, idx in joint_ids.items():
    pos = float(data.joint_pos[0, idx].item())
    row[f"{name}.pos"] = pos
    row[f"{name}.pos_deg"] = math.degrees(pos)
    row[f"{name}.vel"] = float(data.joint_vel[0, idx].item())


def main(cfg: Config) -> None:
  configure_torch_backends()
  checkpoint = Path(cfg.checkpoint).expanduser().resolve()
  if not checkpoint.exists():
    raise FileNotFoundError(checkpoint)

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(cfg.task, play=True)
  env_cfg.scene.num_envs = 1
  if cfg.no_terminations:
    env_cfg.terminations = {}

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  _set_constant_twist(raw_env, cfg.vx, cfg.vy, cfg.wz)

  agent_cfg = load_rl_cfg(cfg.task)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  policy = _make_policy(cfg.task, checkpoint, env, device)

  robot = env.unwrapped.scene["robot"]
  pitch_names = ("ankle_pitch_l_joint", "ankle_pitch_r_joint")
  roll_names = ("ankle_roll_l_joint", "ankle_roll_r_joint")
  pitch_act_ids = _index_by_name(robot.actuator_names, pitch_names)
  pitch_joint_ids = _index_by_name(robot.joint_names, pitch_names)
  roll_joint_ids = _index_by_name(robot.joint_names, roll_names)

  output_dir = (
    Path(cfg.output_dir).expanduser().resolve()
    if cfg.output_dir is not None
    else checkpoint.parent / "ankle_torque_plots"
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  obs = env.get_observations()
  rows: list[dict[str, float]] = []
  with torch.inference_mode():
    for step in range(cfg.warmup_steps + cfg.steps):
      actions = policy(obs)
      obs, _, _, _ = env.step(actions)
      if step < cfg.warmup_steps:
        continue

      data = robot.data
      row: dict[str, float] = {
        "step": float(step - cfg.warmup_steps),
        "time_s": float((step - cfg.warmup_steps) * env.unwrapped.step_dt),
      }
      _add_base_velocity(row, robot, cfg.vx, cfg.vy, cfg.wz)
      _add_joint_state(row, robot, pitch_joint_ids)
      _add_joint_state(row, robot, roll_joint_ids)
      for name, idx in pitch_act_ids.items():
        row[f"{name}.actuator_force"] = float(data.actuator_force[0, idx].item())
      for name, idx in pitch_joint_ids.items():
        row[f"{name}.target_pos"] = float(data.joint_pos_target[0, idx].item())
        row[f"{name}.qfrc_actuator"] = float(data.qfrc_actuator[0, idx].item())
      for name, idx in roll_joint_ids.items():
        passive_torque = float(
          data.data.qfrc_passive[0, data.indexing.joint_v_adr[idx]].item()
        )
        row[f"{name}.passive_torque_nm"] = passive_torque
        row[f"{name}.qfrc_passive"] = passive_torque
        row[f"{name}.qfrc_total"] = float(
          data.data.qfrc_smooth[0, data.indexing.joint_v_adr[idx]].item()
        )
      rows.append(row)

  csv_path = output_dir / f"{checkpoint.stem}_ankle_torques.csv"
  png_path = output_dir / f"{checkpoint.stem}_ankle_torques.png"
  _write_csv(csv_path, rows)
  plot_columns = [c for c in rows[0] if c not in {"step", "time_s"}]
  _plot(png_path, rows, plot_columns)

  print(f"[INFO] checkpoint: {checkpoint}")
  print(f"[INFO] command: vx={cfg.vx:.3f}, vy={cfg.vy:.3f}, wz={cfg.wz:.3f}")
  print(f"[INFO] CSV: {csv_path}")
  print(f"[INFO] PNG: {png_path}")
  print(
    "[INFO] V10 XML has ankle pitch and passive ankle roll joints; no ankle yaw joint."
  )
  print(f"[INFO] plotted columns: {', '.join(plot_columns)}")


if __name__ == "__main__":
  main(tyro.cli(Config))
