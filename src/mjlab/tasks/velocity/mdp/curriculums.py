from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict, total=False):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None
  # Optional: change the fraction of stand-still (zero-command) envs at this
  # stage. Lets a curriculum command more envs to stand still early on (when the
  # policy must first learn to hold a pose) and fewer later.
  rel_standing_envs: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
  min_speed_cap: float | None = None,
  move_up_distance_frac: float | None = None,
) -> dict[str, torch.Tensor]:
  """Distance-based terrain difficulty curriculum (IsaacLab parity).

  Robots that walk far enough are promoted to a harder row; those that cover
  less than half their commanded distance are demoted.

  ``move_up_distance_frac`` controls the *upward* gate. Left ``None`` (default)
  it keeps the legacy IsaacLab rule ``distance > size/2`` -- an absolute
  threshold (half a tile) tuned for ~1 m/s walkers, for which clearing 4 m in an
  episode is real progress. For a high-speed runner that threshold is covered
  trivially every episode (8 m/s over a 20 s episode = up to 160 m vs a 4 m
  gate), so terrain rockets to max difficulty and the downward servo then pins
  the robot at the difficulty where it tracks only ~50% of its command -- the
  observed 4-6 m/s plateau on terrain while flat ramps to ~14. Setting
  ``move_up_distance_frac`` (e.g. ``0.8``) switches the gate to a fraction of
  the *commanded* distance, so terrain only hardens when the robot actually
  keeps up at the commanded speed; speed and difficulty then co-advance instead
  of difficulty capping speed.

  ``min_speed_cap`` gates *upward* progression on the forward-speed curriculum:
  terrain difficulty only starts climbing once the ``command_name`` term's
  ``lin_vel_x`` cap reaches ``min_speed_cap`` (m/s). This breaks the
  curriculum-coupling trap where the distance-based terrain curriculum rockets
  difficulty to mid-level within a few thousand iterations (a 20 s episode
  trivially covers >half a tile even at low speed), pinning the robot on terrain
  too hard for its current speed and starving the speed curriculum -- which can
  only advance when forward-velocity tracking is good. With the gate, the speed
  curriculum gets a clean low-difficulty runway first, then terrain ramps while
  speed keeps climbing (speed leads, terrain follows). Downward moves stay
  ungated so a struggling robot always drops back. ``None`` (default) keeps the
  original ungated behaviour for tasks that don't pair the two curricula.
  """
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Distance the robot would cover by perfectly tracking its command this
  # episode (used by both the commanded-distance upward gate and the demotion
  # threshold below).
  commanded_distance = (
    torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s
  )

  # Robots that walked far enough progress to harder terrains. The absolute
  # half-tile rule is meaningless for a fast runner (always satisfied), so when
  # ``move_up_distance_frac`` is set, gate on a fraction of commanded distance.
  if move_up_distance_frac is not None:
    move_up = distance > commanded_distance * move_up_distance_frac
  else:
    move_up = distance > terrain_generator.size[0] / 2

  # Gate upward progression on the forward-speed curriculum cap. Until the robot
  # is commanded fast enough, freeze terrain difficulty so the speed curriculum
  # can ramp on the easy low-difficulty rows first.
  speed_cap = float("inf")
  if min_speed_cap is not None:
    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None
    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
    speed_cap = float(cfg.ranges.lin_vel_x[1])
    if speed_cap < min_speed_cap:
      move_up = move_up & False

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = distance < commanded_distance * 0.5
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  # Fraction of commanded distance actually covered, averaged over reset envs.
  # The downward servo demotes below 0.5 and (when gated) upward promotes above
  # ``move_up_distance_frac``; watching this tells you whether the robot is
  # keeping up with its command or being capped by terrain difficulty.
  dist_frac = torch.mean(distance / commanded_distance.clamp_min(1e-6))
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
    "dist_frac": dist_frac,
  }
  if min_speed_cap is not None:
    result["speed_gate_open"] = torch.tensor(float(speed_cap >= min_speed_cap))

  # In curriculum mode num_cols == num_terrains (one column per type),
  # so the column index directly maps to the sub-terrain name.
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage.get("step", 0):
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
      rel_standing = stage.get("rel_standing_envs")
      if rel_standing is not None:
        cfg.rel_standing_envs = rel_standing
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
    "rel_standing_envs": torch.tensor(cfg.rel_standing_envs),
  }
