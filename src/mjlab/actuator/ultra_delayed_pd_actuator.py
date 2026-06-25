"""Ultra-equivalent delayed PD actuator with asymmetric torque-speed envelope.

This reproduces the behavior used in ultra_run_lab's CustomDelayedPDActuator:
- delayed command path (provided by base ActuatorCfg delay fields)
- PD torque computation
- asymmetric piecewise torque-speed clipping via X1/X2/Y1/Y2
- optional friction model Fs/Fd/Va
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass(kw_only=True)
class UltraDelayedPdActuatorCfg(IdealPdActuatorCfg):
  """Delayed PD actuator with Ultra's asymmetric torque-speed curve.

  Envelope definition (all in SI units):
  - Y1: peak torque when torque and speed have the same sign
  - Y2: peak torque when torque and speed have opposite signs
  - X1: max speed at full torque (knee)
  - X2: no-load speed

  For |v| < X1, limit is Y1 or Y2 depending on direction.
  For |v| >= X1, limit linearly decreases to zero at X2.
  """

  x1: float
  x2: float
  y1: float
  y2: float | None = None
  fs: float = 0.0
  fd: float = 0.0
  va: float = 0.01

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.x1 < 0.0 or self.x2 <= 0.0:
      raise ValueError("x1 must be >= 0 and x2 must be > 0.")
    if self.x2 <= self.x1:
      raise ValueError("x2 must be strictly greater than x1.")
    if self.y1 <= 0.0:
      raise ValueError("y1 must be > 0.")
    if self.y2 is not None and self.y2 <= 0.0:
      raise ValueError("y2 must be > 0 when provided.")
    if self.fs < 0.0 or self.fd < 0.0:
      raise ValueError("fs/fd must be non-negative.")
    if self.va <= 0.0:
      raise ValueError("va must be > 0.")

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> UltraDelayedPdActuator:
    return UltraDelayedPdActuator(self, entity, target_ids, target_names)


class UltraDelayedPdActuator(IdealPdActuator[UltraDelayedPdActuatorCfg]):
  """Ultra-equivalent delayed PD actuator model."""

  def __init__(
    self,
    cfg: UltraDelayedPdActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self._joint_vel: torch.Tensor | None = None
    self._effort_y1: torch.Tensor | None = None
    self._effort_y2: torch.Tensor | None = None
    self._velocity_x1: torch.Tensor | None = None
    self._velocity_x2: torch.Tensor | None = None
    self._friction_static: torch.Tensor | None = None
    self._friction_dynamic: torch.Tensor | None = None
    self._activation_vel: torch.Tensor | None = None
    # Per-env, per-joint multiplicative output-torque scale (motor-strength DR).
    # Defaults to 1.0 (nominal motors); randomized via the `dr.motor_strength`
    # event. Persists across episode resets (startup-style randomization).
    self.motor_strength: torch.Tensor | None = None

  def initialize(self, mj_model, model, data, device: str) -> None:
    super().initialize(mj_model, model, data, device)
    num_envs = data.nworld
    num_joints = len(self.target_names)

    y2 = self.cfg.y2 if self.cfg.y2 is not None else self.cfg.y1

    self._joint_vel = torch.zeros(
      (num_envs, num_joints), dtype=torch.float, device=device
    )
    self._effort_y1 = torch.full(
      (num_envs, num_joints), self.cfg.y1, dtype=torch.float, device=device
    )
    self._effort_y2 = torch.full(
      (num_envs, num_joints), y2, dtype=torch.float, device=device
    )
    self._velocity_x1 = torch.full(
      (num_envs, num_joints), self.cfg.x1, dtype=torch.float, device=device
    )
    self._velocity_x2 = torch.full(
      (num_envs, num_joints), self.cfg.x2, dtype=torch.float, device=device
    )
    self._friction_static = torch.full(
      (num_envs, num_joints), self.cfg.fs, dtype=torch.float, device=device
    )
    self._friction_dynamic = torch.full(
      (num_envs, num_joints), self.cfg.fd, dtype=torch.float, device=device
    )
    self._activation_vel = torch.full(
      (num_envs, num_joints), self.cfg.va, dtype=torch.float, device=device
    )
    self.motor_strength = torch.ones(
      (num_envs, num_joints), dtype=torch.float, device=device
    )

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self._joint_vel is not None
    assert self._friction_static is not None
    assert self._friction_dynamic is not None
    assert self._activation_vel is not None

    self._joint_vel[:] = cmd.vel
    effort = super().compute(cmd)

    # Match Ultra friction model: subtract static + dynamic friction after clipping.
    effort -= self._friction_static * torch.tanh(cmd.vel / self._activation_vel)
    effort -= self._friction_dynamic * cmd.vel

    # Motor-strength domain randomization: applied last (after clip + friction),
    # matching Ultra's `applied_effort *= motor_strength`.
    if self.motor_strength is not None:
      effort = effort * self.motor_strength

    return effort

  def set_motor_strength(
    self, env_ids: torch.Tensor | slice, strength: torch.Tensor
  ) -> None:
    """Set the per-joint output-torque scale for the given environments.

    Args:
      env_ids: Environment indices to update.
      strength: Scales of shape (num_envs, num_joints) or (num_envs,).
    """
    assert self.motor_strength is not None
    if strength.ndim == 1:
      strength = strength.unsqueeze(-1)
    self.motor_strength[env_ids] = strength.to(self.motor_strength.dtype)

  def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
    assert self._joint_vel is not None
    assert self._effort_y1 is not None
    assert self._effort_y2 is not None
    assert self._velocity_x1 is not None
    assert self._velocity_x2 is not None

    same_direction = (self._joint_vel * effort) > 0
    max_effort = torch.where(same_direction, self._effort_y1, self._effort_y2)

    # Above knee speed, linearly decay torque limit to 0 at x2.
    k = -max_effort / (self._velocity_x2 - self._velocity_x1)
    speed_decay_limit = k * (self._joint_vel.abs() - self._velocity_x1) + max_effort
    speed_decay_limit = speed_decay_limit.clamp_min(0.0)
    max_effort = torch.where(
      self._joint_vel.abs() < self._velocity_x1, max_effort, speed_decay_limit
    )

    # Keep inherited continuous effort limit active too (Ultra sets it to y2).
    if self.force_limit is not None:
      max_effort = torch.minimum(max_effort, self.force_limit)

    return torch.clamp(effort, min=-max_effort, max=max_effort)
