"""Ultra GameYaw v10: V9 + passive ankle-roll joints in the XML.

The robot XML now has two additional passive ankle-roll DOFs
(ankle_roll_l_joint / ankle_roll_r_joint, stiffness=20 Nm/rad, damping=0.5).
They are not actuated and must NOT enter the 48-dim actor obs.

Changes vs v9:
1. ``joint_pos`` / ``joint_vel`` actor **and** critic terms are restricted to the
   13 actuated joints by name so the obs vector stays identical to v9 despite
   the extra DOFs in the model.
2. ``encoder_bias`` DR is likewise restricted to the 13 actuated joints so the
   passive joints are not given a random sensor offset at startup.
3. Two new critic-only terms (``ankle_roll_pos``, ``ankle_roll_vel``) expose the
   passive-joint state to the value network; the policy never sees them.
4. Foot contact is measured on the ankle-roll bodies, where the V10 foot
   collision geoms actually live.

Everything else (rewards, curriculum, command sampling, PD, smoothness loss,
DR, HIM history=10) is identical to v9.
"""

from __future__ import annotations

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch
from mjlab.tasks.velocity.config.ultra_game_yaw.amp_him import RslRlAmpHimRunnerCfg
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs_v6 import (
  get_ultra_v10_robot_cfg_v6,
)
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs_v9 import (
  ultra_game_yaw_amp_him_v9_runner_cfg,
  ultra_game_yaw_v9_env_cfg,
)

# Joint-name patterns for the 13 actuated DOFs (excludes ankle_roll_*).
_ACTUATED_JOINTS = [
  "shoulder_pitch.*",
  "hip.*",
  "knee_pitch.*",
  "ankle_pitch.*",
  "waist_yaw.*",
]
_ACT_CFG = SceneEntityCfg("robot", joint_names=_ACTUATED_JOINTS)
_ROLL_CFG = SceneEntityCfg("robot", joint_names=["ankle_roll.*"])


def ultra_game_yaw_v10_env_cfg(play: bool = False):
  """V9 env with passive ankle-roll joints; actor obs stays at 48 dims."""
  cfg = ultra_game_yaw_v9_env_cfg(play=play)

  # ── Switch to V10 robot (15 DoF XML with passive ankle-roll) ─────────
  cfg.scene.entities["robot"] = get_ultra_v10_robot_cfg_v6()

  # ── Actor obs: restrict joint_pos / joint_vel to 13 actuated joints ──
  for term in ("joint_pos", "joint_vel"):
    cfg.observations["actor"].terms[term].params["asset_cfg"] = _ACT_CFG

  # ── Critic obs: same restriction + add passive-joint state ───────────
  for term in ("joint_pos", "joint_vel"):
    cfg.observations["critic"].terms[term].params["asset_cfg"] = _ACT_CFG

  cfg.observations["critic"].terms["ankle_roll_pos"] = ObservationTermCfg(
    func=envs_mdp.joint_pos_rel,
    params={"asset_cfg": _ROLL_CFG},
  )
  cfg.observations["critic"].terms["ankle_roll_vel"] = ObservationTermCfg(
    func=envs_mdp.joint_vel_rel,
    params={"asset_cfg": _ROLL_CFG},
    scale=0.05,
  )

  # ── encoder_bias DR: restrict to actuated joints only ────────────────
  if "encoder_bias" in cfg.events:
    cfg.events["encoder_bias"].params["asset_cfg"] = _ACT_CFG

  # V10's physical foot collision geoms are on ankle_roll_*_link. V9's inherited
  # contact sensor watches ankle_pitch_*_link, which is now the parent link and
  # makes foot-contact rewards/critic terms inconsistent with the actual foot.
  for sensor in cfg.scene.sensors or ():
    if getattr(sensor, "name", None) == "feet_ground_contact":
      sensor.primary = ContactMatch(
        mode="body",
        pattern=r"^(ankle_roll_l_link|ankle_roll_r_link)$",
        entity="robot",
      )
      break

  return cfg


def ultra_game_yaw_amp_him_v10_runner_cfg() -> RslRlAmpHimRunnerCfg:
  """V10 runner: identical to v9 (history=10, HoST loss ON)."""
  cfg = ultra_game_yaw_amp_him_v9_runner_cfg()
  cfg.experiment_name = "ultra_game_yaw_amp_him_v10"
  cfg.wandb_project = "ultra_game_yaw_amp_him_v10"
  return cfg
