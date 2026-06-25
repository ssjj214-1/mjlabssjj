"""Ultra GameYaw (13 DoF biped) constants.

Migrated from `ultra_run_lab/legged_lab/assets/ultra_GameYaw/ultra_game_yaw.py`.
PD gains, effort/velocity limits and armature follow the Isaac Lab training cfg.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import UltraDelayedPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

ULTRA_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "ultra_game_yaw"
  / "xmls"
  / "ultra_game_yaw.xml"
)
assert ULTRA_XML.exists(), f"Missing MJCF: {ULTRA_XML}"

ULTRA_V10_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "ultra_game_yaw"
  / "xmls"
  / "ultra_game_yaw_v10.xml"
)
assert ULTRA_V10_XML.exists(), f"Missing MJCF: {ULTRA_V10_XML}"


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(ULTRA_XML))


def get_v10_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(ULTRA_V10_XML))


##
# Actuator configs.
#
# Effort / velocity / kp / kv come from the original Isaac Lab cfg
# (CustomDelayedPDActuatorCfg). Armature is already set per-joint in the MJCF
# from the URDF, so we leave armature=None here to avoid overriding it.
##

# Hip yaw (left & right): kp=500, kv=10, effort=250, X1/X2/Y1/Y2 from Ultra.
ULTRA_ACT_HIP_YAW = UltraDelayedPdActuatorCfg(
  target_names_expr=("hip_yaw_l_joint", "hip_yaw_r_joint"),
  stiffness=500.0,
  damping=10.0,
  effort_limit=250.0,
  x1=10.0,
  x2=20.9,
  y1=150.0,
  y2=250.0,
  delay_min_lag=0,
  delay_max_lag=4,
)
# Hip roll (left & right): kp=700, kv=20, effort=657.
ULTRA_ACT_HIP_ROLL = UltraDelayedPdActuatorCfg(
  target_names_expr=("hip_roll_l_joint", "hip_roll_r_joint"),
  stiffness=700.0,
  damping=20.0,
  effort_limit=657.0,
  x1=12.0,
  x2=18.8,
  y1=600.0,
  y2=657.0,
  delay_min_lag=0,
  delay_max_lag=4,
)
# Hip pitch + knee pitch (left & right): kp=700, kv=10, effort=657.
ULTRA_ACT_HIP_KNEE_PITCH = UltraDelayedPdActuatorCfg(
  target_names_expr=(
    "hip_pitch_l_joint",
    "hip_pitch_r_joint",
    "knee_pitch_l_joint",
    "knee_pitch_r_joint",
  ),
  stiffness=700.0,
  damping=10.0,
  effort_limit=657.0,
  x1=17.0,
  x2=26.1,
  y1=600.0,
  y2=657.0,
  delay_min_lag=0,
  delay_max_lag=4,
)
# Ankle pitch (left & right): kp=100, kv=6, effort=400.
ULTRA_ACT_ANKLE = UltraDelayedPdActuatorCfg(
  target_names_expr=("ankle_pitch_l_joint", "ankle_pitch_r_joint"),
  stiffness=100.0,
  damping=6.0,
  effort_limit=400.0,
  x1=17.0,
  x2=26.1,
  y1=350.0,
  y2=400.0,
  delay_min_lag=0,
  delay_max_lag=4,
)
# Waist yaw: kp=400, kv=8, effort=250.
ULTRA_ACT_WAIST = UltraDelayedPdActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=400.0,
  damping=8.0,
  effort_limit=250.0,
  x1=10.0,
  x2=20.9,
  y1=150.0,
  y2=250.0,
  delay_min_lag=0,
  delay_max_lag=4,
)
# Shoulder pitch (left & right): kp=50, kv=3, effort=140.
ULTRA_ACT_SHOULDER = UltraDelayedPdActuatorCfg(
  target_names_expr=("shoulder_pitch_l_joint", "shoulder_pitch_r_joint"),
  stiffness=50.0,
  damping=3.0,
  effort_limit=140.0,
  x1=13.0,
  x2=26.1,
  y1=100.0,
  y2=140.0,
  delay_min_lag=0,
  delay_max_lag=4,
)


##
# Initial state (matches Isaac cfg `init_state`).
##

import math as _math  # noqa: E402

ULTRA_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.20),  # slight clearance above ground; freejoint qpos[2]
  joint_pos={
    # Legs (radians).
    "hip_yaw_l_joint": 0.0,
    "hip_yaw_r_joint": 0.0,
    "hip_roll_l_joint": 0.0,
    "hip_roll_r_joint": 0.0,
    "hip_pitch_l_joint": _math.radians(-15.0),
    "hip_pitch_r_joint": _math.radians(-15.0),
    "knee_pitch_l_joint": _math.radians(30.0),
    "knee_pitch_r_joint": _math.radians(30.0),
    "ankle_pitch_l_joint": _math.radians(-15.0),
    "ankle_pitch_r_joint": _math.radians(-15.0),
    # Waist + arms.
    "waist_yaw_joint": 0.0,
    "shoulder_pitch_l_joint": 0.5,
    "shoulder_pitch_r_joint": 0.5,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
#
# Each collision geom is named `*_collision` (or `left/right_foot_collision`).
# Foot geoms get condim=3 with elevated priority for stable ground contact.
##

ULTRA_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot_collision$": 1},
  friction={r"^(left|right)_foot_collision$": (0.6,)},
)


##
# Final config.
##

ULTRA_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    ULTRA_ACT_HIP_YAW,
    ULTRA_ACT_HIP_ROLL,
    ULTRA_ACT_HIP_KNEE_PITCH,
    ULTRA_ACT_ANKLE,
    ULTRA_ACT_WAIST,
    ULTRA_ACT_SHOULDER,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_ultra_robot_cfg() -> EntityCfg:
  """Return a fresh Ultra GameYaw EntityCfg."""
  return EntityCfg(
    init_state=ULTRA_HOME_KEYFRAME,
    collisions=(ULTRA_COLLISION,),
    spec_fn=get_spec,
    articulation=ULTRA_ARTICULATION,
  )


def get_ultra_v10_robot_cfg() -> EntityCfg:
  """Return a fresh Ultra GameYaw V10 EntityCfg (15 DoF, passive ankle-roll)."""
  return EntityCfg(
    init_state=ULTRA_HOME_KEYFRAME,
    collisions=(ULTRA_COLLISION,),
    spec_fn=get_v10_spec,
    articulation=ULTRA_ARTICULATION,
  )


# Per-joint action scale (action_scale * 0.25 like G1 convention; this scales the
# policy output before adding to default joint pos).
# 0.25 here matches the original Isaac cfg `action_scale=0.25`.
ULTRA_ACTION_SCALE: dict[str, float] = {".*": 0.25}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_ultra_robot_cfg())
  viewer.launch(robot.spec.compile())
