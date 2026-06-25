"""Play AMP motion files for Ultra GameYaw in MuJoCo passive viewer.

Usage:
    uv run python play_amp_motions.py [stand|walk|run|run14|accel]

Press Ctrl+C to stop.
"""

import json
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

_SRC = Path(__file__).parent / "src/mjlab"
_XML = _SRC / "asset_zoo/robots/ultra_game_yaw/xmls/ultra_game_yaw.xml"
_AMP = _SRC / "asset_zoo/robots/ultra_game_yaw/amp_motions"

MOTIONS = {
  "stand": _AMP / "stand_ultra_yaw2.txt",
  "walk": _AMP / "walk_ultra_yaw2.txt",
  "run": _AMP / "run_17200_9mps.txt",
  "run14": _AMP / "run_ultrayaw_14mps.txt",
  "accel": _AMP / "accel_stand_to_run.txt",
}

# Frame[0:13] joint order (matches convert_gmr_pkl_to_amp_txt.AMP_JOINT_ORDER)
_AMP_JOINTS = (
  "shoulder_pitch_r_joint",
  "shoulder_pitch_l_joint",
  "hip_yaw_r_joint",
  "hip_roll_r_joint",
  "hip_pitch_r_joint",
  "knee_pitch_r_joint",
  "ankle_pitch_r_joint",
  "hip_yaw_l_joint",
  "hip_roll_l_joint",
  "hip_pitch_l_joint",
  "knee_pitch_l_joint",
  "ankle_pitch_l_joint",
  "waist_yaw_joint",
)


def main() -> None:
  name = sys.argv[1] if len(sys.argv) > 1 else "stand"
  if name not in MOTIONS:
    print(f"Unknown motion '{name}'. Available: {list(MOTIONS)}")
    sys.exit(1)

  model = mujoco.MjModel.from_xml_path(str(_XML))
  data = mujoco.MjData(model)

  # Pre-compute qpos address for each AMP joint
  amp_qadr = [
    model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
    for j in _AMP_JOINTS
  ]

  with open(MOTIONS[name]) as f:
    d = json.load(f)
  joint_pos = np.array(d["Frames"])[:, :13]  # only joint pos, skip vel+ee
  dt = float(d["FrameDuration"])
  n = len(joint_pos)
  print(f"[{name}] {n} frames @ {1 / dt:.0f} fps  ({n * dt:.1f}s total)")

  # Fixed root: motion files are root-relative, place robot at standing height
  data.qpos[:3] = (0.0, 0.0, 1.2)  # base_link z from ULTRA_HOME_KEYFRAME
  data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)  # identity quaternion (wxyz)

  with mujoco.viewer.launch_passive(model, data) as v:
    frame = 0
    t_next = time.perf_counter()
    while v.is_running():
      for i, qadr in enumerate(amp_qadr):
        data.qpos[qadr] = joint_pos[frame, i]
      mujoco.mj_forward(model, data)
      v.sync()
      frame = (frame + 1) % n
      t_next += dt
      slack = t_next - time.perf_counter()
      if slack > 0:
        time.sleep(slack)


if __name__ == "__main__":
  main()
