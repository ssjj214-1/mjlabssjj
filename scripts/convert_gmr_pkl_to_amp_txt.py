"""Convert a GMR retargeted motion ``.pkl`` into the 38-column AMP ``.txt`` that
mjlab's :class:`AMPLoader` consumes (the JSON ``Frames`` format used by
``asset_zoo/robots/ultra_game_yaw/amp_motions/*.txt``).

GMR ``.pkl`` layout (Ultra GameYaw, 13 DoF):
  fps:float, root_pos(T,3), root_rot(T,4)[xyzw], dof_pos(T,13), ...

The AMP frame is 38 columns and is **root-relative** (it never stores the world
root pose), so world heading / origin / height of the clip are irrelevant:
  [0:13]  joint pos   — in ``AMP_JOINT_ORDER``
  [13:26] joint vel   — finite-difference of joint pos (rad/s), same order
  [26:38] end-effector positions in the base/root frame —
          [lhand, rhand, lfoot, rfoot] (hands are shoulder + local z offset)

The joint order, end-effector bodies and hand offset MUST stay in sync with
``mjlab.tasks.velocity.config.ultra_game_yaw.amp_him`` (the live AMP feature
extractor ``get_amp_obs_for_expert_trans``). They are duplicated here so this
converter has no import-time dependency on the task package.

Usage:
  uv run python scripts/convert_gmr_pkl_to_amp_txt.py \
      --src /home/ps/GMR/output_ultra_game_yaw_stand_to_run.pkl \
      --dst src/mjlab/asset_zoo/robots/ultra_game_yaw/amp_motions/accel_stand_to_run.txt \
      --loop Clamp --weight 1.0
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.ultra_game_yaw.ultra_constants import ULTRA_XML

# GMR dof_pos column order == MuJoCo joint tree order of the Ultra model
# (leg-left, leg-right, waist, shoulders).
GMR_DOF_ORDER: tuple[str, ...] = (
  "hip_yaw_l_joint",
  "hip_roll_l_joint",
  "hip_pitch_l_joint",
  "knee_pitch_l_joint",
  "ankle_pitch_l_joint",
  "hip_yaw_r_joint",
  "hip_roll_r_joint",
  "hip_pitch_r_joint",
  "knee_pitch_r_joint",
  "ankle_pitch_r_joint",
  "waist_yaw_joint",
  "shoulder_pitch_l_joint",
  "shoulder_pitch_r_joint",
)

# AMP feature joint order (matches amp_him._AMP_JOINT_NAMES).
AMP_JOINT_ORDER: tuple[str, ...] = (
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

# End-effector proxies (matches amp_him._AMP_EE_BODIES + _HAND_LOCAL_OFFSET).
EE_BODIES: tuple[str, ...] = (
  "shoulder_pitch_l_link",  # left-hand proxy (shoulder + offset)
  "shoulder_pitch_r_link",  # right-hand proxy
  "ankle_pitch_l_link",  # left foot
  "ankle_pitch_r_link",  # right foot
)
HAND_LOCAL_OFFSET = np.array([0.0, 0.0, -0.4], dtype=np.float64)
ROOT_BODY = "base_link"


def _quat_conj(q: np.ndarray) -> np.ndarray:
  out = q.copy()
  out[1:] *= -1.0
  return out


def _rot(vec: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
  res = np.zeros(3)
  mujoco.mju_rotVecQuat(res, np.ascontiguousarray(vec), np.ascontiguousarray(quat_wxyz))
  return res


def convert(
  src: Path,
  dst: Path,
  loop: str,
  weight: float,
  start_frame: int = 0,
  end_frame: int = -1,
) -> None:
  raw = pickle.load(open(src, "rb"))
  fps = float(raw["fps"])
  dof_pos = np.asarray(raw["dof_pos"], dtype=np.float64)  # (T, 13) in GMR order
  assert dof_pos.shape[1] == len(GMR_DOF_ORDER), (
    f"expected 13 dofs, got {dof_pos.shape[1]}"
  )

  # Optional trim. The AMP positives are sampled by *episode time* (frame 0 ==
  # episode reset), so any leading standstill in the clip teaches the launch
  # window that "stand still is correct" while velocity tracking says "run" —
  # use --start-frame to drop the dead lead-in so frame 0 is the motion onset.
  total = dof_pos.shape[0]
  end = total if end_frame < 0 else min(end_frame, total)
  dof_pos = dof_pos[start_frame:end]
  num_frames = dof_pos.shape[0]
  assert num_frames > 1, f"empty clip after trim [{start_frame}:{end}] of {total}"

  # Reindex joints: GMR order -> AMP order.
  gmr_idx = {n: i for i, n in enumerate(GMR_DOF_ORDER)}
  amp_perm = [gmr_idx[n] for n in AMP_JOINT_ORDER]
  joint_pos = dof_pos[:, amp_perm]  # (T, 13) AMP order

  # Joint velocity via finite difference (rad/s), last frame held.
  joint_vel = np.zeros_like(joint_pos)
  joint_vel[:-1] = (joint_pos[1:] - joint_pos[:-1]) * fps
  joint_vel[-1] = joint_vel[-2] if num_frames > 1 else 0.0

  # End-effector positions in the base/root frame via FK (root-invariant, so we
  # spawn the model at the origin with identity orientation).
  model = mujoco.MjModel.from_xml_path(str(ULTRA_XML))
  data = mujoco.MjData(model)
  jnt_qadr = {
    n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    for n in GMR_DOF_ORDER
  }
  bid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in EE_BODIES}
  root_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)

  ee = np.zeros((num_frames, 12), dtype=np.float64)
  for t in range(num_frames):
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = (0.0, 0.0, 1.0)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for n in GMR_DOF_ORDER:
      data.qpos[jnt_qadr[n]] = dof_pos[t, gmr_idx[n]]
    mujoco.mj_forward(model, data)

    root_pos = data.xpos[root_bid].copy()
    root_quat_inv = _quat_conj(data.xquat[root_bid].copy())

    sh_l, sh_r, an_l, an_r = (data.xpos[bid[n]].copy() for n in EE_BODIES)
    lhand = sh_l + _rot(HAND_LOCAL_OFFSET, data.xquat[bid[EE_BODIES[0]]].copy())
    rhand = sh_r + _rot(HAND_LOCAL_OFFSET, data.xquat[bid[EE_BODIES[1]]].copy())
    pts = [lhand, rhand, an_l, an_r]
    for k, p in enumerate(pts):
      ee[t, 3 * k : 3 * k + 3] = _rot(p - root_pos, root_quat_inv)

  frames = np.concatenate([joint_pos, joint_vel, ee], axis=1)  # (T, 38)
  assert frames.shape[1] == 38, frames.shape

  payload = {
    "LoopMode": loop,
    "FrameDuration": round(1.0 / fps, 6),
    "EnableCycleOffsetPosition": True,
    "EnableCycleOffsetRotation": True,
    "MotionWeight": weight,
    "Frames": [[round(float(x), 6) for x in row] for row in frames],
  }
  dst.parent.mkdir(parents=True, exist_ok=True)
  with open(dst, "w") as f:
    # Match the existing motion-file style: one frame per line.
    f.write("{\n")
    for key in (
      "LoopMode",
      "FrameDuration",
      "EnableCycleOffsetPosition",
      "EnableCycleOffsetRotation",
      "MotionWeight",
    ):
      f.write(f'"{key}": {json.dumps(payload[key])},\n')
    f.write('\n"Frames":\n[\n')
    for i, row in enumerate(payload["Frames"]):
      sep = "," if i < len(payload["Frames"]) - 1 else ""
      f.write("  " + json.dumps(row) + sep + "\n")
    f.write("]\n}\n")

  print(
    f"Wrote {num_frames} frames ({num_frames / fps:.2f}s @ {fps:.2f}fps, "
    f"loop={loop}) -> {dst}"
  )


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--src", required=True, type=Path)
  ap.add_argument("--dst", required=True, type=Path)
  ap.add_argument("--loop", default="Clamp", choices=["Clamp", "Wrap"])
  ap.add_argument("--weight", default=1.0, type=float)
  ap.add_argument(
    "--start-frame",
    default=0,
    type=int,
    help="drop frames before this index (e.g. trim a standstill lead-in)",
  )
  ap.add_argument(
    "--end-frame",
    default=-1,
    type=int,
    help="last frame index (exclusive); -1 keeps through the end",
  )
  args = ap.parse_args()
  convert(args.src, args.dst, args.loop, args.weight, args.start_frame, args.end_frame)


if __name__ == "__main__":
  main()
