"""Convert Ultra recovery ``.npz`` clips into the 38-column AMP JSON format.

The recovery clips in ``ultra_yaw_0526`` already contain retargeted V9 joint
states plus world body poses. V9's AMP loader consumes a root-relative 38-column
feature vector:

  [0:13]  joint pos in AMP joint order
  [13:26] joint vel in AMP joint order
  [26:38] [lhand, rhand, lfoot, rfoot] positions in the root frame

Usage:
  uv run python scripts/convert_ultra_recovery_npz_to_amp_txt.py \
      --src src/mjlab/asset_zoo/robots/ultra_game_yaw/recovery_motions \
      --dst src/mjlab/asset_zoo/robots/ultra_game_yaw/amp_motions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RECOVERY_JOINT_ORDER: tuple[str, ...] = (
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

# Body order used by the recovery NPZ exporter. It is not the XML tree order.
RECOVERY_BODY_ORDER: tuple[str, ...] = (
  "base_link",
  "hip_yaw_l_link",
  "hip_yaw_r_link",
  "waist_yaw_link",
  "hip_roll_l_link",
  "hip_roll_r_link",
  "shoulder_pitch_l_link",
  "shoulder_pitch_r_link",
  "hip_pitch_l_link",
  "hip_pitch_r_link",
  "knee_pitch_l_link",
  "knee_pitch_r_link",
  "ankle_pitch_l_link",
  "ankle_pitch_r_link",
)

EE_BODIES: tuple[str, ...] = (
  "shoulder_pitch_l_link",
  "shoulder_pitch_r_link",
  "ankle_pitch_l_link",
  "ankle_pitch_r_link",
)
HAND_LOCAL_OFFSET = np.array([0.0, 0.0, -0.4], dtype=np.float64)


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Apply WXYZ quaternion(s) to vec3(s), broadcasting over leading dims."""
  w = q[..., 0:1]
  xyz = q[..., 1:4]
  t = 2.0 * np.cross(xyz, v)
  return v + w * t + np.cross(xyz, t)


def _quat_conj(q: np.ndarray) -> np.ndarray:
  out = q.copy()
  out[..., 1:4] *= -1.0
  return out


def _write_json_motion(
  dst: Path,
  frames: np.ndarray,
  frame_duration: float,
  loop: str,
  weight: float,
) -> None:
  dst.parent.mkdir(parents=True, exist_ok=True)
  frame_rows = [[round(float(x), 6) for x in row] for row in frames]
  payload: dict[str, object] = {
    "LoopMode": loop,
    "FrameDuration": round(frame_duration, 6),
    "EnableCycleOffsetPosition": True,
    "EnableCycleOffsetRotation": True,
    "MotionWeight": weight,
    "Frames": frame_rows,
  }
  with open(dst, "w") as f:
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
    for i, row in enumerate(frame_rows):
      sep = "," if i < len(frame_rows) - 1 else ""
      f.write("  " + json.dumps(row) + sep + "\n")
    f.write("]\n}\n")


def convert_file(src: Path, dst: Path, loop: str, weight: float) -> None:
  data = np.load(src)
  fps = float(np.asarray(data["fps"]).reshape(-1)[0])
  joint_pos_raw = np.asarray(data["joint_pos"], dtype=np.float64)
  joint_vel_raw = np.asarray(data["joint_vel"], dtype=np.float64)
  body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)
  body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float64)

  assert joint_pos_raw.shape[1] == len(RECOVERY_JOINT_ORDER), joint_pos_raw.shape
  assert body_pos_w.shape[1] == len(RECOVERY_BODY_ORDER), body_pos_w.shape

  recovery_joint_idx = {name: i for i, name in enumerate(RECOVERY_JOINT_ORDER)}
  amp_perm = [recovery_joint_idx[name] for name in AMP_JOINT_ORDER]
  joint_pos = joint_pos_raw[:, amp_perm]
  joint_vel = joint_vel_raw[:, amp_perm]

  body_idx = {name: i for i, name in enumerate(RECOVERY_BODY_ORDER)}
  root_pos = body_pos_w[:, body_idx["base_link"], :]
  root_quat_inv = _quat_conj(body_quat_w[:, body_idx["base_link"], :])

  sh_l = body_idx["shoulder_pitch_l_link"]
  sh_r = body_idx["shoulder_pitch_r_link"]
  an_l = body_idx["ankle_pitch_l_link"]
  an_r = body_idx["ankle_pitch_r_link"]

  hand_offset = np.broadcast_to(HAND_LOCAL_OFFSET, body_pos_w[:, sh_l, :].shape)
  lhand = body_pos_w[:, sh_l, :] + _quat_apply(body_quat_w[:, sh_l, :], hand_offset)
  rhand = body_pos_w[:, sh_r, :] + _quat_apply(body_quat_w[:, sh_r, :], hand_offset)
  lfoot = body_pos_w[:, an_l, :]
  rfoot = body_pos_w[:, an_r, :]

  rel_w = np.stack([lhand, rhand, lfoot, rfoot], axis=1) - root_pos[:, None, :]
  rel_b = _quat_apply(root_quat_inv[:, None, :], rel_w)
  frames = np.concatenate(
    [joint_pos, joint_vel, rel_b.reshape(len(joint_pos), -1)], axis=1
  )
  assert frames.shape[1] == 38, frames.shape

  _write_json_motion(dst, frames, 1.0 / fps, loop, weight)
  print(
    f"Wrote {len(frames)} frames ({len(frames) / fps:.2f}s @ {fps:.1f}fps) -> {dst}"
  )


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--src", required=True, type=Path, help=".npz file or directory")
  ap.add_argument("--dst", required=True, type=Path, help="output file or directory")
  ap.add_argument("--loop", default="Wrap", choices=["Clamp", "Wrap"])
  ap.add_argument("--weight", default=1.0, type=float)
  args = ap.parse_args()

  if args.src.is_dir():
    args.dst.mkdir(parents=True, exist_ok=True)
    for src in sorted(args.src.glob("*.npz")):
      stem = src.stem.replace("_50fps", "")
      convert_file(src, args.dst / f"recovery_{stem}.txt", args.loop, args.weight)
  else:
    convert_file(args.src, args.dst, args.loop, args.weight)


if __name__ == "__main__":
  main()
