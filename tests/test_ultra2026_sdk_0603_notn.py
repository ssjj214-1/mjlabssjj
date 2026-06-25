from pathlib import Path

import yaml

SDK_ROOT = Path("/home/ps/ultra2026_rl_sdk")


def test_ultra2026_sdk_has_dedicated_0603_notn_sim2sim_entry():
  entry = SDK_ROOT / "deploy_ultra_sim_real" / "main_0603_notn.py"
  loco_cfg_path = (
    SDK_ROOT / "policy" / "loco_mode" / "config" / "LocoMode_0603_notn.yaml"
  )
  env_cfg_path = (
    SDK_ROOT / "deploy_ultra_sim_real" / "config" / "env-ultra-0603-notn.yaml"
  )
  model_path = (
    SDK_ROOT / "policy" / "loco_mode" / "model" / "ultra_game_yaw_policy_0603_notn.onnx"
  )

  assert entry.exists()
  assert loco_cfg_path.exists()
  assert env_cfg_path.exists()
  assert model_path.exists()

  entry_text = entry.read_text(encoding="utf-8")
  assert "LocoMode_0603_notn.yaml" in entry_text
  assert "env-ultra-0603-notn.yaml" in entry_text
  assert "ULTRA_SPAWN_Z" not in entry_text

  with loco_cfg_path.open("r", encoding="utf-8") as f:
    loco_cfg = yaml.safe_load(f)
  assert loco_cfg["model"]["path"] == "../model/ultra_game_yaw_policy_0603_notn.onnx"
  assert loco_cfg["observation"]["obs_scale_ang_vel"] == 1.0
  assert loco_cfg["observation"]["obs_scale_gravity"] == 1.0
  assert loco_cfg["observation"]["obs_scale_dof_pos"] == 1.0
  assert loco_cfg["observation"]["obs_scale_dof_vel"] == 1.0
  assert loco_cfg["observation"]["obs_scale_actions"] == 1.0
  assert loco_cfg["observation"]["command_obs_scale"] == [1.0, 1.0, 1.0]
  assert loco_cfg["max_cmd"] == [15.0, 0.5, 0.5]
  assert loco_cfg["max_cmd_vx_neg"] == 0.5
  assert loco_cfg["heading_kp"] == 0.5
  assert loco_cfg["max_lin_accel"] == 5.0
  assert loco_cfg["dof"]["kp"] == [
    500.0,
    700.0,
    700.0,
    700.0,
    100.0,
    500.0,
    700.0,
    700.0,
    700.0,
    100.0,
    400.0,
    50.0,
    50.0,
  ]
  assert loco_cfg["dof"]["kd"] == [
    10.0,
    20.0,
    10.0,
    10.0,
    6.0,
    10.0,
    20.0,
    10.0,
    10.0,
    6.0,
    8.0,
    3.0,
    3.0,
  ]
  assert loco_cfg["dof"]["default_pos"] == [
    0.0,
    0.0,
    -0.2617993877991494,
    0.5235987755982988,
    -0.2617993877991494,
    0.0,
    0.0,
    -0.2617993877991494,
    0.5235987755982988,
    -0.2617993877991494,
    0.0,
    0.5,
    0.5,
  ]

  with env_cfg_path.open("r", encoding="utf-8") as f:
    env_cfg = yaml.safe_load(f)
  assert env_cfg["sim_model_path"] == "assets/ultra_yaw_0523/mjcf/ultra_game_yaw.xml"
  assert [joint["zero_pos"] for joint in env_cfg["joints"][:13]] == [
    0.0,
    0.0,
    -0.2617993877991494,
    0.5235987755982988,
    -0.2617993877991494,
    0.0,
    0.0,
    -0.2617993877991494,
    0.5235987755982988,
    -0.2617993877991494,
    0.0,
    0.5,
    0.5,
  ]
  assert [joint["sim"]["kp"] for joint in env_cfg["joints"][:13]] == loco_cfg["dof"][
    "kp"
  ]
  assert [joint["sim"]["kd"] for joint in env_cfg["joints"][:13]] == loco_cfg["dof"][
    "kd"
  ]


def test_ultra2026_sdk_0603_logs_true_motion_and_sent_targets():
  main_text = (SDK_ROOT / "deploy_ultra_sim_real" / "main.py").read_text(
    encoding="utf-8"
  )

  assert 'bind_vector("joint_pos_target_sent"' in main_text
  assert 'bind_vector("base_pos"' in main_text
  assert 'bind_vector("base_lin_vel"' in main_text
