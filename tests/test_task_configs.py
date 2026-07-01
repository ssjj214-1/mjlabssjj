"""Generic tests for task config integrity."""

import math
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.velocity.config.ultra_game_yaw.amp_him import (
  _select_amp_foot_bodies,
)
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs_v9 import (
  ultra_game_yaw_amp_him_v9_runner_cfg,
  ultra_game_yaw_v9_env_cfg,
)
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs_v9plus import (
  ultra_game_yaw_amp_him_v9plus_runner_cfg,
  ultra_game_yaw_v9plus_env_cfg,
)
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs_v10 import (
  ultra_game_yaw_v10_env_cfg,
)
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs_v11 import (
  ultra_game_yaw_v11_env_cfg,
)
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs_v15 import (
  ultra_game_yaw_amp_him_v15_runner_cfg,
  ultra_game_yaw_v15_env_cfg,
)
from mjlab.tasks.velocity.config.ultra_game_yaw.env_cfgs_v16 import (
  ultra_game_yaw_amp_him_v16_runner_cfg,
  ultra_game_yaw_v16_env_cfg,
)
from mjlab.tasks.velocity.config.ultra_game_yaw.recovery_mdp import (
  DelayedTerminationManager,
  RecoveryEpisodeOutcomeMetric,
  RecoveryTransitionMetric,
)
from mjlab.tasks.velocity.config.ultra_game_yaw.ultra_mdp import (
  select_foot_body_names,
)

_ULTRA_V8_DAMPING_BY_GROUP = {
  ("hip_yaw_l_joint", "hip_yaw_r_joint"): 20.0,
  ("hip_roll_l_joint", "hip_roll_r_joint"): 40.0,
  (
    "hip_pitch_l_joint",
    "hip_pitch_r_joint",
    "knee_pitch_l_joint",
    "knee_pitch_r_joint",
  ): 40.0,
  ("ankle_pitch_l_joint", "ankle_pitch_r_joint"): 10.0,
  ("waist_yaw_joint",): 10.0,
  ("shoulder_pitch_l_joint", "shoulder_pitch_r_joint"): 4.0,
}


@pytest.fixture(scope="module")
def all_task_ids() -> list[str]:
  """Get all registered task IDs."""
  return list_tasks()


def test_all_tasks_loadable(all_task_ids: list[str]) -> None:
  """All registered tasks should be loadable without errors."""
  for task_id in all_task_ids:
    try:
      cfg = load_env_cfg(task_id)
      assert isinstance(cfg, ManagerBasedRlEnvCfg), (
        f"Task {task_id} did not return ManagerBasedRlEnvCfg"
      )
    except Exception as e:
      pytest.fail(f"Failed to load task '{task_id}': {e}")


def test_all_tasks_have_play_config(all_task_ids: list[str]) -> None:
  """All tasks should be loadable in play mode."""
  for task_id in all_task_ids:
    try:
      cfg = load_env_cfg(task_id, play=True)
      assert isinstance(cfg, ManagerBasedRlEnvCfg), (
        f"Task {task_id} play mode did not return ManagerBasedRlEnvCfg"
      )
    except Exception as e:
      pytest.fail(f"Failed to load task '{task_id}' in play mode: {e}")


def test_play_mode_episode_length(all_task_ids: list[str]) -> None:
  """Play mode tasks should have infinite episode length."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id, play=True)
    assert cfg.episode_length_s >= 1e9, (
      f"{task_id} (play mode) episode_length_s={cfg.episode_length_s}, expected >= 1e9"
    )


def test_play_mode_observation_corruption_disabled(all_task_ids: list[str]) -> None:
  """Play mode tasks should have observation corruption disabled for policy."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id, play=True)

    assert "actor" in cfg.observations, (
      f"Play mode task {task_id} missing 'policy' observation group"
    )

    policy_obs = cfg.observations["actor"]
    assert isinstance(policy_obs, ObservationGroupCfg), (
      f"Play mode task {task_id} policy observation is not ObservationGroupCfg"
    )

    assert not policy_obs.enable_corruption, (
      f"Play mode task {task_id} has enable_corruption=True, expected False"
    )


def test_training_mode_observation_corruption_enabled(all_task_ids: list[str]) -> None:
  """Training mode tasks should have observation corruption enabled for policy."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id)

    assert "actor" in cfg.observations, (
      f"Training task {task_id} missing 'policy' observation group"
    )

    policy_obs = cfg.observations["actor"]
    assert isinstance(policy_obs, ObservationGroupCfg), (
      f"Training task {task_id} policy observation is not ObservationGroupCfg"
    )

    assert policy_obs.enable_corruption, (
      f"Training task {task_id} has enable_corruption=False, expected True"
    )


def test_critic_observation_corruption_always_disabled(all_task_ids: list[str]) -> None:
  """Critic observations should always have corruption disabled."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id)

    if "critic" not in cfg.observations:
      continue

    critic_obs = cfg.observations["critic"]
    assert isinstance(critic_obs, ObservationGroupCfg), (
      f"Task {task_id} critic observation is not ObservationGroupCfg"
    )

    assert not critic_obs.enable_corruption, (
      f"Task {task_id} has critic enable_corruption=True, expected False"
    )


def test_play_training_observation_structure_match(all_task_ids: list[str]) -> None:
  """Play and training configs should have matching observation structure."""
  for task_id in all_task_ids:
    training_cfg = load_env_cfg(task_id)
    play_cfg = load_env_cfg(task_id, play=True)

    # Same observation groups.
    assert set(training_cfg.observations.keys()) == set(play_cfg.observations.keys()), (
      f"Observation groups mismatch between {task_id} training and play modes"
    )

    # Same observation terms within each group.
    for obs_group_name in training_cfg.observations:
      training_terms = set(training_cfg.observations[obs_group_name].terms.keys())
      play_terms = set(play_cfg.observations[obs_group_name].terms.keys())

      assert training_terms == play_terms, (
        f"Observation terms mismatch in group '{obs_group_name}' "
        f"between {task_id} training and play modes"
      )


def test_play_training_action_structure_match(all_task_ids: list[str]) -> None:
  """Play and training configs should have matching action structure."""
  for task_id in all_task_ids:
    training_cfg = load_env_cfg(task_id)
    play_cfg = load_env_cfg(task_id, play=True)

    assert set(training_cfg.actions.keys()) == set(play_cfg.actions.keys()), (
      f"Action structure mismatch between {task_id} training and play modes"
    )


def test_play_mode_disables_push_robot(all_task_ids: list[str]) -> None:
  """Play mode tasks should disable push_robot event."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id, play=True)
    assert "push_robot" not in cfg.events, (
      f"Play mode task {task_id} has push_robot event, expected it to be removed"
    )


@pytest.mark.parametrize(
  "env_cfg_fn",
  [
    ultra_game_yaw_v9_env_cfg,
    ultra_game_yaw_v9plus_env_cfg,
    ultra_game_yaw_v10_env_cfg,
  ],
)
def test_ultra_v9_v9plus_v10_use_v8_kd(env_cfg_fn) -> None:
  """V9/V9plus/V10 should keep the V8 retuned actuator damping."""
  cfg = env_cfg_fn()
  robot_cfg = cfg.scene.entities["robot"]
  assert robot_cfg.articulation is not None

  actual = {
    tuple(act.target_names_expr): act.damping
    for act in robot_cfg.articulation.actuators
  }

  assert actual == _ULTRA_V8_DAMPING_BY_GROUP


def test_ultra_v10_uses_ankle_roll_physical_feet_without_changing_v9() -> None:
  """V10's physical foot is ankle_roll; V9 remains ankle_pitch."""
  assert select_foot_body_names(
    ("base_link", "ankle_pitch_l_link", "ankle_pitch_r_link")
  ) == ("ankle_pitch_l_link", "ankle_pitch_r_link")
  assert select_foot_body_names(
    (
      "base_link",
      "ankle_pitch_l_link",
      "ankle_pitch_r_link",
      "ankle_roll_l_link",
      "ankle_roll_r_link",
    )
  ) == ("ankle_roll_l_link", "ankle_roll_r_link")

  cfg = ultra_game_yaw_v10_env_cfg()
  feet_sensor = cast(
    ContactSensorCfg,
    next(
      sensor
      for sensor in cfg.scene.sensors or ()
      if getattr(sensor, "name", None) == "feet_ground_contact"
    ),
  )
  assert feet_sensor.primary.mode == "body"
  assert feet_sensor.primary.pattern == r"^(ankle_roll_l_link|ankle_roll_r_link)$"


def test_ultra_v11_pseudo_inertia_uses_log_alpha_for_mass_scale() -> None:
  """V11 pseudo-inertia should encode 0.8-1.2 mass scale as log alpha."""
  cfg = ultra_game_yaw_v11_env_cfg()
  event = cfg.events["randomize_body_inertia"]

  assert event.params["alpha_range"] == pytest.approx(
    (0.5 * math.log(0.8), 0.5 * math.log(1.2))
  )


def test_ultra_v15_is_v9_flat_with_14mps_motion_removed() -> None:
  """V15 should only remove the 14 m/s run AMP clip from V9."""
  v9_cfg = ultra_game_yaw_v9_env_cfg()
  v15_cfg = ultra_game_yaw_v15_env_cfg()

  assert v15_cfg.scene.terrain is not None
  assert v15_cfg.scene.terrain.terrain_type == "plane"
  assert v15_cfg.rewards.keys() == v9_cfg.rewards.keys()
  assert "terrain_levels_vel" not in v15_cfg.curriculum

  runner_cfg = ultra_game_yaw_amp_him_v15_runner_cfg()
  style_2_names = [
    Path(path).name for path in runner_cfg.amp_motion_files_dict["style_2"]
  ]
  assert style_2_names == ["run_17200_9mps.txt"]


def test_ultra_v16_is_v9_flat_with_feet_height_symmetry_reward_only() -> None:
  """V16 should keep V9 motions and add the foot-height symmetry reward."""
  v9_cfg = ultra_game_yaw_v9_env_cfg()
  v16_cfg = ultra_game_yaw_v16_env_cfg()

  assert v16_cfg.scene.terrain is not None
  assert v16_cfg.scene.terrain.terrain_type == "plane"
  assert "terrain_levels_vel" not in v16_cfg.curriculum
  assert set(v16_cfg.rewards) == set(v9_cfg.rewards) | {"feet_swing_height_symmetry"}

  runner_cfg = ultra_game_yaw_amp_him_v16_runner_cfg()
  v9_runner_cfg = ultra_game_yaw_amp_him_v9_runner_cfg()
  assert runner_cfg.amp_motion_files_dict == v9_runner_cfg.amp_motion_files_dict


def test_ultra_amp_foot_proxy_stays_on_ankle_pitch() -> None:
  """AMP foot endpoint should match the pitch-link motion-file proxy."""
  assert _select_amp_foot_bodies(
    (
      "base_link",
      "ankle_pitch_l_link",
      "ankle_pitch_r_link",
      "ankle_roll_l_link",
      "ankle_roll_r_link",
    )
  ) == ("ankle_pitch_l_link", "ankle_pitch_r_link")


def test_ultra_v9plus_adds_recovery_pipeline_without_changing_actor_obs() -> None:
  """V9plus should add delayed recovery while preserving V9 actor obs terms."""

  v9_cfg = ultra_game_yaw_v9_env_cfg()
  v9plus_cfg = ultra_game_yaw_v9plus_env_cfg()
  v9plus_play_cfg = ultra_game_yaw_v9plus_env_cfg(play=True)

  assert (
    v9plus_cfg.observations["actor"].terms.keys()
    == v9_cfg.observations["actor"].terms.keys()
  )

  init_event = v9plus_cfg.events["init_recovery_motion_loader"]
  reset_event = v9plus_cfg.events["reset_from_recovery_motion"]
  assert init_event.mode == "startup"
  assert reset_event.mode == "reset"
  assert init_event.params["delay_reset_env_ratio"] == 0.4
  assert init_event.params["max_delay_steps"] == 250
  assert (
    v9plus_play_cfg.events["init_recovery_motion_loader"].params[
      "delay_reset_env_ratio"
    ]
    == 1.0
  )

  recovery_dir = Path(init_event.params["recovery_dir"])
  assert reset_event.params["recovery_dir"] == str(recovery_dir)
  assert {
    "Take_040_Skeleton0_50fps.npz",
    "fallAndGetUp1_subject1_best_getup_50fps.npz",
  }.issubset({path.name for path in recovery_dir.iterdir()})

  assert v9plus_cfg.events["ultra_style_update"].func.__name__ == (
    "ultra_recovery_style_update"
  )
  assert "recovery_root_height" in v9plus_cfg.rewards
  assert "recovery_body_orientation" in v9plus_cfg.rewards
  assert v9plus_cfg.rewards["recovery_root_height"].params["std"] == pytest.approx(
    0.3 * 1.80 / 1.33
  )
  assert {
    "recovery_active_ratio",
    "recovery_delay_progress",
    "recovery_attempt_rate",
    "recovery_success_rate",
    "recovery_failure_rate",
    "recovery_success_episode",
    "recovery_failure_episode",
  }.issubset(v9plus_cfg.metrics)
  assert v9plus_cfg.metrics["recovery_success_episode"].reduce == "last"
  assert v9plus_cfg.metrics["recovery_failure_episode"].reduce == "last"
  for term_name in ("ang_vel_xy_l2", "body_contact_neg", "feet_stumble"):
    assert v9plus_cfg.rewards[term_name].params["style_mask"] == [0, 1, 2]

  runner_cfg = ultra_game_yaw_amp_him_v9plus_runner_cfg()
  assert runner_cfg.num_styles == 4
  assert set(runner_cfg.amp_motion_files_dict) == {
    "style_0",
    "style_1",
    "style_2",
    "style_3",
  }
  style_3_files = [Path(path) for path in runner_cfg.amp_motion_files_dict["style_3"]]
  assert all(path.exists() for path in style_3_files)
  assert runner_cfg.amp_reward_coef_dict["style_3"] == 1.0


def test_ultra_v9plus_recovery_metrics_detect_outcomes() -> None:
  """Recovery metrics should distinguish attempts, successes, and failures."""

  manager = DelayedTerminationManager.__new__(DelayedTerminationManager)
  manager._delay_env_mask = torch.tensor([True, True])
  manager._delay_counters = torch.tensor([0, 0])
  manager._max_delay_steps = 250
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    termination_manager=manager,
    reset_buf=torch.tensor([False, False]),
    reset_terminated=torch.tensor([False, False]),
  )
  typed_env = cast(ManagerBasedRlEnv, env)

  attempt = RecoveryTransitionMetric(cfg=None, env=typed_env)
  manager._delay_counters = torch.tensor([3, 0])
  assert attempt(typed_env, mode="attempt").tolist() == [1.0, 0.0]

  success = RecoveryTransitionMetric(cfg=None, env=typed_env)
  assert success(typed_env, mode="success").tolist() == [0.0, 0.0]
  manager._delay_counters = torch.tensor([0, 0])
  assert success(typed_env, mode="success").tolist() == [1.0, 0.0]

  failure = RecoveryTransitionMetric(cfg=None, env=typed_env)
  manager._delay_counters = torch.tensor([5, 0])
  assert failure(typed_env, mode="failure").tolist() == [0.0, 0.0]
  manager._delay_counters = torch.tensor([0, 0])
  env.reset_buf = torch.tensor([True, False])
  env.reset_terminated = torch.tensor([True, False])
  assert failure(typed_env, mode="failure").tolist() == [1.0, 0.0]

  outcome = RecoveryEpisodeOutcomeMetric(cfg=None, env=typed_env)
  env.reset_buf = torch.tensor([False, False])
  env.reset_terminated = torch.tensor([False, False])
  manager._delay_counters = torch.tensor([2, 0])
  assert outcome(typed_env, mode="success").tolist() == [0.0, 0.0]
  manager._delay_counters = torch.tensor([0, 0])
  assert outcome(typed_env, mode="success").tolist() == [1.0, 0.0]
  assert outcome(typed_env, mode="success").tolist() == [1.0, 0.0]
  outcome.reset(env_ids=torch.tensor([0]))
  assert outcome(typed_env, mode="success").tolist() == [0.0, 0.0]
