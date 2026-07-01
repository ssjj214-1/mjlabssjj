"""Tests for per-foot height sensor and terrain-aware rewards."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from conftest import get_test_device, make_scene_and_sim

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

# Platform top at z=0.5 (box center z=0.25, half-height 0.25).
# Body at z=1.0, left_foot at z=0.8 (offset -0.2), right_foot at z=0.6 (offset -0.4).
# Expected: left 0.3m above platform, right 0.1m above platform.
# Both within max_distance=1.0.
TWO_FEET_ABOVE_PLATFORM_XML = """
  <mujoco>
    <worldbody>
      <geom name="platform" type="box" size="5 5 0.25" pos="0 0 0.25"/>
      <body name="base" pos="0 0 1">
        <freejoint name="free_joint"/>
        <geom name="base_geom" type="sphere" size="0.05" mass="1.0"/>
        <site name="left_foot" pos="-0.1 0 -0.2"/>
        <site name="right_foot" pos="0.1 0 -0.4"/>
      </body>
    </worldbody>
  </mujoco>
"""

# Stepped terrain: step top at z=0.4 for x<0, ground at z=0 for x>0.
# Body at x=0, z=0.8. left_foot at x=-0.5 (over step), right_foot at x=0.5 (over ground).
# Expected: left 0.4m above step, right 0.8m above ground.
STEPPED_TERRAIN_XML = """
  <mujoco>
    <worldbody>
      <geom name="ground" type="plane" size="5 5 0.1" pos="0 0 0"/>
      <geom name="step" type="box" size="5 5 0.2" pos="-5 0 0.2"/>
      <body name="base" pos="0 0 0.8">
        <freejoint name="free_joint"/>
        <geom name="base_geom" type="sphere" size="0.05" mass="1.0"/>
        <site name="left_foot" pos="-0.5 0 0"/>
        <site name="right_foot" pos="0.5 0 0"/>
      </body>
    </worldbody>
  </mujoco>
"""


def _foot_sensor_cfg() -> TerrainHeightSensorCfg:
  """Match the shipped config: yaw alignment, max_distance=1.0, group 0."""
  return TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=(
      ObjRef(type="site", name="left_foot", entity="robot"),
      ObjRef(type="site", name="right_foot", entity="robot"),
    ),
    ray_alignment="yaw",
    pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=4),
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
  )


@pytest.fixture(scope="module")
def device():
  return get_test_device()


class _FakeEnv:
  def __init__(self, scene):
    self.scene = scene


class _FakeContactSensor:
  def __init__(self, found, first_contact):
    self.data = type("Data", (), {"found": found})()
    self._first_contact = first_contact

  def compute_first_contact(self, dt):
    del dt
    return self._first_contact


def test_foot_height_flat_platform(device):
  """Two feet at different heights above a flat platform."""
  cfg = _foot_sensor_cfg()
  scene, sim = make_scene_and_sim(device, TWO_FEET_ABOVE_PLATFORM_XML, (cfg,))
  sim.step()
  sim.sense()

  sensor: TerrainHeightSensor = scene["foot_height_scan"]
  heights = sensor.data.heights

  assert heights.shape == (1, 2)
  # Left foot: z=0.8, platform top at z=0.5, height = 0.3.
  assert heights[0, 0].item() == pytest.approx(0.3, abs=0.05)
  # Right foot: z=0.6, platform top at z=0.5, height = 0.1.
  assert heights[0, 1].item() == pytest.approx(0.1, abs=0.05)


def test_foot_height_stepped_terrain(device):
  """Feet over different terrain heights give different readings."""
  cfg = _foot_sensor_cfg()
  scene, sim = make_scene_and_sim(device, STEPPED_TERRAIN_XML, (cfg,))
  sim.step()
  sim.sense()

  sensor: TerrainHeightSensor = scene["foot_height_scan"]
  heights = sensor.data.heights

  assert heights.shape == (1, 2)
  # Left foot at x=-0.5, z=0.8, over step top at z=0.4 -> height ~0.4.
  assert heights[0, 0].item() == pytest.approx(0.4, abs=0.1)
  # Right foot at x=0.5, z=0.8, over ground at z=0 -> height ~0.8.
  assert heights[0, 1].item() == pytest.approx(0.8, abs=0.1)


def test_foot_height_observation(device):
  """foot_height observation delegates to sensor.data.heights."""
  from mjlab.tasks.velocity.mdp.observations import foot_height

  cfg = _foot_sensor_cfg()
  scene, sim = make_scene_and_sim(device, TWO_FEET_ABOVE_PLATFORM_XML, (cfg,))
  sim.step()
  sim.sense()

  env = _FakeEnv(scene)
  obs = foot_height(env, "foot_height_scan")  # type: ignore[invalid-argument-type]
  sensor: TerrainHeightSensor = scene["foot_height_scan"]
  direct = sensor.data.heights

  assert torch.allclose(obs, direct)


def test_feet_swing_height_symmetry_penalizes_uneven_recent_peaks() -> None:
  """Foot-height symmetry cost compares the latest left/right swing peaks."""
  from mjlab.managers.reward_manager import RewardTermCfg
  from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
  from mjlab.tasks.velocity.config.ultra_game_yaw import ultra_mdp

  height_sensor = TerrainHeightSensor.__new__(TerrainHeightSensor)
  height_sensor._num_frames = 2
  height_sensor._cache_valid = True
  height_sensor._cached_data = type(
    "Data",
    (),
    {"heights": torch.tensor([[0.30, 0.10], [0.20, 0.20]])},
  )()
  contact_sensor = _FakeContactSensor(
    found=torch.tensor([[1, 1], [1, 1]]),
    first_contact=torch.tensor([[True, True], [True, True]]),
  )
  env = type(
    "Env",
    (),
    {
      "num_envs": 2,
      "device": "cpu",
      "step_dt": 0.01,
      "scene": {
        "foot_height_scan": height_sensor,
        "feet_ground_contact": contact_sensor,
      },
      "extras": {"log": {}},
      "style_ids": torch.tensor([2, 2]),
    },
  )()
  typed_env = cast(ManagerBasedRlEnv, env)
  reward = ultra_mdp.feet_swing_height_symmetry(
    RewardTermCfg(
      func=ultra_mdp.feet_swing_height_symmetry,
      weight=-1.0,
      params={"height_sensor_name": "foot_height_scan"},
    ),
    typed_env,
  )

  contact_sensor.data.found = torch.tensor([[0, 0], [0, 0]])
  contact_sensor._first_contact = torch.tensor([[False, False], [False, False]])
  assert reward(
    typed_env,
    sensor_name="feet_ground_contact",
    height_sensor_name="foot_height_scan",
    target_height=0.20,
    style_mask=[2],
  ).tolist() == pytest.approx([0.0, 0.0])

  contact_sensor.data.found = torch.tensor([[1, 1], [1, 1]])
  contact_sensor._first_contact = torch.tensor([[True, True], [True, True]])
  cost = reward(
    typed_env,
    sensor_name="feet_ground_contact",
    height_sensor_name="foot_height_scan",
    target_height=0.20,
    style_mask=[2],
  )

  assert cost.tolist() == pytest.approx([1.0, 0.0])


def test_foot_height_multi_env(device):
  """Sensor works correctly across multiple environments."""
  cfg = _foot_sensor_cfg()
  scene, sim = make_scene_and_sim(
    device, TWO_FEET_ABOVE_PLATFORM_XML, (cfg,), num_envs=4
  )
  sim.step()
  sim.sense()

  sensor: TerrainHeightSensor = scene["foot_height_scan"]
  heights = sensor.data.heights

  assert heights.shape == (4, 2)
  # All envs should report same heights (identical geometry).
  for i in range(4):
    assert heights[i, 0].item() == pytest.approx(0.3, abs=0.05)
    assert heights[i, 1].item() == pytest.approx(0.1, abs=0.05)


def test_foot_height_miss_returns_max_distance(device):
  """Feet beyond max_distance report max_distance, not -1."""
  # Body at z=3 with feet at z=2.8 and z=2.6.
  # max_distance=1.0, ground at z=0, so both feet are >1m above ground.
  miss_xml = """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"/>
        <body name="base" pos="0 0 3">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="sphere" size="0.05" mass="1.0"/>
          <site name="left_foot" pos="-0.1 0 -0.2"/>
          <site name="right_foot" pos="0.1 0 -0.4"/>
        </body>
      </worldbody>
    </mujoco>
  """
  cfg = _foot_sensor_cfg()
  scene, sim = make_scene_and_sim(device, miss_xml, (cfg,))
  sim.step()
  sim.sense()

  sensor: TerrainHeightSensor = scene["foot_height_scan"]
  heights = sensor.data.heights

  # Both feet are >1m above ground, beyond max_distance=1.0.
  assert heights[0, 0].item() == pytest.approx(1.0, abs=0.01)
  assert heights[0, 1].item() == pytest.approx(1.0, abs=0.01)


def test_foot_penetration_plane(device):
  """Foot below a ground plane should report near-zero, not max_distance."""
  xml = """
    <mujoco>
      <worldbody>
        <geom name="ground" type="plane" size="5 5 0.1" pos="0 0 0"/>
        <body name="base" pos="0 0 0.05">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="sphere" size="0.02" mass="1.0" group="1"/>
          <site name="left_foot" pos="0 0 -0.04"/>
          <site name="right_foot" pos="0 0 -0.06"/>
        </body>
      </worldbody>
    </mujoco>
  """
  # left_foot at z=0.01 (above), right_foot at z=-0.01 (below).
  cfg = _foot_sensor_cfg()
  scene, sim = make_scene_and_sim(device, xml, (cfg,))
  sim.step()
  sim.forward()
  sim.sense()

  heights = scene["foot_height_scan"].data.heights[0]
  assert heights[0].item() < 0.1
  assert heights[1].item() < 0.5


def test_foot_penetration_box(device):
  """Foot inside box terrain should report near-zero, not box thickness.

  Regression: rays inside a box hit the bottom face, producing a bogus
  height equal to the box thickness.
  """
  xml = """
    <mujoco>
      <worldbody>
        <geom name="terrain" type="box" size="5 5 0.5" pos="0 0 0.5"/>
        <body name="base" pos="0 0 1.05">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="sphere" size="0.02" mass="1.0" group="1"/>
          <site name="left_foot" pos="0 0 -0.04"/>
          <site name="right_foot" pos="0 0 -0.08"/>
        </body>
      </worldbody>
    </mujoco>
  """
  # Terrain top at z=1.0. left_foot at z=1.01, right_foot at z=0.97.
  cfg = _foot_sensor_cfg()
  scene, sim = make_scene_and_sim(device, xml, (cfg,))
  sim.step()
  sim.forward()
  sim.sense()

  heights = scene["foot_height_scan"].data.heights[0]
  assert heights[0].item() < 0.1
  assert heights[1].item() < 0.5
