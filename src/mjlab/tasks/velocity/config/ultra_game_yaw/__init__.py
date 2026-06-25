from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .amp_him import (
  UltraGameYawAMPHIMRunner,
  ultra_game_yaw_amp_him_accel_runner_cfg,
  ultra_game_yaw_amp_him_runner_cfg,
)
from .env_cfgs import (
  ultra_game_yaw_accel_env_cfg,
  ultra_game_yaw_aligned_env_cfg,
  ultra_game_yaw_flat_env_cfg,
)
from .env_cfgs_v2 import (
  ultra_game_yaw_amp_him_v2_runner_cfg,
  ultra_game_yaw_v2_env_cfg,
)
from .env_cfgs_v3 import (
  ultra_game_yaw_amp_him_v3_runner_cfg,
  ultra_game_yaw_v3_env_cfg,
)
from .env_cfgs_v4 import (
  ultra_game_yaw_amp_him_v4_runner_cfg,
  ultra_game_yaw_v4_env_cfg,
)
from .env_cfgs_v5 import (
  ultra_game_yaw_amp_him_v5_runner_cfg,
  ultra_game_yaw_v5_env_cfg,
)
from .env_cfgs_v6 import (
  ultra_game_yaw_amp_him_v6_runner_cfg,
  ultra_game_yaw_v6_env_cfg,
)
from .env_cfgs_v7 import (
  ultra_game_yaw_amp_him_v7_runner_cfg,
  ultra_game_yaw_v7_env_cfg,
)
from .env_cfgs_v8 import (
  ultra_game_yaw_amp_him_v8_runner_cfg,
  ultra_game_yaw_v8_env_cfg,
)
from .env_cfgs_v9 import (
  ultra_game_yaw_amp_him_v9_runner_cfg,
  ultra_game_yaw_v9_env_cfg,
)
from .env_cfgs_v9plus import (
  ultra_game_yaw_amp_him_v9plus_runner_cfg,
  ultra_game_yaw_v9plus_env_cfg,
)
from .env_cfgs_v10 import (
  ultra_game_yaw_amp_him_v10_runner_cfg,
  ultra_game_yaw_v10_env_cfg,
)
from .env_cfgs_v11 import (
  ultra_game_yaw_amp_him_v11_runner_cfg,
  ultra_game_yaw_v11_env_cfg,
)
from .env_cfgs_v12 import (
  ultra_game_yaw_amp_him_v12_runner_cfg,
  ultra_game_yaw_v12_env_cfg,
)
from .env_cfgs_v13 import (
  ultra_game_yaw_amp_him_v13_runner_cfg,
  ultra_game_yaw_v13_env_cfg,
)
from .rl_cfg import ultra_game_yaw_ppo_runner_cfg

# Phase-1: plain PPO (kept for ablation).
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Ultra-GameYaw",
  env_cfg=ultra_game_yaw_flat_env_cfg(),
  play_env_cfg=ultra_game_yaw_flat_env_cfg(play=True),
  rl_cfg=ultra_game_yaw_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Phase-2: AMP + HIM + WGAN MULTIAMPPPO + Ultra-aligned rewards/critic +
# symmetry mirror loss (full alignment with Ultra Isaac-Lab task).
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM",
  env_cfg=ultra_game_yaw_aligned_env_cfg(),
  play_env_cfg=ultra_game_yaw_aligned_env_cfg(play=True),
  rl_cfg=ultra_game_yaw_amp_him_runner_cfg(),
  runner_cls=UltraGameYawAMPHIMRunner,
)

# Phase-2 (parallel "accel" variant): same as AMP-HIM above, plus a 4th AMP
# style fed by the stand-to-run expert clip to shape the start/acceleration
# phase. Trains independently from the baseline task above.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-Accel",
  env_cfg=ultra_game_yaw_accel_env_cfg(),
  play_env_cfg=ultra_game_yaw_accel_env_cfg(play=True),
  rl_cfg=ultra_game_yaw_amp_him_accel_runner_cfg(),
  runner_cls=UltraGameYawAMPHIMRunner,
)

# v2: same AMP-HIM task but with motor-response-aligned actuators (motor friction
# fs/fd added) and lowered kd on the buzz-prone low-load joints (hip_yaw, waist,
# shoulder), to fight the real-robot standstill jitter. Trains independently.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V2",
  env_cfg=ultra_game_yaw_v2_env_cfg(),
  play_env_cfg=ultra_game_yaw_v2_env_cfg(play=True),
  rl_cfg=ultra_game_yaw_amp_him_v2_runner_cfg(),
  runner_cls=UltraGameYawAMPHIMRunner,
)

# v3: same AMP-HIM task but with a softer/safer controller — every joint's kp is
# halved, kd doubled, and the torque-speed safety ceiling (y1/y2/effort_limit)
# scaled to 80%. Keeps v2's motor friction. Targets the real-robot waist
# structural failure (lower kp + lower torque ceiling = smaller torque spikes on
# a position-error transient). Trains independently.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V3",
  env_cfg=ultra_game_yaw_v3_env_cfg(),
  play_env_cfg=ultra_game_yaw_v3_env_cfg(play=True),
  rl_cfg=ultra_game_yaw_amp_him_v3_runner_cfg(),
  runner_cls=UltraGameYawAMPHIMRunner,
)

# v4: baseline AMP-HIM (same Isaac PD + original expert trajectories), but with
# the Isaac-style PROGRESSIVE velocity curriculum (performance-gated +0.2 m/s
# ramp) replacing the fixed-step stage table, plus stand-stability terms
# (feet_slide extended to the stand style + a stand base-velocity penalty). Aims
# to reproduce Isaac's rock-still standstill. Trains independently.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V4",
  env_cfg=ultra_game_yaw_v4_env_cfg(),
  play_env_cfg=ultra_game_yaw_v4_env_cfg(play=True),
  rl_cfg=ultra_game_yaw_amp_him_v4_runner_cfg(),
  runner_cls=UltraGameYawAMPHIMRunner,
)

# v5: v4 env (ORIGINAL Isaac PD + original experts + progressive curriculum +
# stand-stability terms + Isaac heading), PLUS HoST's smoothness loss turned on
# in the PPO update (interpolation-based policy + value smoothness, ported from
# HoST/rsl_rl). This is the "smooth loss + regularization" route to killing
# real-robot jitter WITHOUT lowering kp/kd, stacked on v4's standstill fixes.
# Trains independently.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V5",
  env_cfg=ultra_game_yaw_v5_env_cfg(),
  play_env_cfg=ultra_game_yaw_v5_env_cfg(play=True),
  rl_cfg=ultra_game_yaw_amp_him_v5_runner_cfg(),
  runner_cls=UltraGameYawAMPHIMRunner,
)

# v6: v5 line (v4 stand fixes + HoST smoothness loss) with two changes — the PD
# gains are retuned per-joint (explicit kp/kd table, not v3's global transform),
# and the velocity curriculum is reverted from v4/v5's progressive ramp back to
# the baseline STEP (stage-table) curriculum. Trains independently.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V6",
  env_cfg=ultra_game_yaw_v6_env_cfg(),
  play_env_cfg=ultra_game_yaw_v6_env_cfg(play=True),
  rl_cfg=ultra_game_yaw_amp_him_v6_runner_cfg(),
  runner_cls=UltraGameYawAMPHIMRunner,
)

# v7: v6 (retuned PD + standstill/saturation shaping + HoST smoothness loss +
# Isaac heading) with two anti-lunge changes — the velocity command is
# rate-limited (slew toward the target, restoring ultra_run_lab's max_lin_accel)
# so acceleration is gradual and posture adapts to the current speed instead of
# lunging, and the curriculum is reverted from v6's STEP table back to the
# ultra_run_lab / Isaac PROGRESSIVE ramp. Trains independently.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V7",
  env_cfg=ultra_game_yaw_v7_env_cfg(),
  play_env_cfg=ultra_game_yaw_v7_env_cfg(play=True),
  rl_cfg=ultra_game_yaw_amp_him_v7_runner_cfg(),
  runner_cls=UltraGameYawAMPHIMRunner,
)

# v8: v7 (command slew + restored DR + stronger landing-impact penalties +
# top-speed coverage) with the curriculum reverted from v7's full 0..15 m/s
# from step 0 back to the baseline STEP (stage-table) curriculum (same as v6).
# The command slew already smooths per-step transitions; the stage table adds
# a coarse velocity ramp for early-stage gait quality. Trains independently.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V8",
  env_cfg=ultra_game_yaw_v8_env_cfg(),
  play_env_cfg=ultra_game_yaw_v8_env_cfg(play=True),
  rl_cfg=ultra_game_yaw_amp_him_v8_runner_cfg(),
  runner_cls=UltraGameYawAMPHIMRunner,
)

# v9: the ultra_run_lab hist10 reward set + hist10 velocity curriculum (cap ramp
# with retreat), ported onto the AMP+HIM stack (HIM kept; hist10 is MLP-only).
# All other env config (HIM obs, PD, action_scale, sim, DR, command slew) is
# inherited from v7 so the other versions are untouched. Trains independently.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V9",
  env_cfg=ultra_game_yaw_v9_env_cfg(),
  play_env_cfg=ultra_game_yaw_v9_env_cfg(play=True),
  rl_cfg=ultra_game_yaw_amp_him_v9_runner_cfg(),
  runner_cls=UltraGameYawAMPHIMRunner,
)

# v9plus: v9 with AMP_mjlab-style fall/get-up recovery. A subset of envs delay
# fall termination for a short window and reset from recovery motion frames, so
# the same stand/walk/run policy also learns to recover after a fall.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V9plus",
  env_cfg=ultra_game_yaw_v9plus_env_cfg(),
  play_env_cfg=ultra_game_yaw_v9plus_env_cfg(play=True),
  rl_cfg=ultra_game_yaw_amp_him_v9plus_runner_cfg(),
  runner_cls=UltraGameYawAMPHIMRunner,
)

# v10: v9 (hist10 rewards + curriculum + HIM history=10) with the robot XML
# updated to include two passive ankle-roll DOFs (stiffness=20 Nm/rad,
# damping=0.5). Actor obs stays at 48 dims (13 actuated joints only);
# critic additionally observes ankle_roll_pos/vel. encoder_bias DR is
# restricted to actuated joints. Everything else identical to v9.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V10",
  env_cfg=ultra_game_yaw_v10_env_cfg(),
  play_env_cfg=ultra_game_yaw_v10_env_cfg(play=True),
  rl_cfg=ultra_game_yaw_amp_him_v10_runner_cfg(),
  runner_cls=UltraGameYawAMPHIMRunner,
)

# v11: tn-yaw-fix (aligned AMP+HIM, step-based curriculum) with ultra_run_lab
# new_him DR parity — adds missing body mass / inertia / friction DR, aligns COM
# (waist_yaw_link), encoder bias (±0.03 reset-mode), reset joints (by_scale),
# reset base velocity, and brings command sampling + init state + push frequency
# + action delay in line with the original Isaac-Lab new_him config.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V11",
  env_cfg=ultra_game_yaw_v11_env_cfg(),
  play_env_cfg=ultra_game_yaw_v11_env_cfg(play=True),
  rl_cfg=ultra_game_yaw_amp_him_v11_runner_cfg(),
  runner_cls=UltraGameYawAMPHIMRunner,
)

# v12: v9 (hist10 rewards + curriculum + HIM history=10) with the flat-plane
# terrain replaced by GRAVEL_CURRICULUM_TERRAINS_CFG (matching ultra_run_lab
# hist10's terrain: flat 30%, random_rough 40%, mild up/down slopes).
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V12",
  env_cfg=ultra_game_yaw_v12_env_cfg(),
  play_env_cfg=ultra_game_yaw_v12_env_cfg(play=True),
  rl_cfg=ultra_game_yaw_amp_him_v12_runner_cfg(),
  runner_cls=UltraGameYawAMPHIMRunner,
)

# v13: v10 (15 DoF passive ankle-roll) with GRAVEL_CURRICULUM_TERRAINS_CFG.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V13",
  env_cfg=ultra_game_yaw_v13_env_cfg(),
  play_env_cfg=ultra_game_yaw_v13_env_cfg(play=True),
  rl_cfg=ultra_game_yaw_amp_him_v13_runner_cfg(),
  runner_cls=UltraGameYawAMPHIMRunner,
)
