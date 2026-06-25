# Ultra -> mjlab Motor / Joint Audit

Date: 2026-06-03
Scope: compare Ultra Isaac config and mjlab migration for:
- motor control mode
- kp/kv/effort
- joint position limits
- joint max velocity limits

## 1) Executive conclusion

- Motor mode: ALIGNED
  - Ultra: delayed PD position control (CustomDelayedPDActuatorCfg)
  - mjlab: MuJoCo builtin position actuator (BuiltinPositionActuatorCfg)
- kp/kv/effort: ALIGNED
- Joint position limits (hard ranges): ALIGNED (13/13 exact match)
- soft joint limit factor: ALIGNED (0.9)
- Joint max velocity limits: ALIGNED
  - Implemented via Ultra-equivalent asymmetric torque-speed envelope X1/X2/Y1/Y2.
  - This matches Ultra's CustomDelayedPDActuator clipping behavior.

## 2) Source files checked

Ultra side:
- /home/ps/ultra_run_lab/legged_lab/assets/ultra_GameYaw/ultra_game_yaw.py
- /home/ps/ultra_run_lab/legged_lab/assets/ultra_GameYaw/mjcf/ultra_game_yaw.xml
- /home/ps/ultra_run_lab/legged_lab/scripts/sim2sim_ultraYaw.py

mjlab side:
- /home/ps/mjlab/src/mjlab/asset_zoo/robots/ultra_game_yaw/ultra_constants.py
- /home/ps/mjlab/src/mjlab/asset_zoo/robots/ultra_game_yaw/xmls/ultra_game_yaw.xml
- /home/ps/mjlab/src/mjlab/actuator/builtin_actuator.py
- /home/ps/mjlab/src/mjlab/actuator/ultra_delayed_pd_actuator.py

## 3) Motor parameters (kp/kv/effort)

Grouped values are identical between Ultra and mjlab:

- hip_yaw_*: kp=500, kv=10, effort=250
- hip_roll_*: kp=700, kv=20, effort=657
- hip_pitch_*: kp=700, kv=10, effort=657
- knee_pitch_*: kp=700, kv=10, effort=657
- ankle_pitch_*: kp=100, kv=6, effort=400
- waist_yaw_joint: kp=400, kv=8, effort=250
- shoulder_pitch_*: kp=50, kv=3, effort=140

Action scale:
- Ultra: 0.25
- mjlab: 0.25

## 4) Joint position limits (hard range) audit

Checked fields from MJCF for each of 13 joints:
- range
- armature
- actuatorfrcrange

Result:
- 13/13 exact match (Ultra XML vs mjlab XML)
- No differences found

Per-joint ranges (rad):
- hip_yaw_l_joint: [-0.436332, 0.436332]
- hip_roll_l_joint: [-0.349066, 0.523599]
- hip_pitch_l_joint: [-2.79253, 2.0944]
- knee_pitch_l_joint: [-0.0872665, 2.44346]
- ankle_pitch_l_joint: [-1.22173, 1.0472]
- hip_yaw_r_joint: [-0.436332, 0.436332]
- hip_roll_r_joint: [-0.523599, 0.349066]
- hip_pitch_r_joint: [-2.79253, 2.0944]
- knee_pitch_r_joint: [-0.0872665, 2.44346]
- ankle_pitch_r_joint: [-1.22173, 1.0472]
- waist_yaw_joint: [-1.5708, 1.5708]
- shoulder_pitch_l_joint: [-2.96706, 2.96706]
- shoulder_pitch_r_joint: [-2.96706, 2.96706]

soft_joint_pos_limit_factor:
- Ultra: 0.9
- mjlab: 0.9

## 5) Joint max velocity limits audit

Ultra explicit velocity_limit_sim (rad/s):
- hip_yaw_*: 20.9
- hip_roll_*: 18.8
- hip_pitch_*: 26.1
- knee_pitch_*: 26.1
- ankle_pitch_*: 26.1
- waist_yaw_joint: 20.9
- shoulder_pitch_*: 26.1

mjlab current status:
- Control path uses UltraDelayedPdActuatorCfg (custom actuator) with:
  - delayed command path: delay_min_lag=0, delay_max_lag=4
  - identical asymmetric torque-speed envelope X1/X2/Y1/Y2
  - optional friction terms Fs/Fd/Va (defaults match Ultra cfg usage)
- Therefore velocity-dependent torque limiting is now behaviorally aligned with Ultra.

## 6) Final parity status

- Motor mode parity: YES
- kp/kv/effort parity: YES
- joint range parity: YES
- max velocity parity: YES

## 7) Notes

- The MuJoCo/Issac numerical integrators are still different engines, but the actuator law
  and limits are now matched at the control-physics interface level.
- For deployment-oriented parity, this is the required level (same commanded law and constraints).
