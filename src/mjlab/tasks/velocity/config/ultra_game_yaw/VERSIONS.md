# Ultra GameYaw 版本差异总览

本文按版本说明 Ultra GameYaw 训练任务相对“最原始版本”的变化。这里的
“最原始版本”指第一个完整 AMP-HIM 任务：

`Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM`

也就是 13 DoF Ultra GameYaw、平地、Multi-AMP(WGAN) + HIM-GRU + PPO +
对称镜像损失、原始 Ultra XML/PD、step 速度课程、aligned reward set 的
baseline。纯 PPO 任务单独列为消融，不作为 V2-V13 的比较基准。

## 通用稳定性修复(框架层，影响所有 AMP+HIM 版本)

以下修复不属于某个版本，而是修在共享的 XML / 框架 / 算法里，所有 AMP+HIM
版本(baseline、V9–V16、V9plus)一并生效。它们解决的是"加地形 / 上高速后
训练发散到 NaN"的共性问题——根因都是**移植时丢掉了 Isaac 端的边界保护**，
不是物理本身(同机器人同 15 m/s 在 IsaacSim/ultra_run_lab 能稳定训练)。

1. **obs / action ±100 钳位(Isaac parity，治本)**：ultra_run_lab 每步把
   actor/critic obs 和 action 都 `clip(±100)`，mjlab 移植时两者都丢了。
   mjlab 的 critic obs 含无界项(裸接触力 `foot_force_z`、yaw 系
   `base_lin_vel`、`joint_vel`)，高速接触一爆 → 污染 critic obs(HIM 护栏
   触发，日志 `estimation/swap loss = 0`)→ 策略输出爆 → 未封顶的动作惩罚
   (`hip_yaw_action`/`action_rate_l2`)爆成 ~1e31 → `value/symmetry loss` NaN
   → 死锁。修复：`clip_observations()` 给所有 obs 项设 `clip=(-100,100)`
   (在 aligned env 和 `add_terrain_relative_base_height` 各调一次)，AMP+HIM
   runner 设 `clip_actions=100.0`。这是 V11/V12/V13 高速 NaN 的根因修复。
2. **PPO minibatch 级非有限跳过(纵深防御)**：`multi_amp_ppo` 在 loss 非有限
   时跳过该 minibatch 的 backward / step / 记账(对齐 Isaac AMP-PPO),并记录
   `skipped_minibatches`，避免单个坏 minibatch 毒化梯度或把日志刷成 NaN。
3. **HIM estimator 非有限护栏 + sinkhorn 稳定**：estimator 独立优化器在输入
   非有限或 loss/梯度非有限时跳过 step(否则一次坏 batch 就把 encoder/target/
   proto 永久写成 NaN)；`sinkhorn` 减最大值 + 分母兜底，杜绝 Inf/NaN。
4. **碰撞体全部改为 EPA 友好的光滑体**：脚=胶囊，`base`/`waist`=球，
   `hip_pitch`/`knee_pitch`=保长胶囊(base 与 V10 两个 XML)。圆柱-vs-heightfield
   走 GJK/EPA，扁盘压斜面会退化触发 "EPA horizon isn't large enough" + 垃圾
   接触力,主要在摔倒/恢复(V9plus)时出现;改光滑体后实测躺地 EPA 39 → 0。

## Baseline: AMP-HIM

任务名：`Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM`

网络 / 算法：

- ActorCritic + HIM-GRU velocity estimator。
- actor 单帧观测 48 维，actor history 10 帧；actor 实际只取最近 5 帧 +
  HIM 估计速度 + latent 进 MLP。
- critic 单帧 privileged obs 约 100 维，critic history 默认 1 帧。
- Multi-style AMP(WGAN)，style 0/1/2 对应 stand/walk/run。
- 对称数据增强 + mirror loss。
- policy std 原设计是 scalar std；当前代码已增加 `min_normalized_std`
  clamp，避免 std 被优化到负数导致训练中断。

结构 / 命令 / 地形：

- 平地 plane，移除 rough-terrain scanner。
- 4096 env 常用；400 Hz physics，100 Hz control。
- velocity command 为 step-style 采样，step curriculum 逐步把 `lin_vel_x`
  上限推到 15 m/s。
- heading hold 比例较高，用于训练高速直行保持。

奖励：

- Ultra aligned reward set：速度跟踪、yaw 跟踪、高速直行 yaw 抑制、
  stand_still、alive、termination、energy、action rate、joint limit、
  ankle limit、foot slide/contact/impact/stumble、feet spacing、body posture 等。

XML / 资产：

- 原始 13 DoF Ultra GameYaw XML。
- 原始 actuator/PD 配置来自 `ultra_constants.py`。
- AMP 专家轨迹为 stand/walk/run。

## Plain PPO 消融

任务名：`Mjlab-Velocity-Flat-Ultra-GameYaw`

相对 AMP-HIM baseline：

- 网络 / 算法：普通 PPO runner，没有 AMP、HIM、WGAN 判别器、style 调度、
  mirror loss。
- 结构 / 奖励 / XML：使用 Ultra GameYaw velocity env 的早期平地配置。
- 用途：只用于消融，不作为正式训练线。

## Accel

任务名：`Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-Accel`

相对 AMP-HIM baseline：

- 网络 / 算法：Multi-AMP style 从 3 个扩到 4 个，新增 `style_3`。
- 结构：episode 前约 2.4 s 的 run 起步窗口被标记为 accel style，窗口后回到 run。
- 奖励：run 相关奖励 mask 扩展到 accel style，起跑窗口同时受速度跟踪和 accel AMP 风格约束。
- XML / 资产：新增 stand-to-run AMP 专家轨迹
  `accel_stand_to_run.txt`；机器人 XML 不变。
- 目的：专门塑造站立到冲刺的起跑动作，不影响 baseline 任务。

## V2

任务名：`Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V2`

相对 AMP-HIM baseline：

- 网络 / 算法：不变。
- 结构 / 课程 / 奖励：不变。
- XML / actuator：仍是 13 DoF XML，但替换 actuator 参数：
  加入估计的 motor friction (`fs`/`fd`/`va`)，并降低低负载易抖关节
  的 damping，例如 hip_yaw、waist、shoulder。
- 目的：对齐真实电机摩擦，压真机 standstill 高频抖动。

## V3

任务名：`Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V3`

相对 AMP-HIM baseline：

- 网络 / 算法：不变。
- 结构 / 课程 / 奖励：不变。
- XML / actuator：仍是 13 DoF XML，但全局改 actuator：
  `kp * 0.5`、`kd * 2.0`、`effort/y1/y2 * 0.8`，并保留 V2 的 motor friction。
- 目的：降低位置误差瞬态带来的结构风险，给腰部和电机力矩留安全余量。

## V4

任务名：`Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V4`

相对 AMP-HIM baseline：

- 网络 / 算法：不变。
- 结构 / 命令：把 step curriculum 换成 Isaac-style performance-gated
  progressive curriculum；heading 分布对齐 Isaac (`rel_heading_envs=0.6`,
  `heading_control_stiffness=0.5`)。
- 奖励：`feet_slide` 扩展到 stand style；新增 `stand_base_vel` 压站立漂移。
- XML / actuator：原始 13 DoF XML 和原始 PD，不走 V2/V3 的软 PD。
- 目的：修复 mjlab 自训策略站立碎步/漂移，同时尽量贴近 Isaac baseline。

## V5

任务名：`Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V5`

相对 AMP-HIM baseline：

- 网络 / 算法：继承 V4，并在 PPO update 中开启 HoST-style smoothness loss：
  对 policy mean 和 value 做观测插值平滑约束。
- 结构 / 命令：继承 V4 的 progressive curriculum 和 heading 对齐。
- 奖励：继承 V4；没有新增 reward-side action L2，平滑来自算法 loss。
- XML / actuator：原始 13 DoF XML 和原始 PD。
- 目的：不降低 kp/kd，通过网络级平滑压真实机器人高频动作抖动。

## V6

任务名：`Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V6`

相对 AMP-HIM baseline：

- 网络 / 算法：继承 V5，HoST smoothness loss 开启。
- 结构 / 命令：课程从 V4/V5 progressive 改回 baseline step curriculum。
- 奖励：新增/加强站立和安全项：
  `stand_base_motion` 替代 V4 的 `stand_base_vel`，新增 `stand_joint_vel`，
  加强 `joint_pos_limits` 和 `ankle_pitch_pos_limits`，新增 `torque_saturation`。
- XML / actuator：仍是 13 DoF XML，但换成手调 per-joint PD：
  hip_yaw 400/20，hip_roll 600/40，hip/knee pitch 600/40，
  ankle 100/10，waist 200/10，shoulder 40/4。
- 目的：结合更强阻尼比 PD、step 课程和站立/饱和 shaping，压原地晃动和 5 m/s 饱和。

## V7

任务名：`Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V7`

相对 AMP-HIM baseline：

- 网络 / 算法：继承 V6/V5，HoST smoothness loss 开启。
- 结构 / 命令：新增 command slew/ramp，`max_lin_accel=4.0`，
  `cmd_step_sample_frac=0.5`；课程改为 ultra_run_lab-style progressive，
  但初始速度上限已经是 15 m/s，因此实际从 step 0 覆盖全速域。
- DR：补齐更多 ultra_run_lab parity：pseudo_inertia、waist payload、
  reset base velocity、wider encoder bias/base COM、top-speed command sampling。
- 奖励：继承 V6，并强化 landing-impact 相关项的意图
  (`contact_impact_vel`、`gait_feet_force_max_neg`)。
- XML / actuator：继承 V6 的 13 DoF + retuned PD。
- 目的：解决低速命令 step 导致的前冲/lunge，并补齐迁移中漏掉的 DR/command sampling。

## V8

任务名：`Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V8`

相对 AMP-HIM baseline：

- 网络 / 算法：继承 V7，HoST smoothness loss 开启。
- 结构 / 命令：保留 V7 command slew、DR、top-speed sampling，但把课程从
  V7 的 full-range progressive/no-op 改回 baseline step curriculum。
- 奖励：继承 V7/V6。
- XML / actuator：继承 V6/V7 的 13 DoF + retuned PD。
- 目的：保留 per-step 命令平滑，同时恢复 coarse velocity ramp，提高早期 gait 质量。

## V9

任务名：`Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V9`

相对 AMP-HIM baseline：

- 网络 / 算法：继承 V7 line，保留 HIM；actor history 10 帧；
  critic history 改为 10 帧，所以 critic MLP 输入从约 100 维变为约 1000 维。
- 结构 / 命令：对齐 ultra_run_lab hist10 command sampling：
  `max_lin_accel=3.0`、`cmd_step_sample_frac=0.3`、
  stand/walk/run bucket sampling、heading 0.6/0.9。
- 课程：hist10 performance-gated cap ramp with retreat，速度上限从 4 m/s
  以 0.5 m/s 进退到 15 m/s。
- 奖励：完整替换为 hist10 reward set，恢复 EMA 平均速度跟踪、
  2nd-order action smoothness、action L2、arm/waist posture、hip roll/yaw action
  和 deviation、dof velocity、biped feet air time 等。
- XML / actuator：13 DoF XML，继承 V7 的 retuned PD / DR / command slew。
- 目的：把 ultra_run_lab hist10 reward/curriculum/command distribution 移植到 AMP+HIM stack。

## V9plus

任务名：`Mjlab-Velocity-Rough-Ultra-GameYaw-AMP-HIM-V9plus`

相对 AMP-HIM baseline：

- 网络 / 算法：继承 V9，但 AMP style 从 3 个扩到 4 个，新增 recovery style。
- 结构：一部分 env 延迟 fall termination，并从 fall/get-up motion frames reset。
- 地形：与 V12 一致，换成 GRAVEL_CURRICULUM_TERRAINS_CFG，并启用 terrain-relative
  base height + rough-terrain solver 容量。recovery reset 的 root z 改为
  `terrain_z + motion_z`（地形上不再穿模/悬空），recovery root-height 奖励改为
  地形相对（复用 base 下射 height sensor）。
- 奖励 / metrics：新增 recovery root height、body orientation 奖励和 recovery
  active/progress/attempt/success/failure metrics；恢复期间放宽部分 locomotion penalty mask。
- XML / 资产：机器人 XML 继承 base（含胶囊脚)；新增 recovery motion 数据和 recovery AMP motion。
- 目的：让同一个 stand/walk/run policy 在地形上同时见到摔倒恢复状态。

## V10

任务名：`Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V10`

相对 AMP-HIM baseline：

- 网络 / 算法：继承 V9，actor history 10，critic history 10，HoST smoothness 开启。
- 结构 / 观测：actor 的 `joint_pos`/`joint_vel` 限定为 13 个 actuated joints，
  actor obs 仍保持 48 维；critic 也限制 actuated joints，同时新增
  `ankle_roll_pos` 和 `ankle_roll_vel` critic-only 观测。
- 奖励 / 课程：继承 V9 hist10 reward/curriculum。
- XML / 资产：切到 V10 15 DoF XML，新增两个 passive ankle-roll joints
  (`ankle_roll_l/r_joint`)；不进入 actor、不被 actuator 控制。
- 接触：foot contact sensor 从 ankle_pitch bodies 改到 ankle_roll bodies。
- 目的：测试带被动 ankle-roll 物理脚的模型，同时保持 policy 输入和 V9 兼容。

## V11

任务名：`Mjlab-Velocity-Rough-Ultra-GameYaw-AMP-HIM-V11`

相对 AMP-HIM baseline：

- 网络 / 算法：回到 base AMP-HIM runner，critic history 为 1 帧；没有 V9 的
  10-frame critic，也没有 HoST smoothness loss。
- 结构 / 地形：当前 V11 已切到 `GRAVEL_CURRICULUM_TERRAINS_CFG`，不是纯平地。
- DR / reset：补齐 new_him parity：waist_yaw_link body mass、all-body pseudo_inertia、
  full-body pair friction、reset base velocity、waist COM、encoder bias reset-mode
  ±0.03、reset joints by_scale、push interval 5-10 s。
- 命令 / init：command resampling 固定 10 s；base z 从 1.20 改 1.23；
  action delay max 从 4 改 3。
- 奖励：继承 aligned reward set，但把 yaw tracking 权重调回 new_him 风格：
  `track_ang_vel_z` 8 -> 3，`track_ang_vel_z_bonus` 3 -> 1；
  移除 `ang_vel_z_straight_neg` 和 `arm_dof_pos_neg`。
- XML / actuator：13 DoF XML，PD retune 到 V9/V6 表。
- 目的：在 tn-yaw-fix/aligned baseline 上补齐 new_him 的 DR、reset、command 和 PD 差异。

## V12

任务名：`Mjlab-Velocity-Rough-Ultra-GameYaw-AMP-HIM-V12`

相对 AMP-HIM baseline：

- 网络 / 算法：继承 V9；actor history 10，critic history 10，critic 输入约 1000 维；
  HoST smoothness loss 开启。
- 结构 / 地形：V9 + `GRAVEL_CURRICULUM_TERRAINS_CFG`，匹配 hist10 gravel terrain
  设置：flat、random rough、mild slopes 的 terrain curriculum。
- 奖励 / 课程 / 命令：完全继承 V9 hist10 reward/curriculum/command sampling。
- XML / actuator：13 DoF XML，继承 V9/V7 line 的 retuned PD / DR。
- 目的：测试 hist10 reward/curriculum 在 gravel curriculum terrain 上的表现。

## V13

任务名：`Mjlab-Velocity-Rough-Ultra-GameYaw-AMP-HIM-V13`

相对 AMP-HIM baseline：

- 网络 / 算法：继承 V10；actor history 10，critic history 10，critic 输入包含
  passive ankle-roll critic-only state。
- 结构 / 地形：V10 + `GRAVEL_CURRICULUM_TERRAINS_CFG`。
- 奖励 / 课程 / 命令：继承 V10/V9 hist10 reward/curriculum/command sampling。
- XML / 资产：15 DoF V10 XML，两个 passive ankle-roll joints；physical foot contact
  使用 ankle-roll bodies。
- 目的：把 passive ankle-roll XML 和 gravel terrain 组合起来，作为 V10 的地形版。

## V14

任务名：`Mjlab-Velocity-Rough-Ultra-GameYaw-AMP-HIM-V14`

相对 AMP-HIM baseline：

- 与 V12 **完全相同**,仅 experiment / wandb 名不同,作为独立训练槽。
- 网络 / 算法：继承 V9；actor history 10，critic history 10，critic 输入约 1000 维；
  HoST smoothness loss 开启。
- 结构 / 地形：V9 + `GRAVEL_CURRICULUM_TERRAINS_CFG`(ultra_run_lab hist10 真实地形的
  忠实移植：flat 0.30、random_rough 0.40、上/下缓坡各 0.15，本身不含台阶)。
  `terrain_levels_vel` 距离课程(= mjlab 自带 = hist10 `update_terrain_levels`,
  逐行一致)、地形相对 base height、rough-terrain solver 容量均与 V12 一致。
- 奖励 / 课程 / 命令：完全继承 V9 hist10 reward/curriculum/command sampling。
- XML / actuator：13 DoF XML，继承 V9/V7 line 的 retuned PD / DR。
- 备注：reset 位置已天然地形相对——`reset_root_state_uniform` 通过
  `env_origins`(含地形表面高度)向量相加带入,不存在 AMP_MjLab 那种"只写 z、
  丢地形高度"的 bug;恢复 reset 路径(仅 V9plus)也已在 V9plus 修过。
- 目的：作为 V12(= hist10 地形忠实移植)的独立重跑/对照槽。

## 版本线索速查

| 版本 | 网络 / 算法 | 结构 / 课程 / DR | 奖励 | XML / 资产 |
| --- | --- | --- | --- | --- |
| Plain PPO | 无 AMP/HIM | 早期平地 velocity env | 普通 velocity reward | 13 DoF |
| AMP-HIM | AMP+HIM+symmetry | 平地 + step curriculum | aligned reward set | 13 DoF 原始 PD |
| Accel | + style_3 accel | 起跑窗口 style 调度 | run reward mask 覆盖 accel | + stand-to-run AMP |
| V2 | 同 baseline | 同 baseline | 同 baseline | motor friction + 低负载 kd 下调 |
| V3 | 同 baseline | 同 baseline | 同 baseline | kp 半、kd 双、力矩上限 80% |
| V4 | 同 baseline | progressive curriculum + Isaac heading | stand foot slide + base drift | 原始 13 DoF |
| V5 | + HoST smooth loss | 继承 V4 | 继承 V4 | 原始 13 DoF |
| V6 | HoST on | step curriculum | 更强 stand / limit / saturation | 13 DoF + retuned PD |
| V7 | HoST on | command slew + full-speed/progressive + DR parity | landing-impact intent | 13 DoF + retuned PD |
| V8 | HoST on | V7 + step curriculum | 继承 V7 | 13 DoF + retuned PD |
| V9 | critic history 10 | hist10 command/curriculum | hist10 reward set | 13 DoF + retuned PD |
| V9plus | V9 + recovery style | delayed fall/get-up reset | recovery rewards/metrics | + recovery motions |
| V10 | V9 network | V9 structure | V9 rewards | 15 DoF passive ankle-roll |
| V11 | base AMP-HIM, critic history 1 | gravel + new_him DR/reset parity | yaw weight rebalance | 13 DoF + retuned PD |
| V12 | V9 network | V9 + gravel | V9 hist10 rewards | 13 DoF + retuned PD |
| V13 | V10 network | V10 + gravel | V10/V9 rewards | 15 DoF passive ankle-roll |
| V14 | 同 V12 | 同 V12 (V9 + gravel, 独立训练槽) | V9 hist10 rewards | 13 DoF + retuned PD |
