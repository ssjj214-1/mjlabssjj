# Ultra GameYaw 训练 / Play 指南

13-DoF 双足机器人 `ultra_game_yaw` 的速度跑步策略（PPO + Multi-AMP(WGAN) + HIM-GRU +
对称镜像损失）。本文给出从头训练、续训、play 可视化的命令，以及与部署（ONNX/sim2sim）的衔接。

版本差异总览见 [`VERSIONS.md`](VERSIONS.md)：按 Plain/AMP-HIM/V2...V13 说明网络、结构/课程/DR、奖励、XML/资产变化。

任务名（gym id）：

- `Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM` —— 正式任务（AMP+HIM+对称，奖励/critic 全对齐）。
- `Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-Accel` —— **并行的「起跑加速」变体**：在正式任务
  基础上**额外加一条 stand-to-run 专家轨迹**（第 4 个 AMP style）来塑造从站立到冲刺的起跑加速段。
  与正式任务**互不影响、可同时训练**。详见第七节。
- `Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V4` —— **渐进式课程 + 站立稳定修复**：用 Isaac 风格
  的性能门控渐进课程替换阶跃课程，并对齐 heading 参数、加站立稳定项，专治"站不住/碎步"。
  PD 和专家轨迹保持最原始。详见第八节。
- `Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V5` —— **V4 + HoST smooth loss**：在 V4 全部站立
  修复之上，再在 PPO update 里加 HoST 的平滑损失（插值式 policy + value 平滑，移植自 HoST 库），
  **不降 kp/kd** 来止真机抖动。是 V2/V3（软 PD）的干净对照。详见第九节。
- `Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V6` —— **V5 + 自定义阻尼比 PD + 阶梯式课程**：在 V5
  基础上换上手调的 per-joint kp/kd、课程改回阶跃式，并升级站立 shaping（站立环境比例分段 0.5→0.1 +
  命令门控的整机晃动惩罚 `stand_base_motion` + 全关节速度惩罚 `stand_joint_vel`）治 V5 的原地漂移/
  躯干与肢体乱晃，并加狠关节限位 + 新增力矩饱和惩罚 `torque_saturation` 治 5 m/s 力矩饱和。详见第十节。
- `Mjlab-Velocity-Flat-Ultra-GameYaw` —— 纯 PPO，仅用于对照消融。

---

## 一、环境

```bash
cd /home/ps/mjlab
# 所有命令都用 uv run，不要直接用 python。
```

---

## 二、从头训练（电机域随机化已开启）

```bash
cd /home/ps/mjlab
ULTRA_MOTOR_DR=1 uv run train Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM \
  --env.scene.num-envs 4096 \
  --agent.run-name tn-yaw-fix \
  --agent.max-iterations 20000
```

- `ULTRA_MOTOR_DR=1`：**打开电机域随机化**（per-joint 输出力矩 scale 0.7~1.2 +
  PD 增益 scale 0.7~1.2，startup 模式跨 episode 持久）。这是默认值，写出来是为了明确；
  设 `ULTRA_MOTOR_DR=0` 可关闭。play 时自动关闭，用标称电机。
- `--env.scene.num-envs`：并行环境数，按显存调整（4096 是常用值，显存不足就调小）。
- `--agent.run-name`：日志子目录后缀，日志落在
  `logs/rsl_rl/ultra_game_yaw_amp_him/<时间戳>_<run-name>/`。
- 速度课程自动从 1.5 m/s 逐级升到 15 m/s（见 `env_cfgs.py` 的 `command_vel` 阶段表）。
- 日志：TensorBoard + W&B（W&B 不上传 checkpoint，只记标量）。
  TensorBoard：`uv run tensorboard --logdir logs/rsl_rl`。

### 续训（接着某个 checkpoint）

```bash
cd /home/ps/mjlab
ULTRA_MOTOR_DR=1 uv run train Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM \
  --env.scene.num-envs 4096 \
  --agent.resume \
  --agent.load-run 2026-06-05_10-05-36_aligned-fix \
  --agent.load-checkpoint model_13800.pt
```

> 续训会自动对齐 `common_step_counter`，让速度课程从对应阶段继续，而不是回到第一阶段。

---

## 三、Play（可视化 / 手动遥控）

```bash
cd /home/ps/mjlab
uv run play Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM \
  --checkpoint-file logs/rsl_rl/ultra_game_yaw_amp_him/<RUN>/model_XXXX.pt \
  --num-envs 1
```

- `--keyboard-cmd`：开启键盘遥控（关闭随机命令重采样），用方向键/按键给速度命令观察跑动。
- `--no-terminations`：禁用摔倒终止，便于持续观察。
- `--video`：录制视频到 `logs/.../videos/play`。
- `--viewer native|viser|auto`：选择查看器（默认 auto）。

---

## 四、导出 ONNX 部署到机器人 / sim2sim

训练出的 `.pt` 用 `ultra2026_rl_sdk` 的导出脚本转 ONNX（输出已是 ISAAC 关节顺序）：

```bash
cd /home/ps/mjlab
uv run python /home/ps/ultra2026_rl_sdk/tools/export_mjlab_onnx.py \
  --checkpoint logs/rsl_rl/ultra_game_yaw_amp_him/<RUN>/model_XXXX.pt
```

部署/sim2sim 的完整说明见 `ultra2026_rl_sdk/DEPLOY_MJLAB_README.md`。

---

## 五、高速直线偏航优化（已生效）

为解决"高速跑偏、上跑道会冲出"的问题，奖励/命令里针对 run 风格做了加强（`env_cfgs.py` /
`ultra_mdp.py`）。**注意：15 m/s 下 0.15 rad/s 的残余偏航就能把 100m 直线弯出赛道**
（转弯半径 = v/ω = 100m，100m 内转近 1 rad），所以第二轮把容差进一步收紧：

- `track_ang_vel_z`：偏航角速度容差 `std` 随速度收紧到 **`std_speed_max=0.08`**（原 0.15），
  权重 **5→8**。
- `track_ang_vel_z_bonus`：同样收紧到 0.08，权重 **2→3**。
- `ang_vel_z_straight_neg`：L1 偏航误差惩罚（随前进速度放大、仅在偏航命令≈0 时激活），
  权重 **−6→−10**，专压"高速直行航向漂移"。
- **命令侧（关键）**：`rel_heading_envs 0.6→0.85`，让更多 env 用 **航向保持** 训练
  （`wz = clip(heading_control_stiffness·heading_err, ±0.5)`）；`heading_control_stiffness`
  显式设为 **1.0**。这正是部署侧 heading-hold 喂给策略的信号，必须和部署 `heading_kp` 一致
  （见下），否则部署纠偏比训练弱、修不回来。

> 部署对齐：`ultra2026_rl_sdk/policy/loco_mode/config/LocoMode.yaml` 的 `heading_kp` 已从
> 0.5 → **1.0** 对齐本处 `heading_control_stiffness`。两边不一致是高速跑偏的直接原因之一。

训练时关注 TensorBoard：`Rewards/ang_vel_z_straight_neg` 应趋近 0，`Rewards/track_ang_vel_z` 应升高。

---

## 六、关于 400m（含转弯）赛道

当前策略**能转弯**：命令里 `ang_vel_z ∈ (-0.5, 0.5) rad/s`，标准 400m 内道弯半径约 36.5m，
所需偏航角速度 = v/r，10 m/s 时约 0.27、15 m/s 时约 0.41 rad/s，都在训练范围内。
新增的直行偏航惩罚按 `|cmd_yaw|<0.3` 门控，转弯命令较大时基本不介入，不会压制主动转弯。

但**高速过弯**还有两个待优化点（如果 400m 是主要目标，建议追加）：

1. **侧倾（banking）**：平地高速转弯需要靠摩擦提供向心力、身体要往弯内侧倾。15 m/s、0.41 rad/s
   时向心加速度 ~6 m/s²，需侧倾约 30°。而当前 `body_orientation_speed_aware` 的 roll 惩罚会压制
   这种侧倾，可能限制过弯速度或导致不稳。建议：**转弯命令下放宽 roll 惩罚**（允许侧倾随 v·ω 增大）。
2. **持续高速转弯的课程覆盖**：建议在课程里显式采样"高速 + 持续转弯"组合，并适当延长命令重采样时间，
   让策略学会稳定地长时间过弯。

部署侧：过弯时需要给一个**稳定的偏航角速度命令 `wz`**（开环按弯道给，或用航向/路径控制器闭环算 `wz`
喂给策略）——策略本身只跟踪 `wz`，不感知绝对航向。

> 需要的话可以帮你把上面 1/2 两点（转弯放宽 roll + 高速转弯课程）也加到训练配置里。

---

## 七、起跑加速变体：stand-to-run 专家轨迹（并行训练）

任务 `Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-Accel`。在正式 AMP+HIM 任务（含上面第五节的
收紧偏航奖励）之上，**额外加一条「起跑」专家轨迹**来塑造从站立到冲刺的加速段。**和正式任务完全
独立、可同时训练**，正式任务的配置一行没动。

### 它和正式任务的唯一区别

1. **第 4 个 AMP style（id=3，"accel/launch"）**：判别器正样本来自 stand-to-run 专家片段
   `asset_zoo/robots/ultra_game_yaw/amp_motions/accel_stand_to_run.txt`（由 GMR 的
   `output_ultra_game_yaw_stand_to_run.pkl` 转出，**已裁掉前 1.8s 的原地站立**，
   保留 71 帧 / 2.38s / 29.8fps 的真实"起步→加速到约 5m/s"段；末端约 4.7m/s）。
2. **按 episode 时间触发**：style 调度换成 `ultra_style_update_accel`——被命令"跑"（`|cmd_vx|≥1.05`）
   的 env，在 episode 前 `accel_window_s=2.4s` 内被标成 style 3，**窗口后回落到 run(2)**。
   因为 AMP 正样本是**按 episode 时间采样**的，而每个 episode 都从站姿 reset 起步，所以专家片段
   （第 0 帧已是起步动作→末帧加速）会和机器人自己的起跑过程**时间对齐**，判别器正好监督"起步→加速→跑起来"。
   注意片段峰值只到约 5.9m/s，无法监督 8m/s 以上的冲刺姿态——它只负责教**站立→起跑**的摆臂/迈步启动。
3. **奖励掩码扩展**：凡是覆盖 run(2) 的奖励（速度跟踪、直行偏航抑制、正则等）都同时覆盖 accel(3)，
   所以起跑窗口里照样有完整的速度跟踪 + 直行偏航惩罚，外加 accel 的 AMP 风格奖励。
   `amp_reward_coef["style_3"]=0.8`。

### 训练命令

```bash
cd /home/ps/mjlab
ULTRA_MOTOR_DR=1 uv run train Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-Accel \
  --env.scene.num-envs 4096 \
  --agent.run-name accel-startfix \
  --agent.max-iterations 20000
```

日志落在 `logs/rsl_rl/ultra_game_yaw_amp_him_accel/<时间戳>_<run-name>/`（和正式任务的
`ultra_game_yaw_amp_him/` 分开，互不覆盖）。续训 / play / 导出 ONNX 的命令把任务名换成
`-Accel` 即可，其余与第二~四节一致。

### 重新生成专家轨迹（如果换了 GMR 片段）

txt 由脚本从 GMR pkl 转出（38 列 = 13 关节角 + 13 关节角速度 + 4 末端在根坐标系下的位置；
全部是根相对量，所以片段在世界系里的朝向/位置/高度都无所谓）：

```bash
cd /home/ps/mjlab
uv run python scripts/convert_gmr_pkl_to_amp_txt.py \
  --src /home/ps/GMR/output_ultra_game_yaw_stand_to_run.pkl \
  --dst src/mjlab/asset_zoo/robots/ultra_game_yaw/amp_motions/accel_stand_to_run.txt \
  --loop Clamp --weight 1.0 --start-frame 50
```

> 关节列顺序、末端 body、手部偏移必须和 `amp_him.py` 的 `get_amp_obs_for_expert_trans` 保持一致，
> 脚本里已对齐（GMR dof 顺序→AMP 顺序、`(0,0,-0.4)` 手部偏移、肩/踝末端）。`Clamp` 表示一次性
> 片段（起跑窗口 ≈ 片段长度，正好不需要循环）。
> **`--start-frame 50`**：原片段前 ~1.8s 是原地站立，而 AMP 正样本按 episode 时间采样，
> 不裁的话会在起跑窗口前段告诉判别器"原地不动才对"，和速度跟踪奖励直接打架（表现为小碎步蹭），
> 所以从动作真正起步处（第 50 帧）开始截取。

---

## 八、V4：渐进式课程 + 站立稳定（"站不住"的修复）

任务 `Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V4`（`env_cfgs_v4.py`）。背景：把关节顺序
对齐后，**Isaac 训出来的策略在 mjlab 里能稳稳站住**，而 mjlab 自己训的策略站立会碎步/漂移。
PD、机器人、物理、整套奖励都是忠实移植（Isaac checkpoint 能零样本迁移就是证据），所以差异在
**训练动态**——主因是课程：baseline 用**阶跃**速度课程（到固定 step 就把上限往上跳，不看学得好不好），
Isaac 用**性能门控的渐进式**课程。V4 在 baseline AMP-HIM 之上只改下面几处，**PD 和专家轨迹都保持
最原始**（与 Isaac 一致，**不是** V3 的软 PD）。

### 它和正式任务的区别

1. **渐进式速度课程**（`ultra_mdp.commands_vel_progressive` 替换阶跃 `command_vel`）：
   前进速度上限只有当 **run 风格(style 2)** 的 env **同时**满足"平均 episode 长度 > 0.8·max"
   且"平均 `track_lin_vel_x` 的 exp 奖励 > 速度相关门槛（随上限 5→15 从 0.6 线性降到 0.4）"时，
   才 **+0.2 m/s**（EMA 平滑、每个 episode 窗口评估一次，封顶 15）。忠实复刻 Isaac
   `update_command_curriculum`。起步 `(-0.5, 1.5)`。
2. **heading 对齐 Isaac**：`rel_heading_envs 0.85→0.6`、`heading_control_stiffness 1.0→0.5`
   （第五节为高速跑偏调到 0.85/1.0，会让策略更"爱扭"、渗到站立晃动；V4 回到 Isaac 的 0.6/0.5
   让偏航命令分布一致）。
3. **站立稳定双保险**（Isaac 没有，mjlab 这边额外加，pro-stability）：
   - `feet_slide` 的 `style_mask` 由 `[1,2]` 扩到 `[0,1,2]`——站立时脚打滑/碎步也被惩罚；
   - 新增 `stand_base_vel`（仅 style 0，权重 −2.0，见 `_STAND_BASE_VEL_WEIGHT`）——对本体水平
     速度做 L2 惩罚，直接压"乱走/漂移"。

### 训练命令

```bash
cd /home/ps/mjlab
ULTRA_MOTOR_DR=1 uv run train Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V4 \
  --env.scene.num-envs 4096 \
  --agent.run-name v4-progressive-standfix \
  --agent.max-iterations 20000
```

日志落在 `logs/rsl_rl/ultra_game_yaw_amp_him_v4/<时间戳>_<run-name>/`。续训 / play / 导出 ONNX
把任务名换成 `-V4` 即可（其余同第二~四节）。TensorBoard 关注 `Curriculum/command_vel/`
（`lin_vel_x_max`、`ema_len_ratio`、`ema_track`、`advanced`）确认速度是**平滑爬升**而非阶跃。

### 调参点（先按默认训，不行再动）

- `_STAND_BASE_VEL_WEIGHT`（现 −2.0）：站立还飘就加到 −4~−5。
- 课程 `update_step`/`len_ratio`：想更稳就把 `update_step` 调到 0.1、`len_ratio` 提到 0.85。

> **部署对齐（重要）**：V4 把 `heading_control_stiffness` 设回 0.5。等 V4 真正部署时，要把
> `ultra2026_rl_sdk/policy/loco_mode/config/LocoMode.yaml` 的 `heading_kp` 也从 1.0 改回 **0.5**，
> 否则部署 heading-hold 喂的 `wz` 会超出训练分布。**现在别改**（会破坏在用的 v2/v3 策略），选定 V4 再同步。

---

## 九、V5：V4 站立修复 + HoST smooth loss（不降 kp/kd 止抖）

任务 `Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V5`（`env_cfgs_v5.py`）。

**背景**：同事的 HIM 策略在**基线（Isaac）PD** 下上真机就不抖——他们没降 kp/kd，而是加了
**网络级的 smooth loss + 正则**。V5 直接移植 **HoST 库**（`HoST/rsl_rl/.../algorithms/ppo.py`）
里的那套平滑损失。这跟我们 reward 里的 `action_rate_l2` 是两种机制：`action_rate_l2` 只看已发生
轨迹上相邻动作之差、经 advantage 间接回传、梯度很弱；HoST smooth loss 直接约束策略/价值函数本身
的平滑性，梯度强，能治"观测噪声 → 动作高频抖"。降 kp/kd 是从硬件侧砍增益（副作用：响应钝、跟踪
差），smooth loss 是从策略侧让输出本身就平滑，所以能**保持正常 PD 还不抖**。

### HoST smooth loss 是什么（V5 采用的就是它）

不是"空间噪声 + 时间差分"两项分开，而是**一项插值式平滑**，且**同时平滑 policy 和 value**：

1. 沿"相邻步方向"做带随机幅度的扰动（`cont = 1 − done`，episode 边界处 `cont=0` 不扰动）：

```python
mix_w  = cont * (rand - 0.5) * 2.0        # 落在 cont * [-1, 1]
s_mix  = s_t + mix_w * (s_{t+1} - s_t)
```

2. 惩罚扰动带来的 policy 均值 / value 变化：

```python
L = c_pi * ‖μ(s_t) − μ(s_mix)‖² + c_V * ‖V(s_t) − V(s_mix)‖²
```

3. 系数用 HoST 的 bound 公式：

```python
epsilon = lower / (upper - lower)
c_pi    = upper * epsilon
c_V     = value_smoothness_coef * c_pi
```

**HoST 的 PPO 里没有独立的动作 L2 正则**——这个 smooth loss（policy + value 一起）本身就是"正则"。

### V5 = V4 + HoST smooth loss

V5 **完整继承 V4 的一切**（所以它同时治"站不住/碎步漂移"和"高频抖"）：

- 最原始 Isaac PD（**不是** V2/V3 的软 PD）、最原始专家轨迹（stand/walk/run）；
- Isaac 风格**渐进式课程**（性能门控 +0.2 m/s）替代阶跃课程；
- 站立稳定项：`feet_slide` 扩到 stand style（`style_mask=[0,1,2]`）+ `stand_base_vel` 本体水平
  速度 L2 惩罚（仅 style 0，权重 −2.0）；
- Isaac 对齐的 heading 命令（`rel_heading_envs=0.6`、`heading_control_stiffness=0.5`）。

**V5 相比 V4 唯一的新增**：在 PPO update（`multi_amp_ppo.py`）里打开 HoST smooth loss。

### 相比基线/V4 的代码改动

改动落在共享算法 `third_party/amp_rsl_rl/algorithms/multi_amp_ppo.py`，**默认 `lower_bound=0.0`
→ 系数 0（关闭）**，所以基线 / Accel / V2 / V3 / V4 行为完全不变；只有 V5 的 runner cfg 打开。
V5 默认系数对齐 HoST 的 G1 ground 配置：

| 参数 | V5 取值 | 说明 |
|---|---|---|
| `smoothness_upper_bound` | `1.0` | bound 公式上界 |
| `smoothness_lower_bound` | `0.1` | 越大平滑越狠；设 0 即关闭 |
| `value_smoothness_coef` | `0.1` | value 平滑相对 policy 平滑的权重 |

实现要点：连续帧对 `(s_t, s_{t+1})` 直接从 rollout buffer 取（`observations[:-1]/[1:]`、
`cont = 1 − dones[:-1]`），跟 HoST storage 的做法一致；**不动 storage / runner / 共享 minibatch
生成器**（避免连带 `ppo.py`/`amp_ppo.py` 崩）。因为 mjlab 是非对称 obs（critic 带特权信息），
value 平滑用同一个 per-sample 扰动权重作用在 critic-obs 帧对上。HIM estimator 在 `no_grad` 下，
policy 平滑只作用在 actor 映射。日志里会出现 `Loss/smooth_policy`、`Loss/smooth_value`、
`Loss/action_smoothness`（最后这个是只读诊断量：实际相邻步动作差 `‖μ(s_t) − μ(s_{t+1})‖`）。

### 训练命令

```bash
cd /home/ps/mjlab
ULTRA_MOTOR_DR=1 uv run train Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V5 \
  --env.scene.num-envs 4096 \
  --agent.run-name v5-host-smooth \
  --agent.max-iterations 20000
```

日志落在 `logs/rsl_rl/ultra_game_yaw_amp_him_v5/<时间戳>_v5-host-smooth/`，WandB project 是
`ultra_game_yaw_amp_him_v5`。续训 / play / 导出 ONNX 把任务名换成 `-V5` 即可（其余同第二~四节）。
TensorBoard 同时关注 V4 那套课程曲线（`Curriculum/command_vel/`）、`Loss/smooth_*` 和
`Loss/action_smoothness`（动作越来越平滑应逐步下降）。

### 调参点（先按默认训，不行再动）

- 还抖：把 `_SMOOTHNESS_LOWER_BOUND` 往上加（0.15~0.3，平滑更狠）；HoST 的 G1 wall 用到
  `value_smoothness_coef=0.25`，value 端想更稳可跟进。
- 跟踪变钝 / 反应慢：把 `_SMOOTHNESS_LOWER_BOUND` 调小（趋近 0 即关闭平滑）。
- 站立还飘：同 V4，`_STAND_BASE_VEL_WEIGHT` 加到 −4~−5。
- 太慢（smooth loss 每个 minibatch 多几次前向）：把 `_SMOOTHNESS_LOWER_BOUND` 调小或先关 value
  平滑（`value_smoothness_coef=0`）。

### 对照关系

- **V5 vs V4**：隔离"在站立修复之上，HoST smooth loss 还能额外带来什么"。
- **V5 vs V2/V3**：隔离"用 smooth loss 代替降 kp/kd"——V5 保持原始 PD。
- **V5 → V6**：V5 仍原地站不住、且想换上手调阻尼比 PD + 阶梯式课程，见第十节。

> **部署对齐**：smooth loss 只在训练期生效，导出的 ONNX 就是个更平滑的 actor，**部署不用加任何
> 东西、PD 不用改**。heading 耦合同 V4：V5 真要部署时，把 `LocoMode.yaml` 的 `heading_kp` 设成
> **0.5** 以匹配 `heading_control_stiffness`。**现在别改**（会破坏在用的策略），选定 V5 再同步。

---

## 十、V6：V5 + 手调阻尼比 PD + 阶梯式课程（站立 shaping 升级）

任务 `Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V6`（`env_cfgs_v6.py`）。

V6 在 V5 这条线（V4 站立修复 + HoST smooth loss）上**只动三件事**，其余（最原始专家轨迹、HIM、
HoST smooth loss 及其系数、`feet_slide` 扩到 stand style、Isaac heading 0.6/0.5）全部继承：

1. **PD：换成手调的 per-joint kp/kd**（按阻尼比调的，不是 V3 的全局 0.5·kp / 2·kd）。每个关节的
   effort 上限、力矩-转速包络（x1/x2/y1/y2）、命令延迟、摩擦都**保持基线**（`ultra_constants`），
   只替换 stiffness/damping：

   | 关节组 | kp | kd |
   |---|---|---|
   | `hip_yaw_*` | 400 | 20 |
   | `hip_roll_*` | 600 | 40 |
   | `hip_pitch_*` / `knee_pitch_*`（同一 actuator 组，必须同值） | 600 | 40 |
   | `ankle_pitch_*` | 100 | 10 |
   | `waist_yaw` | 200 | 10 |
   | `shoulder_pitch_*` | 40 | 4 |

2. **课程：阶梯式（STEP）替回渐进式**。把 V4/V5 的性能门控渐进课程 pop 掉，换回基线 `commands_vel`
   阶跃表（速度按训练轮数到点跳档）。

3. **站立 shaping 升级（治 V5 原地站不住 + 躯干乱晃）**，两处：
   - **分段式站立比例**：把"派去站住（零命令）的环境比例" `rel_standing_envs` 做成**随课程分段**——
     课程**第一阶段 0.5**（一半环境先练站桩），从**第二阶段起回落到基线 0.1**。复用阶梯课程的 step
     表实现（见下）。
   - **命令门控的整机晃动惩罚**：把 V4 的 `stand_base_vel`（仅 stand AMP style、只罚水平线速度）
     **换成 `stand_base_motion`**——当**命令 ≈ 0**（`‖cmd‖ < 0.1`）时，同时罚**基座水平线速度（world
     xy）＋基座角速度（body 三轴）**。V5 站着时躯干来回晃/前后摇，本质是基座角速度没被压住；线速度
     版罚不到这部分，所以躯干一直晃。新项让"浮动基座在零命令下真正静止"。
   - **命令门控的全关节速度惩罚**：再加 `stand_joint_vel`——同样在**命令 ≈ 0**时罚**所有关节的关节
     速度** `‖q̇‖²`（**对全部 13 个关节求和，ankle_pitch 在内**）。`stand_base_motion` 只压住了浮动
     基座，膝/肘/踝等肢体仍会原地抖；站着本就该"全身不动"，所以把每个关节的速度也罚掉。两项都只
     在零命令时触发，**不影响行走/奔跑**。
     > 注意：第一版 V6 run（`stand_base_motion=-2`、`stand_joint_vel=-0.02`）躯干仍前后摇、脚踝
     > pitch 没压住——**不是漏了 ankle**（求和本就含全关节），是**权重太轻 + 才训 1800 代欠训练**。
     > 现已把权重加重到 `-5`（角速度项 2×）/ `-0.1`。

4. **加狠关节限位 + 力矩饱和惩罚（治 5 m/s 力矩饱和）**：V6 的高 kd 让阻尼力矩 `kd·q̇` 随关节
   速度变大，5 m/s 时腿部关节速度高 → 力矩冲破速度相关的力矩包络 → 饱和。两手抓：
   - **加重限位惩罚**：`joint_pos_limits` −10 → **−30**（全关节）、`ankle_pitch_pos_limits_neg`
     −60 → **−150**。
   - **新增力矩饱和惩罚** `torque_saturation`（`ultra_mdp.torque_saturation_neg`）：复刻
     `UltraDelayedPdActuator._clip_effort` 那条**速度/方向相关的真实力矩天花板**，对**施加力矩超过
     `margin`(=0.85)×天花板**的部分按平方惩罚。施加力矩本就被夹在天花板内，所以这一项**恰好在关节
     饱和时最大、在包络内时为 0**，是个直接的"别去要你给不出的力矩"信号。
     > **根因是 kd 太高**（这项只治标）。若它把最高速度压得太死，**优先降 kd**（改 `JOINT_DAMPING`），
     > 别一味靠惩罚硬扛。

### 实现要点

- **分段站立比例**：扩展了共享课程 `mdp.commands_vel`（`curriculums.py`）——`VelocityStage` 多了可选
  键 `rel_standing_envs`，命中该阶段时同步改 `cfg.rel_standing_envs`（每次 resample 即生效）。
  V6 把阶梯表 stage0 设 0.5、stage1 设 0.1（后续阶段不带该键 → 维持 0.1），并把命令的初始
  `rel_standing_envs` 也设 0.5（课程第一次跑前就对齐）。**默认不带该键的任务行为不变。**
- **晃动惩罚**：新增 `ultra_mdp.stand_base_motion_l2`，按 `‖cmd‖<cmd_threshold` 门控，罚
  `‖v_xy‖² + ang_vel_scale·‖ω_body‖²`（V6 权重 **−5.0**、`ang_vel_scale=2.0`）；并新增
  `ultra_mdp.stand_joint_vel_l2`，同样门控、罚 `‖q̇‖²`（**全关节求和，含 ankle_pitch**，V6 权重 **−0.1**）。
  在 V6 里 pop 掉继承自 V4 的 `stand_base_vel`，加上 `stand_base_motion` + `stand_joint_vel`。
- **力矩饱和惩罚**：新增 `ultra_mdp.torque_saturation_neg`（含懒加载缓存 `_ensure_torque_envelope_cache`，
  扫一遍 `asset.actuators` 把每关节的 `y1/y2/x1/x2/force_limit` 散射成 joint-ordered 张量），在 reward 里
  按 `_clip_effort` 同样的公式重算速度/方向相关的力矩天花板 `max_eff`，罚 `(|τ|−margin·max_eff)₊²`。
  V6 权重 −6e-4、`margin=0.85`。同时把 `joint_pos_limits` 调到 −30、`ankle_pitch_pos_limits_neg` 调到 −150。
- PD 用 `dataclasses.replace(act, stiffness=…, damping=…)` 逐个 actuator 改写，断言同一 actuator
  组内各关节 kp/kd 一致（`hip_pitch`/`knee_pitch` 共组，已都设 600/40），再注入
  `cfg.scene.entities["robot"]`。
- 课程仅在**训练**时替换（`play=True` 时基线 env 本就清空课程，不动）。
- runner 完全沿用 V5（HoST smooth loss 开着、系数同 V5），只换 `experiment_name` /
  `wandb_project` 为 `ultra_game_yaw_amp_him_v6`，日志与 V5 互不覆盖。

### 训练命令

```bash
cd /home/ps/mjlab
ULTRA_MOTOR_DR=1 uv run train Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V6 \
  --env.scene.num-envs 4096 \
  --agent.run-name v6-retuned-pd \
  --agent.max-iterations 20000
```

日志落在 `logs/rsl_rl/ultra_game_yaw_amp_him_v6/<时间戳>_v6-retuned-pd/`，WandB project 是
`ultra_game_yaw_amp_him_v6`。续训 / play / 导出 ONNX 把任务名换成 `-V6` 即可（其余同第二~四节）。

### 调参点（先按默认训，不行再动）

- **躯干还晃（前后摇）**：`_STAND_BASE_MOTION_ANG_SCALE`（现 2.0）再加到 3~4，或
  `_STAND_BASE_MOTION_WEIGHT`（现 −5.0）整体加重；门控可在 `stand_base_motion_l2` 的
  `cmd_threshold`（现 0.1）上调。
- **脚踝/膝/肘等关节还抖**：`_STAND_JOINT_VEL_WEIGHT`（现 −0.1）加到 −0.2；但**最关键的是把训练跑满
  20000 代**——1800 代时站立远没收敛，权重再大也压不住欠训练的抖。若站姿被压得太"僵"、起步迟钝则调小。
- **还往前飘/迈步**：第一阶段站立比例 `_REL_STANDING_FIRST_STAGE`（现 0.5）提到 0.6~0.7，或抬回落值
  `_REL_STANDING_LATER`（现 0.1）。
- **5 m/s 还力矩饱和**：先加 `_TORQUE_SATURATION_WEIGHT`（现 −6e-4）到 −1e-3，并把 `_TORQUE_SATURATION_MARGIN`
  （现 0.85）降到 0.8（更早开始罚）；**但根因是高 kd**，治本是把 `JOINT_DAMPING` 里 hip/knee 的 40 往
  下调（如 20~30），阻尼力矩 `kd·q̇` 在高速时才不会顶满包络。限位继续加狠就调
  `_JOINT_POS_LIMITS_WEIGHT`（现 −30）、`_ANKLE_PITCH_LIMITS_WEIGHT`（现 −150）。
- **起跑不连贯**：训练侧站立项都按 `‖cmd‖<0.1` 门控，**不会**渗进低速行走；起跑顿挫主要来自
  ①欠训练（1800/20000），②低速命令和 AMP walk 专家步频不匹配。先训满代；部署侧已做缓冲
  （`max_lin_accel 5→1.5`、`stand_settle_s 0.6→1.0`，见 `ultra2026_rl_sdk` 的 `LocoMode.yaml`）。
  想要更顺可加 accel 那条 stand-to-run 专家轨迹（第七节）。
- 想再平滑：同 V5，加 `_SMOOTHNESS_LOWER_BOUND`。
- PD：直接改 `JOINT_STIFFNESS` / `JOINT_DAMPING` 两张表（注意 `hip_pitch`/`knee_pitch` 必须同值）。

### 对照关系

- **V6 vs V5**：同时换了 PD 和课程，**不是单变量消融**，是刻意的组合配置（手调 PD + 阶梯课程 +
  更狠的站立 shaping）。要做纯 PD/课程对照，把 `smoothness_lower_bound` 清零关掉 smooth loss。

> **部署对齐（重要）**：V6 的 kp/kd 是手调值，**选定 V6 部署时必须把 `ultra2026_rl_sdk` 的
> env / `LocoMode.yaml` PD 增益同步成上表**，否则 sim2real 不一致。heading 耦合同 V4
> （部署 `heading_kp` 设 0.5）。**现在别改**部署配置（会破坏在用的策略），选定 V6 再同步。
