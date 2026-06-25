# Ultra GameYaw —— 训练 / Play 命令速查

13-DoF 双足机器人 `ultra_game_yaw` 的速度跑步策略（PPO + Multi-AMP(WGAN) + HIM-GRU + 对称镜像损失）。
本文是**命令速查**；训练原理、奖励细节、高速偏航优化、起跑加速专家轨迹等详见
[`src/mjlab/tasks/velocity/config/ultra_game_yaw/README.md`](src/mjlab/tasks/velocity/config/ultra_game_yaw/README.md)。

> 所有命令都在 `cd /home/ps/mjlab` 下、用 **`uv run`**（不要直接用 `python`）。

## 任务名（gym id）

| 任务 | 说明 |
|---|---|
| `Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM` | **正式任务**：AMP+HIM+对称，含高速直线偏航优化。 |
| `Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-Accel` | **起跑加速变体**：正式任务 + 一条 stand-to-run 专家轨迹（第 4 个 AMP style）专门优化起跑加速段。与正式任务互不影响、可并行训练。 |
| `Mjlab-Velocity-Flat-Ultra-GameYaw` | 纯 PPO，仅作对照消融。 |

日志落在 `logs/rsl_rl/<experiment>/<时间戳>_<run-name>/`，两个 AMP 任务的 `experiment` 分别是
`ultra_game_yaw_amp_him` 和 `ultra_game_yaw_amp_him_accel`，互不覆盖。

---

## 一、从头训练

正式任务：

```bash
cd /home/ps/mjlab
ULTRA_MOTOR_DR=1 uv run train Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM \
  --env.scene.num-envs 4096 \
  --agent.run-name tn-yaw-fix \
  --agent.max-iterations 20000
```

起跑加速变体（命令一样，只换任务名）：

```bash
cd /home/ps/mjlab
ULTRA_MOTOR_DR=1 uv run train Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-Accel \
  --env.scene.num-envs 4096 \
  --agent.run-name accel-startfix \
  --agent.max-iterations 20000
```

- `ULTRA_MOTOR_DR=1`：**打开电机域随机化**（per-joint 输出力矩 scale 0.7~1.2 + PD 增益 scale 0.7~1.2，
  startup 模式跨 episode 持久）。这是默认值，写出来只为明确；设 `ULTRA_MOTOR_DR=0` 关闭，play 时自动关闭。
- `--env.scene.num-envs`：并行环境数，按显存调整（4096 常用，显存不足就调小）。
- `--agent.run-name`：日志子目录后缀。
- 速度课程自动从 1.5 m/s 逐级升到 15 m/s（见 `env_cfgs.py` 的 `command_vel` 阶段表）。
- 单轮耗时参考：正式 ~3.8s、accel ~4.7s（多一个判别器），20000 轮 ≈ 21~26H。
  **别同时开 deploy sim2sim**，它会抢 CPU 把 collection 拖慢数倍。

### 续训（接某个 checkpoint）

```bash
cd /home/ps/mjlab
ULTRA_MOTOR_DR=1 uv run train Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM \
  --env.scene.num-envs 4096 \
  --agent.resume True \
  --agent.load-run 2026-06-05_14-40-37_tn-yaw-fix \
  --agent.load-checkpoint model_19999.pt
```

> `--agent.resume` 是布尔参数，**必须显式写 `--agent.resume True`**，裸写会报
> `invalid choice ... Expected one of ('True', 'False')`。续训 accel 变体时把任务名换成
> `-Accel`、`--agent.load-run` 换成 `ultra_game_yaw_amp_him_accel/` 下的对应 run。

> 续训会自动对齐 `common_step_counter`，让速度课程从对应阶段继续，而不是回到第一阶段。

---

## 二、Play（可视化 / 手动遥控）

> ⚠️ **任务名 ↔ checkpoint 目录必须配套**（两个任务的 AMP style 数不同，3 vs 4，加载会校验判别器）：
>
> | checkpoint 在这个目录 | 必须用这个任务名 |
> |---|---|
> | `logs/rsl_rl/ultra_game_yaw_amp_him/...` | `Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM` |
> | `logs/rsl_rl/ultra_game_yaw_amp_him_accel/...` | `Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-Accel` |
>
> 配错会报 `KeyError: 'style_3'`（用 baseline 任务名去加载 accel checkpoint）或类似的判别器键不匹配。

```bash
cd /home/ps/mjlab
uv run play Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM \
  --checkpoint-file logs/rsl_rl/ultra_game_yaw_amp_him/<RUN>/model_XXXX.pt \
  --num-envs 1
```

accel 变体把任务名换成 `-Accel`、checkpoint 路径换成 `ultra_game_yaw_amp_him_accel/<RUN>/...` 即可：

```bash
cd /home/ps/mjlab
uv run play Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-Accel \
  --checkpoint-file logs/rsl_rl/ultra_game_yaw_amp_him_accel/<RUN>/model_XXXX.pt \
  --num-envs 1
```

### 带键盘遥控的 Play（推荐）

键盘遥控**必须配 `--viewer native`**（键位绑在原生查看器上），并加 `--keyboard-cmd True`：

```bash
cd /home/ps/mjlab


```

> ⚠️ `--keyboard-cmd` / `--no-terminations` / `--video` 都是布尔参数，**必须显式写 `True`**
> （例如 `--keyboard-cmd True`）；裸写会报
> `invalid choice ... Expected one of ('True', 'False')`。
>
> accel checkpoint 同理：任务名换 `-Accel`、路径换 `ultra_game_yaw_amp_him_accel/<RUN>/...`。例如：
>
> ```bash
> uv run play Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-Accel \
>   --checkpoint-file logs/rsl_rl/ultra_game_yaw_amp_him_accel/<RUN>/model_XXXX.pt \
>   --num-envs 1 --keyboard-cmd True --viewer native --no-terminations True
> ```

### 直接起跑到指定速度（看起跑加速姿态）

加 `--init-vx 8.0`：进入即把前向命令锁到 8 m/s，机器人从站姿直接起跑加速到 8 m/s，正好观察
起跑加速段姿态（之后 `W/S` 仍可实时增减）。需配 `--keyboard-cmd True`：

```bash
cd /home/ps/mjlab
uv run play Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-Accel \
  --checkpoint-file logs/rsl_rl/ultra_game_yaw_amp_him_accel/<RUN>/model_XXXX.pt \
  --num-envs 1 \
  --keyboard-cmd True \
  --init-vx 8.0 \
  --viewer native \
  --no-terminations True
```

> 也有 `--init-vy`（横移）、`--init-wz`（偏航角速度）。机器人每次 episode 复位都从站姿起步，
> 命令立即是 8 m/s，所以能反复看起跑加速过程。

**键位**（焦点在原生查看器窗口时按）：

| 键 | 作用 |
|---|---|
| `W` / `S` | 前进速度 +/− |
| `Q` / `E` | 偏航 左/右（原地转向） |
| `Z` / `C` | 横移 左/右 |
| `X` | 所有速度命令清零 |

`--keyboard-cmd` 会关闭随机命令重采样，并在终端实时打印当前速度。

### 带手柄遥控的 Play（Xbox / XInput）

手柄遥控加 `--joystick-cmd True`（**两种 viewer 都支持**，后台线程读手柄）。可调满杆前进速度
`--joy-vx-max`（默认 5.0 m/s）。可与 `--keyboard-cmd True` 同时开（手柄连上时手柄优先）。
实测可用命令：

```bash
cd /home/ps/mjlab
uv run play Mjlab-Velocity-Flat-Ultra-GameYaw-AMP-HIM-V6 \
  --checkpoint-file logs/rsl_rl/ultra_game_yaw_amp_him_v6/<RUN>/model_XXXX.pt \
  --num-envs 1 \
  --joystick-cmd True \
  --joy-vx-max 5.0 \
  --viewer native \
  --no-terminations True
```

> ⚠️ `--joystick-cmd` 是布尔参数，**必须写 `--joystick-cmd True`**（写成 `ture` 或裸写都会报
> `invalid choice ... Expected one of ('True', 'False')`）。

**手柄键位**（标准 Xbox/XInput 布局）：

| 操作 | 手柄 |
|---|---|
| **前进/后退 vx（一下一档 ±0.5，持续保持，类比键盘 W/S）** | **十字键 上/下** |
| **转向 wz（类比键盘 A/D）** | **十字键 左/右** |
| 前进/后退 vx（比例，推多少给多少） | 左摇杆 上/下 |
| 左右平移 vy（比例） | 左摇杆 左/右 |
| 转向 wz（比例） | 右摇杆 左/右 |
| 满速前进（= `--joy-vx-max`） | `RB` |
| 速度清零 | `Y` |

> 想要"像键盘一样一点一点加速"就用**十字键**（D-pad）：按一下 ↑ 加 0.5 m/s 并保持，不用一直推摇杆。
> 摇杆是比例控制，只在推出死区时覆盖对应轴、回中则保持十字键设定的命令。注：`A/B/X` 在 play 里
> **不做事**（play 没有 FSM）；它们只在 SDK 真·部署里是 行走/起立/Passive。

> **无线手柄会休眠**：空闲几分钟自动断连（`/dev/input/js0` 消失），启动时若手柄睡着会提示
> “armed but no pad detected yet … will auto-connect”，**按一下手柄唤醒**后后台会自动重连开始遥控，
> 无需重启。手柄需处于 **XInput 模式**。需要 `pygame`（已装）。轴/键对不上可用环境变量改映射，
> 见 SDK 侧 `common/joystick_command.py` 顶部注释（`ULTRA_JOY_AXIS_*` / `ULTRA_JOY_BTN_*`）。

### 其它常用开关

- `--no-terminations True`：禁用摔倒终止，便于持续观察。
- `--video True`：录制视频到 `logs/.../videos/play`。
- `--viewer native|viser|auto`：选择查看器（默认 auto；键盘遥控用 `native`，手柄两种都行）。
- `--joystick-cmd True` / `--joy-vx-max 5.0`：Xbox 手柄遥控（见上节）。
- 注：上面带 `True` 的都是布尔参数，必须显式给值。

---

## 三、TensorBoard

```bash
cd /home/ps/mjlab
uv run tensorboard --logdir logs/rsl_rl
```

---

## 四、导出 ONNX → 部署 / sim2sim

```bash
cd /home/ps/mjlab
uv run python /home/ps/ultra2026_rl_sdk/tools/export_mjlab_onnx.py \
  --checkpoint logs/rsl_rl/ultra_game_yaw_amp_him/<RUN>/model_XXXX.pt
```

导出的 ONNX 已是 ISAAC 关节顺序。部署 / sim2sim 的完整说明见
`/home/ps/ultra2026_rl_sdk/DEPLOY_MJLAB_README.md`。
