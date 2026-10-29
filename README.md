# MimicLite：862、977、5292 版本说明

本仓库记录 G1 CHIP 控制器的三个训练版本：**862、977、5292**。这些数字是 Lambda Slurm 作业编号，**不是训练步数**。这里的“步”指 PPO 更新次数，不是仿真步数。

- **862**：三点位姿＋下肢关节命令，保留全身 link 跟踪，是本次对比的基线。
- **977**：仍是三点＋下肢关节；放松上肢 link 跟踪，改变右手柔顺采样并提高旋转柔顺上限。
- **5292**：五点位姿命令，脚部刚性跟踪；恢复轻量上肢跟踪，并加强 torso。四卡从头训练到 **12000** 次更新，已导出到 Robofarm。

本文只对比这三个版本。通用项目介绍见 [上游说明](docs/UPSTREAM_README.md)，环境安装见 [训练项目](mimic-lite/README.md) 与 [训练框架](active-adaptation/README.md)。

## 1. 版本与模型对应关系

| 版本 | 本文核对的 checkpoint | 控制接口 | 配置依据 |
| --- | --- | --- | --- |
| 862 | `lambda-862-11500/checkpoint_11500.pt` | 上肢三点＋12 个腿关节 q/dq | [训练配置快照](docs/versions/862.yaml) |
| 977 | `lambda-977-24500/checkpoint_24500.pt`；另有 11500 步用于同期比较 | 上肢三点＋12 个腿关节 q/dq | [训练配置快照](docs/versions/977.yaml) |
| 5292 | `lambda-5292-12000/checkpoint_12000.pt` | 全身五点的位置与姿态 | [训练配置快照](docs/versions/5292.yaml) |

模型文件保存在本机 `checkpoints/` 下，不随 Git 提交。862、977 的配置中 `total_iters=30000` 是训练预算，不能用它推断某个已下载 checkpoint 的实际步数。5292 的预算与最终 checkpoint 均为 12000。

配置快照是各模型原始 `cfg.yaml` 的副本，保留历史路径；它们用于审计参数，不能不改路径就直接在另一台机器训练。当前 `chip_6d.yaml` 是可调基础配置，并不等同于历史 862 或 977。

## 2. 相同的部分与输入输出

三版均使用原生 MimicLite PPO、G1 29 关节控制、MJLab 后端；控制频率 50 Hz，物理仿真频率 200 Hz。历史状态输入为 10 帧 × 93 维 = **930 维**：每帧重力方向 3、角速度 3、关节位置 29、关节速度 29、上一动作 29。输出均为 **29 维关节动作**，经部署配置的动作缩放与 PD 控制执行。

| 项目 | 862 / 977 | 5292 |
| --- | --- | --- |
| 控制点顺序 | 左手、右手、torso | 左手、右手、torso、左脚、右脚 |
| 参考下肢 q/dq | 12＋12 维 | 从 actor 命令移除 |
| 点位置 / 四元数 | 9＋12 维 | 15＋20 维 |
| anchor 朝向 | 6 维 | 6 维 |
| 上肢三点柔顺参数 | 18 维 | 18 维 |
| command 总维数 | **69** | **59** |
| actor 总输入 | **999 = 930＋69** | **989 = 930＋59** |
| 输出动作 | 29 | 29 |

因此三点 checkpoint 不能只替换文件就装进五点输入网络，也不能从三点权重直接按原形状 resume。脚部命令从关节表达变成位姿表达，并不意味着删除机器人自身的腿部关节反馈；930 维状态历史仍包含所有关节。

三版使用 pelvis 作为参考 anchor。手与 torso 控制点相对所属 link 的偏移依次为 `[0.18,-0.025,0]`、`[0.0719,-0.003,0]`、`[0,0,0.35]` 米；5292 两脚使用 ankle-roll link 原点，偏移均为零，不能直接当作足底中心。

部署 actor 接收当前目标，不需要未来目标轨迹，也不需要运行时 mocap。机器人状态来自 IMU/编码器，目标仍需由参考播放器、遥操作或上层策略提供。训练使用参考动作数据；训练 critic/reward 中的 `future_steps: [0,1]` 不等于部署 actor 需要未来帧。三版都是全身联合策略，并没有左右臂独立网络或硬隔离约束。

## 3. 862 → 977：不只是下半身不同

两者下半身命令接口完全相同，主要变化发生在**上肢监督和右手柔顺分布**。

### 全身 link 跟踪

四个 body 跟踪项的权重均为位置 `0.50`、姿态 `0.50`、线速度 `0.25`、角速度 `0.25`，sigma 分别为 `0.30 m / 0.40 rad / 1.0 m/s / 3.14 rad/s`。

- 862：上述项覆盖 pelvis、torso、双侧 hip-roll、knee、ankle-roll，**以及双侧 shoulder-roll、elbow、wrist-yaw**。
- 977：上述项只覆盖 pelvis、torso 和下肢 link，移除了六个上肢 link。

977 仍保留手部末端跟踪、pelvis 跟踪、下肢关节跟踪与正则项，不能理解为“完全没有全身约束”。但手部位姿相近时，对肩肘构型的直接参考约束变弱，这可能影响冗余自由度的稳定性。

### 右手柔顺参数与采样

三版均为 `wrist_6d`，左手刚性、torso 柔顺为零；5292 的两脚也保持刚性。柔顺平移单位为 m/N，旋转单位为 rad/Nm。

| 参数 | 862 | 977 | 5292 |
| --- | --- | --- | --- |
| 平移柔顺上限 xyz | `[.02,.02,.02]` | 同左 | 同左 |
| 旋转柔顺上限 xyz | `[.8,.8,.8]` | `[1,1,1]` | `[1,1,1]` |
| actor 柔顺缩放 | `[50,50,50,1.25,1.25,1.25]` | `[50,50,50,1,1,1]` | 同 977 |
| 平移模式概率 | `[0,0,0,.75,.25,0]` | `[.25,.05,.05,.35,.25,.05]` | 同 977 |
| 旋转模式概率 | `[0,0,0,.75,.25,0]` | 同左 | 同左 |

模式顺序为 X-only、Y-only、Z-only、XYZ 同值、刚性、XYZ 独立采样。977 增加了各向异性平移采样，不只是改了一个最大值。三版的最大外力均为 20 N、最大纯力矩均为 4 Nm，角度 hindsight 上限均为 120°；不能把后两项误记为 977 独有变更。

## 4. 5292：五点＋轻量上肢跟踪＋torso 加强

实现入口为 [chip_five_point.py](mimic-lite/mimic_lite/tasks/chip_five_point.py)，配置为 [chip_five_point.yaml](mimic-lite/cfg/task/chip_five_point.yaml)。

### 下半身改为两脚位姿

actor 不再接收参考腿关节 q/dq，而接收左脚、右脚位置和姿态。新增脚位置奖励 `weight=1.0, sigma=0.10 m`，脚姿态奖励 `weight=0.5, sigma=0.40 rad`。原有训练时的下肢关节监督仍保留：位置 `0.5 / 0.25 rad`，速度 `0.5 / 2.5 rad/s`（格式为 weight / sigma）。这些监督不要求推理端提供参考腿关节。

### 恢复轻量上肢监督

保持 977 的下肢 body 四项权重，另外对双侧 shoulder-roll、elbow、wrist-yaw 增加独立奖励：

| 奖励 | weight | sigma |
| --- | ---: | ---: |
| `upper_body_pos` | 0.15 | 0.30 m |
| `upper_body_ori` | 0.15 | 0.40 rad |
| `upper_body_linvel` | 0.05 | 1.0 m/s |
| `upper_body_angvel` | 0.05 | 3.14 rad/s |

其目的，是在手末端跟踪之外给肩肘构型提供参考。它与 862 把上、下肢合并进同一个平均误差的设计不同，不能把权重直接相加后当作完全等价。

### torso 与左手共同项等权，三个专属项也开启

六个共享位置/姿态奖励的 `[左手,右手,torso]` 点权重从默认 **`[2,4,1]` 改为 `[2,4,2]`**。共享项本身的 weight/sigma 保持不变：

| 共享项 | weight | sigma |
| --- | ---: | ---: |
| `chip_pos` / `chip_pos_precision` / `chip_pos_fine` | 2 / 2 / 2 | 0.10 / 0.04 / 0.03 m |
| `chip_ori` / `chip_ori_precision` / `chip_ori_fine` | 0.5 / 0.25 / 0.25 | 0.40 / 0.15 / 0.10 rad |

点权重进入指数内部的加权均方误差：`exp(-weighted_mean(error²) / sigma²)`，并不是三个独立奖励的外部倍率。sigma 是奖励尺度，不是可保证的跟踪误差。

按照最终训练要求，**以下三个 torso 专属奖励没有关闭**，其点权重均为 `[0,0,1]`：

| 专属奖励 | weight | sigma |
| --- | ---: | ---: |
| `chip_torso_pos` | 0.50 | 0.05 m |
| `chip_torso_ori` | 0.75 | 0.30 rad |
| `chip_torso_ori_precision` | 0.50 | 0.10 rad |

因此只能说“torso 与左手在共享项内等权”，不能说两者最终总监督强度完全相同。原有右手独立位置项 `1.0 / 0.03 m` 也仍保留。新增 torso 误差指标用于日志，不额外计入奖励。

## 5. 训练与复现

5292 于 2026-10-27 提交，在四张 GPU 上从头训练，seed 42，每卡 16384 环境，总计 65536 环境；生产训练为 12000 PPO 更新，每 500 次保存。先进行了小规模 50 次和满容量 6 次更新检查，这些检查不计入生产 checkpoint 的 12000 次。W&B run ID 为 `7l6u5dwn`。

使用训练环境已安装、数据已准备好的 Lambda 工作目录，可通过统一五点入口启动相同目标配置：

```bash
cd /share/ml/yangmin/mimiclite-chip/active-adaptation
TOTAL_ITERS=12000 NUM_ENVS=16384 \
  sbatch projects/mimic-lite/scripts/train_chip_five_point_lambda_4gpu.slurm
```

这是新训练命令，不会恢复已有模型。通用 runner 的默认预算是 30000，因此要显式设置 `TOTAL_ITERS=12000`。历史 5292 使用隔离源码和框架副本；复现实验也应固定提交并核对最终生成的 `cfg.yaml`，不能仅凭脚本名称认定复现成功。

训练前确认：

1. `active-adaptation/projects/mimic-lite` 与 `.cache/projects.json` 指向本次要运行的源码；框架会根据该清单加载环境与算法。
2. `loco_manip_physical_rollout_accepted_v1` 数据及四个数据池的 split manifest 存在；`prepare_chip_loco.py --five-point` 生成本机有效的路径。
3. 新生成配置与 [5292 快照](docs/versions/5292.yaml) 的 task、reward、observation、algo、seed 一致；路径、运行目录、W&B 信息按环境调整。
4. 复现 862/977 时应以各自快照恢复参数，再重建数据路径。直接运行当前 `task=chip_6d` 不会自动恢复其历史采样分布与 body 权重。

三点与五点的 command 维度不同，训练脚本会拒绝不兼容 resume。修改奖励配置不会改变已有 checkpoint；必须重新训练或在相同输入契约下明确进行续训。

## 6. Robofarm 部署与统一入口

5292 已部署到 `/home/yixuan/Project/HIL_Dagger`：

- 模型包：`models/mimic_lite_chip_five_point_5292_12000/`
- 配置：`config/mimic_lite_five_point_5292_12000.yaml`
- 原始 checkpoint SHA256：`5b48d00f34820d41016bb4221292892d416430211d7d05404ec09ec784f718d8`

沿用已有 five-point 入口，无需为版本再写另一套推理脚本：

```bash
cd /home/yixuan/Project/HIL_Dagger
bash run_five_point_real.sh --check
bash run_five_point_sim.sh --config config/mimic_lite_five_point_5292_12000.yaml
bash run_five_point_real.sh --config config/mimic_lite_five_point_5292_12000.yaml --net ROBOT_INTERFACE
```

`ROBOT_INTERFACE` 替换为实际机器人网卡。`--check` 只检查模型与配置；最后一条才是实机控制入口。`FIVE_POINT_CONFIG` 可配置默认模型，显式 `--config` 优先。部署脚本属于 HIL_Dagger 仓库，不包含在本仓库的训练源码提交中。

必须配套使用该模型的 ONNX、观测归一化、contract、关节顺序、动作缩放和参考位姿。5292 的 five-point 初始化沿用所选参考首帧；GR00T 流程中的 `raise2cm` 是另一套参考配置，不能仅因启动脚本统一就假设初始姿态相同。862/977 是 69 维三点控制器，应使用三点适配器，不能塞进 59 维 five-point 配置。

## 7. 已验证与尚未证明的部分

已有 5292 部署记录确认：模型哈希与 Lambda 一致；CPU PyTorch/ONNX 三组输出最大绝对误差约 `2.15e-6 / 7.15e-7 / 1.43e-6`；部署与仿真控制测试 23 项通过。默认 wiping 参考的 30 秒仿真完成 1500 步，未跌倒，最大倾角约 7.82°，最低 pelvis 高度约 0.771 m。

该仿真有 25 帧超过 0.6 rad 跟随误差告警阈值，最大约 0.681 rad。它没有执行新的右臂外力扰动测试，**不能据此宣称“拉右手时左臂转圈”已修复，也不能据此保证 torso 实机误差**。

862、977、5292 不构成只改变一个变量的消融实验：上肢监督、柔顺采样、命令接口、torso 奖励与训练步数均有差别。后续判断效果应使用相同参考、相同初始位姿、相同右手外力/力矩、相同部署限速，比较左右臂关节峰值/速度、torso 位置与朝向误差以及跌倒情况。

本次提交另在暂存代码的独立目录上执行了 41 项 CHIP 测试和 4 项 torso/posture 测试，全部通过；Python 与 shell 语法检查通过。这些回归检查覆盖实现与输入契约，不替代受扰动仿真或实机验证。

## 8. 提交范围

本次保存三版相关的六维柔顺、五点命令、奖励、训练/导出支持与测试，并归档三份历史训练配置。不包含模型权重、缓存、完整训练日志，也不包含独立 wrist-joint、force-wall 实验。旧有其他项目说明仍保留，三版比较以本文和配置快照为准。
