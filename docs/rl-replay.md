# 浏览器强化学习回放

## 使用和部署

入口为顶部“强化学习回放”，默认选择 PhysX WASM，依次选择任务、权重来源、“加载策略”和 Play。可显式切换 MuJoCo 进行迁移对照。

- 内置权重：15 份原有训练项目 checkpoint、1 份 IsaacLab 3.0 周期行走最佳结果和 45 份旧 Isaac Gym 迁移权重位于 `public/rl/checkpoints/`，与网站一起部署。
- 本地文件：通过浏览器文件选择器选 `.pt` 或 `.pth`；不上传，也不访问原训练目录。
- 网络网址：下载后在浏览器推理；跨站服务需允许 CORS。文件上限 128 MiB。
- 暂停保持状态，单步执行一个 1/30 秒控制周期，重置恢复模型初始关键帧并清空动作历史与积分量。Play 持续推理，直到手动暂停、重置、切换任务或关闭回放。
- 静止画面不持续重绘。切换到后台标签页时挂起计算，返回后继续；策略控制和物理步长不随绘制帧率改变。
- 关闭回放释放独立 PhysX 场景与 MuJoCo 显示模型，原构型实验台保持原状态。步态控制和末端控制不会覆盖策略动作。

运行时只有静态资源请求，没有 `/api`、Python、Isaac Lab 或 GPU 后端。`npm run build` 后部署完整 `dist/` 到 HTTP(S) 网站根目录；不要直接双击 `index.html` 使用 `file://`。无需第三方 CDN 或跨域隔离头。经用户授权增加固定版本 `physx-js-webidl@2.7.3`；其 JS/WASM 由 Vite 打包为本地静态资源，首次选择 PhysX 时才加载。

## 支持范围

| 任务 ID（除最后一项外前缀均为 `Zbot-Direct-`） | checkpoint | 输入→输出 | 模型 | 验收 |
| --- | --- | --- | --- | --- |
| 8dof-bipedal-v0 | model_4050.pt | 30→8 | zbot_8s_human | 运动通过 |
| 8dof-snake-v0 | model_12050.pt | 30→8 | zbot_8s_snake_v0 | 运动通过 |
| 6dof-bipedal-quat-v0 | model_3850.pt | 26→6 | zbot_6s_new | 运动通过 |
| 6dof-bipedal-to-snake-v0 | model_1000.pt | 26→6 | zbot_6s_new | 运动通过 |
| 6dof-bipedal-to-snake-v1 | model_3400.pt | 26→6 | zbot_6s_new | 运动通过 |
| 6dof-bipedal-v0 | model_8350.pt | 24→6 | zbot_6s_new | 运动通过 |
| 8dof-bipedal-v1 | model_4850.pt | 30→8 | zbot_8s_human_v1 | 运动通过 |
| 8dof-bipedal-v2 | model_1550.pt | 30→8 | zbot_8s_human_v2 | 运动通过 |
| 8dof-bipedal-v3 | model_1700.pt | 30→8 | zbot_8s_human_v3 | 实验：原生也失稳 |
| 8dof-bird-v0 | model_13900.pt | 30→8 | zbot_8s_bird | 运动通过 |
| 8dof-wheel-v0 | model_800.pt | 30→8 | zbot_8s_wheel | 运动通过 |
| 6dof-bipedal-velocity-v0 | model_latest.pt | 33→6 | zbot_6s_new | 实验：部分命令失稳 |
| 8dof-bipedal-velocity-v0 | model_latest.pt | 39→8 | zbot_8s_human_v1 | 实验：部分命令失稳 |
| 6dof-bipedal-velocity-imu-v0 | model_latest.pt | 34→6 | zbot_6s_new | 运动通过 |
| 8dof-bipedal-run-v0 | model_latest.pt | 40→8 | zbot_8s_run | 运动通过 |
| ZbotRlIsaaclab-6DOF-Periodic-Walking | model_1499.pt | 45→6 | zbot_6s_new | 浏览器 PhysX 在 1.5 Hz 下连续 20 秒行走 |

另有 45 组 `IsaacGym-*` 迁移策略，完整数据表由 [legacyProfiles.ts](../src/rl/legacyProfiles.ts) 定义。每个目录均带 `provenance.json`，记录源 checkpoint、源/导出 SHA256 和网络维度。归档中存在 45 个顶层 `.pth`，不是最初目测的 46 个。

批量重新生成时必须显式确认输入是可信的本地 PyTorch pickle：`python scripts/export-legacy-isaacgym.py --trusted /path/to/pth public/rl/checkpoints`。不要对下载或来源不明的 `.pth` 使用转换脚本；浏览器运行时仍只读取转换后的受限数据格式。

每个文件位于对应任务子目录。`exported/policy.pt` 为同一策略的 TorchScript 导出，已核对 15 组 actor 参数一致（跑步归一化统计量也一致），没有作为新网络重复收录。浏览器文件选择应使用原始 `model_*.pt/.pth`。

原有 15 组的完整文件大小、SHA256、网络维度及原生限制见 [网络清单](../results/rl-catalog/catalog.json)。新增周期行走策略的来源与校验值见 [模型记录](../public/rl/checkpoints/ZbotRlIsaaclab-6DOF-Periodic-Walking/provenance.json)。

解析器支持默认 `torch.save` ZIP、受限的数据型 pickle，以及历史 `model_state_dict.actor.*`、新版 `actor_state_dict.mlp.*`、学生 `student_state_dict.mlp.*` 前馈网络。跑步网络按保存的统计量执行 Float32 `(obs-mean)/(std+0.01)`；旧 rl_games 样本执行其原始 `(obs-mean)/sqrt(var+1e-5)` 并裁剪到 ±5。保留 pickle 白名单、20 万操作与张量大小限制，不执行 pickle 全局函数；未知层、循环网络、损坏或非有限数据会被拒绝。

State dict 不包含完整任务或激活函数信息，同维度不代表同观测语义。自定义权重必须来自所选任务的相同网络与训练配置。当前目录 60 组均有显式网络、观测和动作契约；其中缺少唯一同名任务源码的历史权重会在界面标明按同维度任务族恢复。三组现有实验权重原样保留，45 组旧 Isaac Gym 权重统一标为迁移诊断，数值完成不算运动验收通过。

## 观测与动作

姿态和速度主要取机器人中部 **base** 刚体。速度控制和 IMU 任务的投影重力则取自由根 **foot_0**，不能混用。世界四元数采用 wxyz；角速度使用 base 的世界角速度传感器。MuJoCo 自由关节 qvel 的角速度在根的局部坐标系，不能直接替代。参见 [MuJoCo 坐标约定](https://mujoco.readthedocs.io/en/stable/overview.html#floating-objects) 和 [传感器定义](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-frameangvel)。

- 24/30 维：世界角速度 z、机体投影重力 xyz、heading、相对默认关节角、关节速度、上一 tanh 动作、关节速度上限。双足和蛇形的 heading 分别按源任务公式计算。
- 26 维：base 世界四元数 wxyz、世界角速度 xyz、相对默认关节角、关节速度、上一 tanh 动作、关节速度上限。
- 33/39 维速度策略：滤波机身线速度、滤波角速度、根投影重力、三个速度命令、三个步频/相位项、关节误差、关节速度、上一动作。34 维 IMU 策略用 base 四元数替代线速度。滤波时间常数 0.08 秒，每控制步只更新一次；相位固定为 1.2 Hz、初相位 0，以便复现。
- 40 维跑步策略：任务坐标系线速度/角速度、base 重力、heading、高度、**绝对关节角**、关节速度、上一动作、双足接触状态/腾空时间及速度上限。接触用真实冲量计算的法向力，保留 5 个物理步历史。原生传感器顺序是 **foot_1、foot_0**，与关节树相反，模型中显式记录。
- 45 维 IsaacLab 周期行走策略：base 局部线／角速度与重力、6 个相对关节角与关节速度、6 个**原始**上一动作、足端 `foot_0/foot_1` 在 base 坐标系中的法向力、全身 COM 相对双足的位置、频率／支撑脚／正弦余弦相位、支撑距离误差与 heading 误差。使用训练时的 200 Hz 物理、50 Hz 速度积分位置目标和 ±π/2 积分限幅；频率可在 0.5–2.0 Hz 调整。浏览器默认 1.5 Hz；1.0 Hz 可能跌倒，不能据此判断原生训练结果失效。
- 36 维旧 Isaac Gym 策略：关节速度参数、base 世界位置/朝向 z 轴/线速度/角速度、全机质心、质心到两端距离、归一化关节角/速度和两端接触力。网络为 36→256→128→64→6 ELU；确定性动作按 rl_games 规则裁剪到 ±1，再以 60 Hz 积分到关节目标。
- 每个旧任务独立恢复 reset 时的关节角、根位置和根四元数。rolling 使用专用 7 刚体模型并从贴地姿态开始；`ZBotBipedalWalking`、`_20250514`、`_20250527` 共用专用 `ZBot_R1` 七刚体模型并从约 0.254 m 的站立 base 高度开始。两类模型都采用原始 MJCF 碰撞网格、原生质量/COM/惯量和 0.0166 s 求解周期。
- rolling 观测读取刚体原点而不是 COM，并复刻旧 VecTask 首步重置及 CPG 计时归零。6DOF BipedalWalking 复刻 36 维观测、速度上限 0.5、直连速度动作、前两步连续 reset，以及 base 跌破 0.2 m/两端距离过近后的延迟一帧自动 reset；三份权重分别保留各自的 bootstrap 观测，因为这些观测已经受对应网络首步动作影响。
- 24/26 维的 6DOF 回放可在界面输入关节速度参数（0.1–5，默认 2）；该值同时写入策略观测并用于动作积分。速度命令/IMU 策略的观测不包含该字段，因此不显示此输入。跑步使用任务保存的观测值与积分值。
- 基础动作：`a=tanh(actor(obs))`；`delta=clamp(delta+π×2×a/30, -π, π)`；`target=qDefault+delta`。轮式 delta 限幅为 ±π/2；跑步额外 action_scale=1.2；8DOF 速度目标限制在**参考关节角±π/2**，不是绝对±π/2。
- 速度面板支持前进/侧移 ±0.4 m/s、转向 ±1 rad/s，采用源码的手动命令覆盖方式；默认前进 0.2 m/s。四组扩展观测任务暂仅提供 PhysX，MuJoCo 对照选项禁用。
- 默认 PhysX：PD 刚度 50、阻尼 5，TGS 位置迭代 4、速度迭代 0，物理 60 Hz，每个 30 Hz 控制周期两步。它按原 PhysX 隐式驱动和接触约束求解。MuJoCo 对照仍使用 600 Hz，每周期 20 子步；数值稳定不代表策略能稳定行走。时间异常、警告或非有限状态会停止回放。

## 模型来源和再生成

`public/rl/models/` 中 MJCF 包含全部网格，无外部 mesh 引用。`manifest.json` 记录源 USD SHA256、初始状态、关节顺序、刚体数和质量。模型离线从训练项目 USD 提取，无需启动 Isaac Sim 或 GPU。可在已安装 pxr 的开发环境再生成：

```bash
TERM=xterm /home/yhzhu/isaaclab/isaaclab.sh -p scripts/export-rl-models.py --help
```

这个脚本仅用于维护模型资产，不参与前端运行。训练源码依据还包括 `task_cards/zbot_8dof_bipedal_run_v0.py`、`velocity_env.py`、`velocity_commands.py` 和机器人配置。六组旧双足/鸟形任务的紧凑 24/30 维观测从原始 `zbot_rl` 源码恢复：对应 checkpoint 和 USD 与 student 副本逐字节一致。仅修复捕获工具的历史观测列表，不修改训练工程。

PhysX 模型另位于 `public/rl/physx/`。`scripts/capture-isaac-reference.py` 在开发机记录实际运行状态，`scripts/isaac_cooked_colliders.py` 提取实际 PhysX 烘焙凸包，`scripts/export-physx-models.py --reference-dir /tmp` 将参考捕获和 USD 转为静态模型数据。导出文件包含来源校验和，运行时不需要任何 Python 文件。

默认模式中，PhysX 独立执行全部动力学；MuJoCo 只按 PhysX 根位姿、关节位置与速度做正运动学来显示网格，不再二次步进。特别处理根刚体 COM 线速度到坐标原点速度、世界角速度到 MuJoCo 局部角速度的转换。

不执行训练任务的接触终止/自动重置规则。蛇形任务主要奖励沿自身身体方向运动，没有固定世界航向奖励；原生 CPU、GPU 与浏览器的长期转向轨迹也可能不同。此次目标是恢复任务行为和双足持续行走，不宣称逐帧或每个自定义 checkpoint 的效果相同。

## 验证

运行 `npm run validate:rl-motion` 可用 PhysX 验证全部 61 组策略，并对可用任务运行 MuJoCo 对照，生成 [比较结果](../results/sim-to-sim/validation.json)。现有 16 组运行 20 秒；45 组旧 Isaac Gym 迁移策略运行 2 秒数值与动作冒烟测试。它不需要原训练工程；可选 `-- --reference-dir /tmp` 附加原生参考捕获摘要。迁移策略不以当前近似模型的运动质量作为通过条件。

使用当前生产代码和原始 checkpoint 的结果：

| 任务 | MuJoCo 全程样本中站立比例 | PhysX 站立比例 | PhysX 20 秒前进距离 |
| --- | --- | --- | --- |
| 8DOF 双足 v0 | 3.17% | 100% | 9.64 m |
| 6DOF quaternion 双足 | 5.00% | 100% | 3.91 m |

站立阈值取原任务的机身最低高度 0.30 / 0.22 m。蛇形和形态转换有独立贴地/位移判据，不使用双足站立阈值。这里报告的距离是对应任务方向上的位移；全部数值及资产校验和见结果文件。

`npm test` 覆盖安全解析、MLP 数值、11 类旧观测拼接、三类旧动作控制器、关节映射、模型质量与传感器、全部 60 个内置权重的 WASM 回放和确定性重置。`tests/physxReplay.test.ts` 进一步要求已有双足全程站立并有效前进、蛇形持续贴地移动、变形任务降为低姿态，并检查所有迁移权重有限运行、重复加载/重置/释放和显示同步。旧 Isaac Gym 36→6 样本另与原始 rl_games PyTorch CPU 推理核对，固定观测的最大误差低于 `5e-6`。

rolling 的逐步诊断可运行 `npm run compare:legacy-rolling`：它将浏览器 PhysX 的观测、网络动作和积分目标与 `tests/fixtures/rollingNative.json` 中的原生 Isaac Gym trace 分别报告最大误差与 RMS。首两次策略动作及 CPG 目标要求数值一致；后续刚体速度仍会受到 Isaac Gym 旧 GPU PhysX 与当前 WASM PhysX 版本、管线和接触求解差异影响，不把跨版本轨迹逐帧相同作为通过条件。

6DOF 双足逐步诊断使用 `npm run compare:legacy-bipedal`；追加 `-- ZBotBipedalWalking_20250514` 或 `-- ZBotBipedalWalking_20250527` 可检查另外两份权重。三者的初始位姿、前三次网络动作和关节目标均由 `results/legacy-bipedal-walking/` 中的原生 GPU capture 回归；第 3 步后的完整轨迹仍按跨版本诊断处理。

旧工程 `run.sh` 的 10 个唯一回放命令均已建立专项映射：Rolling、SideMoving、ForwardMoving、StandUp、FootDown、SingleLeg、SnakeAndBipedal、BipedalWalking、BigFoot 6DOF 和 BigFoot 8DOF。`ZBot.xml`、`ZBot_R1.xml` 任务复用各自准确的七刚体资产；BigFoot 两项使用独立导出的七/九刚体模型及端部圆盘碰撞。原生启动 trace 位于 `results/run-sh/`，运行 `npm run compare:run-sh` 可批量比较观测、动作和位置目标。

这些映射同时恢复任务自己的默认关节角、关节符号、速度上限、CPG 相位时钟、连续启动 reset 和可表达的跌倒终止规则。原生 GPU PhysX 与 WASM PhysX 在启动 trace 之后仍可能产生接触轨迹分叉，因此验收分别报告网络/控制契约精度与有限回放，不用一个“数值完成”掩盖运动差异。

SingleLeg 的物理侧使用独立 `legacy_zbot_single_leg` 模型，base 明确对应源任务的 `body_2`；显示侧没有特殊模型，复用其他 6DOF 任务的 `zbot_6s_new`，通过显式关节名和方向映射同步。由于当前 PhysX 5 reduced-coordinate force drive 比归档 GPU PhysX 的位置驱动明显偏软，该任务使用经原生 trace 标定的等效 stiffness/damping；120 步原生/Workbench 位移约为 `1.026/0.937 m`，base 高度范围约为 `0.197–0.262/0.195–0.285 m`。验证器因此为 SingleLeg 增加最低高度和位移运动门槛。

旧 MJCF 为同一模块同时定义了 `visual_*` 与 `coliision_*` 重合网格。PhysX 回放的显示 XML 会移除 group 4 碰撞副本，只绘制 visual 网格；碰撞仍由独立 PhysX hull 保留。测试会编译所有 run.sh 显示模型并拒绝任何残留的 group 4 geom，避免画面出现深浅两套网格重叠。

主要实现：`src/rl/checkpoint.ts`、`src/rl/profiles.ts`、`src/rl/replay.ts`、`src/rl/physx.ts`、`src/rl/physxRuntime.ts`、`src/components/RlReplayPanel.tsx`。

## sim-to-sim 修正依据

初始观测与网络输出相符、整个运动中的刚体坐标也与原生捕获一致；质量、COM 和惯量同样匹配。差异在执行器与接触的离散求解：PhysX 的 TGS 隐式关节驱动和接触共同迭代，MuJoCo 的同名增益与更细子步不能自动复制其响应。此前仅验证有限状态和可重复性，未覆盖行走稳定性。

自由空间使用顺序隐式驱动可以显著缩小首步误差，但恢复地面接触后仍无法持续站立；接触、阻尼及速度缩放也未通过运动验收，因此这些试验参数没有写入默认模型。用户选择采用 PhysX CPU WASM 后，使用原权重和原控制定义恢复了双足持续行走。

求解器依据：[PhysX 隐式驱动推导](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_downloads/6acf3afb8f69452757e0e766b5a22978/implicitDrives.pdf)、[PhysX WASM 上游](https://github.com/fabmax/physx-js-webidl)。第三方许可随静态资源发布于 `rl/physx/THIRD_PARTY_NOTICES.txt`。

## 历史权重表现边界

原生 CPU 参考确认：双足 v3 在当前及历史初始朝向下都倒地；6/8DOF 速度权重在固定前进 0.2 m/s、1.2 Hz、初相位 0 的测试中也可能失稳。它们在界面标为实验，未替换权重或用自动重置隐藏表现。其余 12 组通过对应站立、位移、贴地或形态转换判据。命令跟踪精度和逐帧轨迹不作保证。

新增权重和机器人按选择加载，不在启动时批量初始化，保持暂停零重绘、后台挂起以及随机器人移动的阴影。
