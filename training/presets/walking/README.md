# 六自由度已有行走策略微调预设

任务 ID：`Mjlab-Zbot-6dof-Walking-Finetune`。此预设从已训练的本地权重继续优化，不是随机初始化训练。

`model_801.pt` 为相邻 `zbot_rl_mjlab` 工程的 `2026-09-09_exact-ppo-noentropy` 检查点原样副本；来源、迭代和 SHA-256 见 `provenance.json`，训练配置见 `config.json`。每次首次启动此预设都会校验文件哈希，随后恢复 actor、critic、优化器及课程状态。网页自己的后续检查点可续训；任意外部检查点仍不可作为续训输入。

使用检查点的完整环境/PPO 配置：240 Hz 物理、30 Hz 控制、第二阶段行走奖励、0.8–1.2 积分速度参数范围（动作积分还乘以 π）、固定学习率 1e-5、熵系数 0。只覆盖用户选择的环境数量、随机种子、保存间隔和新增更新次数。迭代编号从原检查点后延续。

网络为 `QuatFeatureMLP`：外部26维四元数观测，内部追加投影重力3维和航向误差1维，送入30维MLP。`training/tasks/zbot_rl_mjlab/models.py` 是源工程原样副本。结果包标记 `policyFeatures=quat-gravity-heading-v1`，浏览器执行同样的确定性特征变换，无教师网络或 Python 运行依赖。

继续优化不保证步态质量单调提升；应对导出策略验证完整20秒行走后再作为新基线。
