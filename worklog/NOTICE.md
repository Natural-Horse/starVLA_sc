# 项目工作约定

每次工作前将工作记录写入日志，工作完成后将计划转为工作记录并压缩，保持日志文件精炼且准确。所有日志和说明文档使用中文；每天只写 `worklog/YYYY-MM-DD.md`，结束时刷新精简的 `## 全日总结`。

## 固定目录与分支

- 本机训练仓库：`/home/natural/Desktop/mtr/starVLA_sc`
- GitHub：`github.com/Natural-Horse/starVLA_sc.git`
- 训练分支：`robodog`
- 本地数采仓库：`/home/natural/Desktop/mtr/pct_scene`
- 数采分支：`mtr_dev`
- 数采 Fork：`git@github.com:Natural-Horse/arm-vla-grasp-sim.git`（`origin`）
- 数采上游：`https://github.com/yagami-light7/arm-vla-grasp-sim.git`（`upstream`）
- 本地采集输出：`/home/natural/pct_scene_outputs`
- 本地真机仓库：`/home/natural/Desktop/mtr/gx-real`
- 真机分支：`mtr_dev`
- 真机 Fork：`https://github.com/Natural-Horse/gx-real.git`（`origin`）
- 真机上游：`https://github.com/lemonoscar/gx-real.git`（`upstream`）

## 远程服务器

服务器连接参数以 `/home/natural/.ssh/config` 为准。启动训练、数据处理、服务或 smoke test 前，必须由用户当次确认使用哪台服务器。

- `zju-server`：训练仓库 `/hdd4/MaTianran/pct_workspace/starVLA_sc`，分支 `robodog`。
- `robotdiff_new`：工作目录 `/diff/wallx_workspace/`；该服务器可能无法访问 GitHub，应在本地完成修改和验证后用 `rsync` 同步所需代码、数据或权重。

所有训练必须使用命名明确的 tmux session，并在同一 session 中启动对应 TensorBoard。训练前检查 GPU 占用和进程归属，不得停止、修改或占用他人的进程。

## 仿真与真机合同

仿真闭环测评中，`starVLA_sc` 负责模型加载、版本化远程协议和 route/subtask/机体系稀疏 waypoint 输出；`pct_scene` 负责观测编码、世界系变换、DWA/RL 速度适配、Isaac 状态机及 cuRobo 抓放。远端服务只绑定 `127.0.0.1`，由评测机通过 SSH 本地转发访问，不直接暴露公网。

三仓库统一动作协议为 10 维 `[dx_body,dy_body,dyaw,tcp_x_base,tcp_y_base,tcp_z_base,roll_base,pitch_base,yaw_base,gripper]`。NAV 只监督前三维，GRASP/PLACE 只监督后七维。机械臂输出是机体系 TCP 目标，仿真由 `pct_scene` 本地规划执行，真机由 `gx-real` 本地完成标定、IK/规划、安全限幅、看门狗和 CAN 写入；远端模型不得直接下发关节或 CAN 命令。

工作时保留用户已有改动。先在本地修改和验证，再提交到对应分支；用户确认服务器后，按该服务器的同步方式更新远端并运行。
