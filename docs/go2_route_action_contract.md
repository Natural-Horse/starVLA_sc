# Go2 分路动作协议

## 动作与状态

模型动作头和状态条件统一为 10 维容器：

```text
[dx_body, dy_body, dyaw,
 tcp_x_base, tcp_y_base, tcp_z_base,
 roll_base, pitch_base, yaw_base, gripper]
```

这不是让导航和机械臂执行相同动作。`NAV` 的维度 mask 为前三维有效；`GRASP`、`PLACE` 的维度 mask 为后七维有效。FM loss 同时应用 `[T]` 时间 mask 和 `[10]` 维度 mask，其他维度不产生梯度。

- NAV action：当前机体系下的稀疏 `[dx,dy,dyaw]` waypoint chunk。
- ARM action：base frame 下的 TCP `[x,y,z,roll,pitch,yaw]` 和 `[0,1]` 夹爪目标 chunk。
- 默认不输入机器人 state（`datasets.router_data.include_state=false`），动作头只以视觉、指令与路由为条件。
- 若显式开启 `include_state=true`：NAV state 前三维为 `[vx_body,vy_body,wz_body]`；ARM state 后七维为当前 TCP 与夹爪状态，离线数据使用上一帧已执行目标作为当前控制状态代理。

## Route 与 subtask

每个采样帧按 `task_stage` 得到 route：`nav_to_* -> NAV`、`pick -> GRASP`、`place -> PLACE`。全局 instruction 由 episode 主任务和两个八方向组成：

- box1：从机器狗初始位姿到 box1 的方向，取 `nav_to_pick` 的第一条规范转向 instruction。
- box2：从抓取完成后起始位姿到 box2 的方向，取 `nav_to_place` 的第一条规范转向 instruction。
- 方向仅允许 `front/front-right/right/back-right/back/back-left/left/front-left`；缺失时拒绝加载 episode。

VLM 的 subtask GT 是当帧 Parquet `instruction` 的完整局部指令，不是六类分段标签。`nav_straight/nav_turn/nav_stop/arm_approach/arm_contact/arm_retreat` 仍保留为 `phase_label`，用于数据分析和采样，不作为 VLM 文本目标。输出格式为：

```text
<|route|><|subtask|>local_instruction<|end_subtask|>
```

主任务仍由每个样本的 `task_index` 查询 `meta/tasks.jsonl`，不会固定使用第一条任务描述。

现有 n200 的 `tasks.jsonl` 尚未包含逐 episode 方位，loader 会从该 episode 两段导航的首条转向 instruction 动态构造全局描述。新版本 `pct_scene` 会直接把相同描述写入随机化任务；loader 会校验其与局部 instruction 一致并避免重复拼接。因此无需原地改写 30200 帧 Parquet。

## 三阶段训练

1. `vlm_instruction`：全 route，从已有 VLM checkpoint 微调 Qwen3-VL 的 route/局部 instruction 生成，冻结动作头。
2. `action`：只取 NAV，冻结 VLM，训练动作头前三维。
3. `manip`：只取 GRASP/PLACE，冻结 VLM，训练动作头后七维。

必须先在 held-out episode 上通过 route 与局部 instruction 生成式评测，再把该 VLM checkpoint 交给 `action`；旧六类 phase-label VLM checkpoint 不能直接作为新 action 阶段的冻结条件模型。

10 维动作头与旧 3 维动作头权重形状不兼容。可以通过 `reload_modules=qwen_vl_interface` 只迁移已训练的 VLM，并重新初始化 10 维 action head；不能完整加载旧 NAV 动作 checkpoint。后续 GRASP/PLACE 阶段再完整继承同为 10 维的 NAV checkpoint。
