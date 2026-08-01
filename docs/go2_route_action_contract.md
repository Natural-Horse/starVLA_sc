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
- NAV state：前三维为 `[vx_body,vy_body,wz_body]`。
- ARM state：后七维为当前 TCP 与夹爪状态；离线数据使用上一帧已执行目标作为当前控制状态代理。

## Route 与 subtask

每个采样帧按 `task_stage` 得到 route：`nav_to_* -> NAV`、`pick -> GRASP`、`place -> PLACE`。VLM 标签直接使用该帧真实 `subtask`：导航为 `nav_straight/nav_turn/nav_stop`，机械臂为 `arm_approach/arm_contact/arm_retreat`。输出格式为：

```text
<|route|><|subtask|>subtask_label<|end_subtask|>
```

主任务 instruction 由每个样本自己的 `task_index` 查询 `meta/tasks.jsonl`，不会固定使用第一条任务描述。

## 三阶段训练

1. `vlm`：全 route，训练 Qwen3-VL 的 route/subtask 文本，冻结动作头。
2. `action`：只取 NAV，冻结 VLM，训练动作头前三维。
3. `manip`：只取 GRASP/PLACE，冻结 VLM，训练动作头后七维。

10 维动作头与旧 3 维动作头权重形状不兼容。可以通过 `reload_modules=qwen_vl_interface` 只迁移已训练的 VLM，并重新初始化 10 维 action head；不能完整加载旧 NAV 动作 checkpoint。后续 GRASP/PLACE 阶段再完整继承同为 10 维的 NAV checkpoint。
