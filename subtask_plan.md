# FlyGripper Subtask 模块计划

## 一、数据集原始格式

### 1.1 目录结构

```
{trajectory_id}/
├── {trajectory_id}-{phase}/
│   ├── README.txt                  # 阶段元信息
│   ├── data.csv                    # 时间戳位姿 + 图像引用
│   ├── target_points.csv           # 飞行阶段的目标位姿
│   └── images/ego/                 # FlyGripper 机载相机帧
│       └── flygripper_camera_XXXXX.jpg
```

**注意**：`trajectory_config.json` 是后期手动添加的标注文件，不属于原始数据。

### 1.2 原始文件格式

#### README.txt

记录每个阶段的元信息，格式为键值对：

```
记录编号: 1
记录时间: 2026年 04月 23日 星期四 15:29:38 CST
记录类型: yaw_only          # 仅在 yaw_only 阶段出现，flight 阶段无此字段
记录时长: 10                 # 仅在 yaw_only 阶段出现
```

- 阶段 1,3,5：`记录类型=yaw_only`，`记录时长=10`
- 阶段 2,4,6：无记录类型和时长字段（即 flight 阶段）

#### data.csv

```
时间戳(秒),位置X,位置Y,位置Z,姿态X,姿态Y,姿态Z,姿态W,FlyGripper机载固定摄像头图像
0.007,-0.587,-0.620,0.887,0.001942,-0.000259,0.618083,0.786111,flygripper_camera_00000.jpg
0.212,-0.581,-0.632,0.887,0.001580,-0.000210,0.618072,0.786120,flygripper_camera_00001.jpg
...
```

| 列 | 说明 |
|----|------|
| 时间戳(秒) | 距该阶段开始的秒数 |
| 位置X, Y, Z | 世界坐标系下的位置（米） |
| 姿态X, Y, Z, W | 四元数表示的朝向 |
| FlyGripper机载固定摄像头图像 | `images/ego/` 下的文件名 |

采样频率约 5Hz。每行即一个时刻的 6DOF 位姿 + 一张 ego-view 图像。

#### target_points.csv

```
目标点ID,时间戳(秒),位置X,位置Y,位置Z,姿态X,姿态Y,姿态Z,姿态W
```

- **flight 阶段（2,4,6）**：有目标点位姿（该阶段要飞到的终态位姿）
- **yaw_only 阶段（1,3,5）**：仅有表头，无数据行

### 1.3 任务结构

每条轨迹是一个 6 阶段的 `pick_fly_place` 任务：

| Phase | 类型 | 行为 | 持续时间 |
|-------|------|------|---------|
| 1 | yaw_only | 原地旋转，搜索抓取目标 | ~10s |
| 2 | flight | 飞向目标并抓取 | 变长 |
| 3 | yaw_only | 原地旋转，搜索中间路径点 | ~10s |
| 4 | flight | 飞向中间路径点 | 变长 |
| 5 | yaw_only | 原地旋转，搜索放置目标 | ~10s |
| 6 | flight | 飞向目标并放置 | 变长 |

### 1.4 原始数据中**没有**的信息

- **无** gripper state（夹爪开合状态）
- **无** 自然语言任务指令
- **无** subtask 标注
- **无** 动作类型标签（yaw_only / flight 标签仅在 README.txt 中）

---

## 二、Sub-task 原语与标注格式

### 2.1 Sub-task 原语词表

定义一组可复用的动作原语（Action Primitives），不同的 task_type 通过组合不同数量的原语构建完整任务。模型不自由生成文本，而是从有限词表中选择原语并填入槽位。

| 原语 | 格式 | 语义 |
|------|------|------|
| Search | `Search [Target] at [Direction]` | 原地旋转，搜索目标 |
| Fly to | `Fly to [Location]` | 飞向目标位置 |

当前任务只有这两个原语。可扩展：未来新任务类型可添加新原语（如 `Place`, `Hover`, `Scan` 等）。

**pick_fly_place 任务的 subtask 序列**（每条轨迹 = 6 个 subtask）：

| Subtask | 来源 Phase | 原语 | 填槽示例 |
|---------|-----------|------|---------|
| 1 | Phase 1 | Search | Search juice bottle at back left |
| 2 | Phase 2 | Fly to | Fly to side cabinet |
| 3 | Phase 3 | Search | Search kitchen at back left |
| 4 | Phase 4 | Fly to | Fly to kitchen |
| 5 | Phase 5 | Search | Search round table at back right |
| 6 | Phase 6 | Fly to | Fly to round table |

Phase 2 的抓取动作隐含在飞行中，Phase 6 的放置也隐含在飞行中，均不单独拆分。

### 2.2 轨迹级标注

每条轨迹一个配置文件，标注任务指令和原语序列：

```json
{
  "trajectory_id": 1,
  "instruction": "Pick up the juice bottle on the side cabinet on your back left, turn to your back left and fly to the kitchen, then turn to your back right and place the juice bottle on the round table with a chair next to it on your back right",
  "task_type": "pick_fly_place",
  "object": "juice bottle",
  "subtasks": [
    {
      "subtask_id": 1,
      "phase_id": 1,
      "primitive": "Search",
      "slots": {"target": "juice bottle", "direction": "back left"},
      "subtask_text": "Search juice bottle at back left",
      "type": "yaw_only"
    },
    {
      "subtask_id": 2,
      "phase_id": 2,
      "primitive": "Fly to",
      "slots": {"location": "side cabinet"},
      "subtask_text": "Fly to side cabinet",
      "type": "flight"
    },
    {
      "subtask_id": 3,
      "phase_id": 3,
      "primitive": "Search",
      "slots": {"target": "kitchen", "direction": "back left"},
      "subtask_text": "Search kitchen at back left",
      "type": "yaw_only"
    },
    {
      "subtask_id": 4,
      "phase_id": 4,
      "primitive": "Fly to",
      "slots": {"location": "kitchen"},
      "subtask_text": "Fly to kitchen",
      "type": "flight"
    },
    {
      "subtask_id": 5,
      "phase_id": 5,
      "primitive": "Search",
      "slots": {"target": "round table", "direction": "back right"},
      "subtask_text": "Search round table at back right",
      "type": "yaw_only"
    },
    {
      "subtask_id": 6,
      "phase_id": 6,
      "primitive": "Fly to",
      "slots": {"location": "round table"},
      "subtask_text": "Fly to round table",
      "type": "flight"
    }
  ]
}
```

### 2.3 帧级标注

在 parquet 基础上新增 phase_id、subtask_id、subtask_text 三列（由标注脚本生成，原始数据集无此三列）。

### 2.4 Subtask History（per-trajectory，跨 episode）

subtask_history 是**同一轨迹内**当前帧之前已完成的所有 subtask 文本，按顺序拼接。构建时从 `subtask_configs.json` 查询：`trajectory_id = episode_index // 2`，取 `subtask_id < 当前值` 的所有 subtask_text。

示例（Trajectory 0，instruction = Pick up drink bottle ... on stool）：

**Episode 0 (pick, subtask 1-2)**：

| 帧的 subtask_id | subtask_history | solution |
|---|---|---|
| 1 | `""` | `"Search drink bottle at back right"` |
| 2 | `"Search drink bottle at back right"` | `"Fly to desk"` |

**Episode 1 (place, subtask 3-6)**：

| 帧的 subtask_id | subtask_history | solution |
|---|---|---|
| 3 | `"Search drink bottle at back right. Fly to desk"` | `"Search kitchen at back"` |
| 4 | `"Search drink bottle at back right. Fly to desk. Search kitchen at back"` | `"Fly to kitchen"` |
| 5 | `"Search drink bottle at back right. Fly to desk. Search kitchen at back. Fly to kitchen"` | `"Search stool at left"` |
| 6 | `"Search drink bottle at back right. Fly to desk. Search kitchen at back. Fly to kitchen. Search stool at left"` | `"Fly to stool"` |

per-trajectory 的好处：推理时无人机连续执行，模型知道完整 history，能学到 subtask 之间的顺序关系。

---

## 三、Subtask 与现有 Router 的关系

### 3.1 现有 Router 机制分析

当前 router 的训练流程（`train_starvla_cotrain_router.py` + `WallXRouterDataset`）：

```
1. WallXRouterDataset.__getitem__() 返回:
   - lang: router_prompt_template 填充后的 prompt
   - solution: "<|pred_action|>" 或 "<|pred_bbox|><point>[...]</point>"
   - route: "action" 或 "bbox"
   - action: 归一化后的 action chunk（仅 action route）

2. VLARouterTrainer._train_step():
   - qwen_vl_interface(**batch_inputs) → VLM CE loss + hidden_states
   - model.action_loss_from_hidden_states(hidden_states, action_examples) → action loss
   - backward(vlm_loss + action_loss)
```

核心特征：
- **Framework 的 `forward()` 不被调用**，trainer 直接控制 VLM forward 和 action forward
- **VLM CE loss 驱动路由学习**，模型学习生成 `<|pred_action|>` 或 `<|pred_bbox|>` token
- **Action loss 用 VLM hidden states**，梯度回传到 backbone

### 3.2 Subtask 替代 Router 的设计思路

Subtask Head 的任务（生成 "Search/Fly to/Place" 原语）和 Router 的任务（生成 `<|pred_action|>` / `<|pred_bbox|>`）本质相同：**都是 VLM 文本生成，都通过 lm_head + CE loss 训练**。

因此最干净的方式是：**Subtask 直接替代 Router 的 VLM 输出目标**，不新建 Framework，而是在现有 router trainer 基础上改造。

| 对比 | 现有 Router | Subtask 替代方案 |
|------|-----------|----------------|
| VLM 输出 | `<\|pred_action\|>` 或 `<\|pred_bbox\|><point>[...]` | `"Search juice bottle at back left"` |
| CE loss 标签 | 1 个 route token | 完整 subtask_text（~5-8 tokens） |
| 路由判断 | 首个 token 决定 action/bbox | subtask 原语隐含行为类型 |
| Bbox 分支 | 有（pred_signal=stop 时） | 暂无（当前数据所有帧都是 action） |
| Action loss | hidden_states 直接传 action head | hidden_states detach 后传 action head（Knowledge Isolation） |

### 3.3 Bbox 分支的处理

当前数据中所有帧的 `pred_signal` 要么是 `<pred_action>` 要么是 `<stop>`。`<stop>` 帧用于 bbox 训练。

在 Subtask 模式下，bbox 的定位功能由 **Search 原语**隐含替代——模型输出 "Search X at Y" 等价于 "我需要看到 X 并定位它"，这个行为在 yaw_only 阶段通过视觉搜索实现，不需要显式 bbox 坐标输出。

当前阶段的设计：
- **所有帧统一为 action 路由**（包括原本 `<stop>` 的帧）
- 移除 bbox 分支，`<stop>` 帧也参与 action 训练
- 未来如需 bbox 输出（如 Place 阶段定位放置点），可在 Place 原语后追加 bbox token

---

## 四、训练架构设计

### 4.1 整体架构

```
                         ┌───────────────────────────────────────────────┐
                         │               Qwen3-VL Backbone               │
                         │                                               │
                         │  输入: [Ego Image]                            │
                         │       + [Subtask Prompt]                      │
                         │         (instruction + subtask_history)       │
                         │                                               │
                         │  输出: hidden_states (多层) + VLM CE loss      │
                         └──────────┬────────────────────┬───────────────┘
                                    │                    │
                          ✅ 梯度回传(VLM CE loss)      ❌ 梯度阻断(detach)
                                    │                    │
                                    │                    ▼
                                    │         ┌────────────────────────────┐
                                    │         │  LayerwiseFlowmatching      │
                                    │         │  ActionHead (DiT)          │
                                    │         │                            │
                                    │         │  输入: detached hidden list │
                                    │         │       + state              │
                                    │         │                            │
                                    │         │  输出: 6DOF waypoint       │
                                    │         │                            │
                                    │         │  Loss: FlowMatching loss   │
                                    │         │  ❌ 不更新 backbone        │
                                    │         └────────────────────────────┘
                                    │
                         VLM CE Loss = CE(model_output, subtask_text)
                         subtask_text 是完整原语文本，如:
                         "Search drink bottle at back right"
                         "Fly to desk"
```

### 4.2 与 Router 训练器的对应关系

```
Router Trainer (_train_step)         Subtask Trainer (_train_step)
────────────────────────────         ─────────────────────────────
_prepare_router_batch()              _prepare_subtask_batch()
  └─ prompt: router_prompt             └─ prompt: subtask_prompt
  └─ solution: route token             └─ solution: subtask_text
  └─ route: action/bbox                └─ route: 全部 action

qwen_vl_interface(**inputs)          qwen_vl_interface(**inputs)
  └─ VLM CE loss ✅                    └─ VLM CE loss ✅ (subtask_text)
  └─ hidden_states                     └─ hidden_states

action_loss_from_hidden_states()     action_loss_from_hidden_states()
  └─ hidden 直接传入                   └─ hidden.detach() 后传入
  └─ 仅 action_indices 的样本          └─ 所有样本

backward(vlm_loss + action_loss)     backward(vlm_loss + action_loss)
```

### 4.3 Knowledge Isolation 的实现

在 trainer 层而非 framework 层实现。关键改动只有一行：

```python
# 原来 (router trainer):
action_loss_raw, _, _ = self._compute_router_action_loss(
    qwen_output.hidden_states, batch_router, action_indices,
)

# 改为 (subtask trainer):
detached_hidden = [h.detach() for h in qwen_output.hidden_states]
action_loss_raw = model.action_loss_from_hidden_states(
    detached_hidden, action_examples, indices=action_index_tensor,
)
```

效果：
- VLM CE loss 的梯度 → 正常回传 backbone（学习 subtask 文本生成）
- Action FlowMatching loss 的梯度 → 只更新 ActionHead（不破坏 backbone）

---

## 五、数据集现状分析（LeRobot 格式 + Subtask 标注）

### 5.1 数据集路径

```
/diff/wallx_workspace/xch/lerobot_ego_data_subtask
├── data/chunk-{idx:03d}/episode_{idx:06d}.parquet   # 882 个 episode
├── videos/chunk-{idx:03d}/{video_key}/episode_{idx:06d}.mp4
├── meta/
│   ├── info.json          # 数据集元信息（已注册 subtask 列）
│   ├── tasks.jsonl         # 任务描述
│   ├── episodes.jsonl      # episode 索引 + 长度
│   └── subtask_configs.json # 441 条轨迹级配置
└── norm_stats_ego.json     # 归一化统计
```

### 5.2 Parquet 列

| 列名 | dtype | 说明 |
|------|-------|------|
| index | int32 | 全局帧索引 |
| episode_index | int32 | episode 编号 |
| frame_index | int32 | 帧内索引 |
| timestamp | float64 | 时间戳 |
| task_index | int32 | 任务索引 |
| state | float32 [6] | 当前位姿 (x, y, z, roll=0, pitch=0, yaw) |
| action | float32 [6] | 目标位姿 |
| bbox | float32 [4] | 边界框 |
| grasp | bool | 夹爪状态 |
| keyframe | int32 | 关键帧标记 |
| pred_signal | string | `<pred_action>` / `<stop>` |
| is_rotate | int32 | 1=yaw_only, 0=flight |
| **phase_id** | **int32** | **当前阶段编号 1-6（新增）** |
| **subtask_id** | **int32** | **当前 subtask 编号 1-6（新增）** |
| **subtask_text** | **string** | **subtask 原语文本（新增）** |

### 5.3 Episode 配对结构

882 个 episode = **441 对**，每对共享相同的任务指令：

| 类型 | Episode 索引 | grasp | is_rotate 变化次数 | 阶段数 |
|------|-------------|-------|-------------------|--------|
| Pick 半段 | 偶数 (0,2,4,...) | 始终 False | 1 次 (1→0) | 2 段 |
| Place 半段 | 奇数 (1,3,5,...) | 始终 True | 3 次 (1→0→1→0) | 4 段 |

- 每对 = 1 条完整轨迹，共 441 条轨迹，75 种不同任务指令
- 完整轨迹 = Pick 半段 (Phase 1+2) + Place 半段 (Phase 3+4+5+6)

### 5.4 Subtask 标注验证

已验证 subtask_configs.json：441 条配置，每条 6 个 subtask，全部匹配 parquet 帧级数据。

### 5.5 已知限制

- 当前只有 Search 和 Fly to 两个原语，Phase 2 的抓取和 Phase 6 的放置都隐含在 Fly to 中
- 后续如需 Place 原语，需要 gripper state 过渡帧数据来拆分 Phase 6

---

## 六、实现方案

### 6.1 改动清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `starVLA/dataloader/wallx_cotrain_datasets.py` | **修改** | 新增 `WallXSubtaskDataset` 类 |
| `starVLA/dataloader/__init__.py` | **修改** | 注册 `wallx_subtask_dataset` |
| `starVLA/training/train_starvla_subtask.py` | **新建** | 基于 router trainer 改造的 subtask trainer |
| `starVLA/config/training/starvla_subtask_wallx.yaml` | **新建** | subtask 训练配置 |

**不需要修改的文件**：
- `starVLA/model/framework/QwenPI.py` — framework 代码不变，trainer 直接调 `qwen_vl_interface` + `action_loss_from_hidden_states`
- `starVLA/model/modules/action_model/` — action head 代码不变
- `starVLA/model/tools.py` — 不需要注册新 framework

### 6.2 WallXSubtaskDataset

继承 `WallXRouterDataset`，覆盖 prompt 构造和路由逻辑：

```python
class WallXSubtaskDataset(WallXRouterDataset):
    """Subtask-aware dataset: 替代 router，所有样本都是 action 路由。
    subtask_history 基于 per-trajectory（跨 episode）构建。
    """

    def __init__(self, data_cfg):
        super().__init__(data_cfg)
        self.subtask_prompt_template = str(_cfg_get(
            data_cfg, "subtask_prompt_template", DEFAULT_SUBTASK_PROMPT,
        ))
        # 加载 subtask_configs.json 用于构建 per-trajectory subtask_history
        self._subtask_configs = self._load_subtask_configs()

    def _load_subtask_configs(self) -> dict:
        """加载 meta/subtask_configs.json，按 trajectory_id 索引。"""
        configs_path = Path(self.common.root) / "meta" / "subtask_configs.json"
        if not configs_path.exists():
            return {}
        with open(configs_path) as f:
            configs = json.load(f)
        return {c["trajectory_id"]: c for c in configs}

    def _get_subtask_history(self, sample: dict) -> str:
        """per-trajectory subtask_history：取同一轨迹内 subtask_id < 当前值 的所有 subtask_text。
        trajectory_id = episode_index // 2（偶数 episode=pick, 奇数=place）。
        """
        ep_idx = int(sample.get("episode_index", 0))
        traj_id = ep_idx // 2
        current_subtask_id = int(sample.get("subtask_id", 1))

        config = self._subtask_configs.get(traj_id)
        if config is None or current_subtask_id <= 1:
            return ""

        completed = []
        for st in config["subtasks"]:
            if st["subtask_id"] < current_subtask_id:
                completed.append(st["subtask_text"])
        return ". ".join(completed)

    def _make_subtask_sample(self, index: int) -> dict[str, Any] | None:
        effective_index, _, sample = self._resolve_training_sample(index)

        # 图像处理（复用父类逻辑）
        front = torch.as_tensor(sample["video.front"])
        photometric_params = self._sample_photometric_params()
        current_image = self._image_from_tensor(front, photometric_params)
        history_images = self._collect_history_images(effective_index, sample, photometric_params)
        images = [current_image] + history_images

        # 读取 subtask 标注
        subtask_text = str(sample.get("subtask_text", ""))
        if not subtask_text:
            return None

        subtask_history = self._get_subtask_history(sample)

        # 构建 prompt
        task_str = str(sample.get("task", ""))
        instruction, _, _ = _parse_task(task_str)
        lang = self.subtask_prompt_template.format(
            instruction=instruction if instruction else task_str,
            subtask_history=subtask_history,
        )

        # Action 处理
        state = torch.as_tensor(sample["state"], dtype=torch.float32)
        action = torch.as_tensor(sample["action"], dtype=torch.float32)
        action = self._apply_action_mode(action, state)
        action = self._truncate_with_keyframe(action, sample.get("keyframe", None))
        if self.normalize_action and self.action_norm_stats is not None:
            action = self._normalize_with_stats(action, self.action_norm_stats)

        output = {
            "image": images,
            "lang": lang,
            "solution": subtask_text,        # ← VLM CE loss 的目标
            "route": "action",               # ← 全部走 action 分支
            "subtask_text": subtask_text,     # ← 供 logging / 推理使用
            "action": action.detach().cpu().numpy().astype(np.float16),
        }
        if self.include_state:
            out_state = state
            if self.normalize_state and self.state_norm_stats is not None:
                out_state = self._normalize_with_stats(out_state, self.state_norm_stats)
            output["state"] = out_state.detach().cpu().numpy()[None, :].astype(np.float16)
        return output

    def __getitem__(self, index):
        n = len(self)
        sample = self._make_subtask_sample(index % n)
        if sample is not None:
            return sample
        for _ in range(self.max_retry):
            candidate = self._random.randrange(n)
            sample = self._make_subtask_sample(candidate)
            if sample is not None:
                return sample
        raise RuntimeError("Failed to sample valid subtask example.")
```

### 6.3 Subtask Prompt 模板

```python
DEFAULT_SUBTASK_PROMPT = (
    "{instruction}\n"
    "You are performing a drone navigation task. The first image is the current front view; "
    "any following images are previous keyframes for context.\n"
    "Completed subtasks: {subtask_history}\n"
    "What subtask should you perform now? Output exactly one subtask."
)
```

示例填充结果（Trajectory 0, Episode 1, subtask_id=4）：
```
Pick up the drink bottle on the desk on your back right, turn to your back
and fly to the kitchen, then turn to your left and place the drink bottle
on the stool.
You are performing a drone navigation task. The first image is the current
front view; any following images are previous keyframes for context.
Completed subtasks: Search drink bottle at back right. Fly to desk. Search kitchen at back
What subtask should you perform now? Output exactly one subtask.
```

Solution (GT): `"Fly to kitchen"`

### 6.4 Subtask Trainer

基于 `VLARouterTrainer`，核心改动只有两处：

**改动 1 — Knowledge Isolation**（`_train_step` 中）：

```python
# 原来:
action_loss_raw, _, _ = self._compute_router_action_loss(
    qwen_output.hidden_states, batch_router, action_indices,
)

# 改为:
detached_hidden = [h.detach() for h in qwen_output.hidden_states]
action_loss_raw, _, _ = self._compute_router_action_loss(
    detached_hidden, batch_router, action_indices,
)
```

**改动 2 — 简化路由**（所有样本都是 action）：

```python
# 原来:
action_indices = self._action_indices(batch_router)  # 按 route=="action" 过滤

# 改为: 全部都是 action，不需要过滤
action_indices = list(range(len(batch_router)))
```

**改动 3 — 更新 logging**：

```python
log_dict.update({
    "loss": total_loss_value,
    "subtask_vlm_loss": vlm_loss_raw_value,    # 原来的 vlm_loss
    "action_dit_loss": action_loss_raw_value,   # action head loss
})
```

**其余逻辑完全复用**：
- `_prepare_router_batch` → 重命名为 `_prepare_subtask_batch`，逻辑不变（已经从 dataset 拿到了 lang + solution）
- `_compute_router_action_loss` → 逻辑不变（已经在内部调 `action_loss_from_hidden_states`）
- 数据迭代、checkpoint 保存、分布式训练 → 不变

### 6.5 训练配置 YAML

```yaml
run_id: qwenpi_wallx_subtask_v1
run_root_dir: results/Checkpoints
seed: 42
trackers: [jsonl, wandb]
wandb_entity: your_entity
wandb_project: qwenpi_wallx_subtask
is_debug: false

framework:
  name: QwenPI
  qwenvl:
    base_vlm: /diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct
    attn_implementation: flash_attention_2
    torch_dtype: bfloat16
    vl_hidden_dim: 2560
  dino:
    dino_backbone: dinov2_vits14
  action_model:
    action_model_type: DiT-B
    hidden_size: 1024
    add_pos_embed: true
    max_seq_len: 1024
    action_dim: 6
    state_dim: 6
    future_action_window_size: 31
    action_horizon: 32
    past_action_window_size: 0
    repeated_diffusion_steps: 8
    noise_beta_alpha: 1.5
    noise_beta_beta: 1.0
    noise_s: 0.999
    num_timestep_buckets: 1000
    num_inference_timesteps: 4
    num_target_vision_tokens: 32
    diffusion_model_cfg:
      cross_attention_dim: 2560
      dropout: 0.2
      final_dropout: true
      interleave_self_attention: true
      norm_type: "ada_norm"
      num_layers: 16
      output_dim: 2560
      positional_embeddings: null

datasets:
  split:
    enable: true
    train_ratio: 0.9
    mode: contiguous

  subtask_data:
    dataset_py: wallx_subtask_dataset
    repo_id: dzb/lerobot_ego_data_subtask
    root: /diff/wallx_workspace/xch/lerobot_ego_data_subtask
    per_device_batch_size: 4
    num_workers: 4
    image_size: [224, 224]
    action_horizon: 32
    action_in_ego: true
    use_delta_action: false
    normalize_action: true
    normalize_state: true
    truncate_keyframe_value: 2
    include_history_keyframes: true
    max_history_keyframes: 2
    history_keyframe_values: [1]
    snap_rotation_to_start: true
    include_state: false
    max_retry: 64
    video_backend: pyav
    tolerance_s: 0.0001
    photometric_augmentation:
      enabled: false
      probability: 1.0
      brightness: 0.20
      contrast: 0.20
      saturation: 0.20
      hue: 0.02
      sharpness: 0.10
    subtask_prompt_template: "{instruction}\nYou are performing a drone navigation task. The first image is the current front view; any following images are previous keyframes for context.\nCompleted subtasks: {subtask_history}\nWhat subtask should you perform now? Output exactly one subtask."

trainer:
  epochs: 100
  max_train_steps: 15000
  num_warmup_steps: 500
  save_interval: 1000
  eval_interval: 50
  learning_rate:
    action_model: 3.0e-06
    base: 1.0e-05
    qwen_vl_interface: 1.0e-05
  lr_scheduler_type: cosine_with_min_lr
  scheduler_specific_kwargs:
    min_lr: 2.0e-06
  freeze_modules: ''
  loss_scale:
    vlm: 1.0
    action: 1.0
  repeated_diffusion_steps: 4
  max_grad_norm: 1.0
  warmup_ratio: 0.1
  weight_decay: 0.0
  logging_frequency: 10
  gradient_clipping: 5.0
  gradient_accumulation_steps: 1
  enable_gradient_checkpointing: true
  enable_mixed_precision_training: true
  save_format: pt
  is_resume: false
  resume_epoch: null
  resume_step: null
```

### 6.6 训练启动命令

```bash
cd /diff/wallx_workspace/starVLA

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero3.yaml \
  --num_processes 4 \
  --main_process_port 29523 \
  starVLA/training/train_starvla_subtask.py \
  --config_yaml starVLA/config/training/starvla_subtask_wallx.yaml
```

### 6.7 dataloader/__init__.py 注册

```python
elif dataset_py == "wallx_subtask_dataset":
    from starVLA.dataloader.wallx_cotrain_datasets import collate_fn_router, get_subtask_dataset
    subtask_dataset_cfg = cfg.datasets.subtask_data
    subtask_dataset = get_subtask_dataset(data_cfg=subtask_dataset_cfg)
    subtask_train_dataloader = DataLoader(
        subtask_dataset,
        batch_size=int(subtask_dataset_cfg.per_device_batch_size),
        collate_fn=collate_fn_router,
        num_workers=int(subtask_dataset_cfg.get("num_workers", 4)),
    )
    return subtask_train_dataloader
```

---

## 七、训练阶段

### 7.1 阶段一：Subtask VLM 预热（λ_action=0）

- 设置 `loss_scale.action: 0`，只训练 VLM 生成 subtask_text
- 验证模型能正确预测 Search / Fly to 原语
- 相当于纯 VLM SFT

### 7.2 阶段二：联合训练（λ_action=1）

- 开启 action loss，但 hidden_states detach（Knowledge Isolation）
- VLM 学 subtask 生成 + ActionHead 学动作预测，互不干扰
- `loss_scale.vlm: 1.0`, `loss_scale.action: 1.0`

### 7.3 阶段三：渐进式自主化

- **Guided 模式**：人工给定 subtask，只跑 action head（等效于现有 router 的 action 路径）
- **Semi-autonomous**：模型生成 subtask 候选，人工确认后执行
- **Autonomous**：模型自主预测 subtask + action head 全自主执行

---

## 八、推理接口

### 8.1 推理流程

```python
@torch.inference_mode()
def predict_subtask_and_action(model, example):
    # Step 1: 构建 subtask prompt
    prompt = subtask_prompt_template.format(...)

    # Step 2: VLM 生成 subtask_text
    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=[example["image"]], instructions=[prompt],
    )
    qwen_output = model.qwen_vl_interface(**qwen_inputs, output_hidden_states=True)

    # Step 3: 解码 subtask_text
    generated = model.qwen_vl_interface.generate(**{k: v for k, v in qwen_inputs.items() if k != "labels"})
    subtask_text = tokenizer.decode(generated.sequences[0], skip_special_tokens=True)

    # Step 4: Action head 用 hidden_states 预测动作
    vl_embs_list = model._select_action_hidden_states(qwen_output.hidden_states)
    pred_actions = model.action_model.predict_action(vl_embs_list, state)
    return {"subtask_text": subtask_text, "normalized_actions": pred_actions}
```

### 8.2 周期性重新预测

- 每 N 帧（约 2 秒）重新调用 subtask 生成
- 如果新 subtask 与前一个不同，触发 subtask_history 更新
- Action head 每帧都运行

---

## 九、与原方案的关键差异总结

| 对比项 | 原方案（基于 QwenGR00T） | 新方案（基于 QwenPI + Router） |
|--------|------------------------|---------------------------|
| 基础 Framework | 新建 QwenGR00TSubtask | 不改 Framework，复用 QwenPI |
| Action Head | FlowmatchingActionHead (单层) | LayerwiseFlowmatchingActionHead (多层) |
| hidden states | 取最后一层 h | 取最后 N 层 list |
| Knowledge Isolation | h.detach() | [h.detach() for h in hidden_states] |
| subtask embedding | concat 到 h（需要 _encode_subtask） | 不需要，VLM 自己生成 subtask_text |
| 训练入口 | 改 framework.forward() | 改 trainer._train_step() |
| Router 处理 | 未考虑 | Subtask 替代 Router |
| 新增文件数 | 5 个 | 2 个（trainer + config）+ 2 处修改 |
| lm_head 使用 | 手动调用（需处理 LayerNorm） | 不需要，VLM CE loss 由 qwen_vl_interface 内部处理 |

核心简化：**不需要手动操作 lm_head、不需要 _encode_subtask、不需要 concat subtask_emb**。VLM forward 传 `labels=solution` 时，HuggingFace 模型内部自动处理 norm → lm_head → CE loss。Subtask 文本生成完全通过标准 VLM 训练路径实现。
