# 29f97a9 -> 7d522b1 代码改动开发文档

> 说明：
> 1. 这是一次 **历史提交区间分析**，不是当前 router 工作流的操作手册。
> 2. 当前主用文档请优先看 [`wallx_router_current_workflow.md`](./wallx_router_current_workflow.md)。
> 3. 本文最初基于旧根目录 `/beijing-c/wallx_workspace` 撰写；当前实际仓库根目录是 `/diff/wallx_workspace`。

## 1. 文档范围

- 仓库：`/diff/wallx_workspace/starVLA`
- 提交范围：`29f97a976262247f059e5d3a916bd3eaa69bd1be` -> `7d522b1f68ceec8e768e156a3bd2e308860cab19`
- 这个区间内实际只有 1 个提交落在 `HEAD`，提交信息是 `for diff`。
- 本文主要依据代码内容、配置和文件命名来分析改动目的；没有从提交信息里获得额外业务背景。
- 不确定的地方会直接标注为“不确定”。

## 2. 改动总览

这次提交的主目标可以概括为：

1. 把 `WallX` 场景完整接入 `StarVLA/QwenPI` 的训练、评估和闭环服务链路。
2. 把原来的共训练流程从“VLA + 单一 VLM”扩展成更细的多任务结构：
   - `VLA action`
   - `VLM bbox`
   - `VLM signal`
3. 为 `WallX` 增加一套更贴近实际使用方式的推理工具：
   - 开放环评估脚本
   - 与 `wall-x` 模拟客户端兼容的 websocket 闭环服务
4. 为 VLM 文本监督调整 Qwen 输入构造方式，使其能直接监督 assistant 输出文本，而不是继续依赖旧的 action token 范围做 label masking。
5. 增加多 VLM 数据源混训能力，把 `WallX` 自身数据和 `GQA / Visual Genome` 这样的外部视觉语言数据拼接到同一个训练入口里。

从改动体量看，这不是一次局部修补，而是一次新的 `WallX` 专项训练/评估工作流落地。

## 3. 文件级别分组

| 类别 | 文件 | 作用判断 |
| --- | --- | --- |
| 运行产物管理 | `.gitignore` | 忽略 `debug/*`、`assets/*`、`results/*`，明显是为了容纳新脚本生成的大量产物 |
| 推理/评估脚本 | `scripts/eval_open_loop_wallx.py` | 新增开放环评估 |
| 推理/服务脚本 | `scripts/serve_wallx_closed_loop.py` | 新增面向 `wall-x` 客户端的闭环 websocket 服务 |
| 推理/服务备份 | `scripts/serve_wallx_closed_loop.py.bak_before_frame_debug`、`scripts/serve_wallx_closed_loop_bk.py` | 开发过程中的阶段性备份，保留旧版服务实现 |
| DeepSpeed 配置 | `starVLA/config/deepseeds/ds_config.yaml`、`zero3.yaml` | 打开 `activation_checkpointing`，并切到 `bf16` |
| WallX 训练配置 | `starVLA/config/training/starvla_cotrain_wallx_qwenpi*.yaml` | 新增 WallX 专用训练入口配置 |
| 数据加载入口 | `starVLA/dataloader/__init__.py` | 新增 WallX 专属 dataset 路由和 Qwen3 SFT dataloader 路由 |
| 外部 VLM 数据配置 | `starVLA/dataloader/qwenvl_llavajson/qwen_data_config.py` | 注册本地 `GQA`、`Visual Genome` 数据源 |
| 新数据集实现 | `starVLA/dataloader/wallx_cotrain_datasets.py` | 新增 WallX VLA/VLM bbox/VLM signal 三类数据集 |
| 外部 VLM SFT 数据集 | `starVLA/dataloader/vlm_sft_qwen3_datasets.py` | 新增 Qwen3 风格 VLM SFT 数据读取 |
| 模型适配 | `starVLA/model/framework/QwenPI.py`、`starVLA/model/modules/vlm/QWen2_5.py`、`QWen3.py` | 适配新训练目标和监督方式 |
| 训练入口 | `starVLA/training/train_starvla_cotrain.py`、`train_starvla_cotrain_multi_vlm.py`、`train_starvlm_cotrain_vlm.py`、`train_starvla_cotrain_backup.py` | 新增或重构 WallX 训练流程 |

## 4. 核心改动主线

### 4.1 主线结论

如果只抓一条主线，这次提交做的是：

> 以 `QwenPI` 为核心，把 `WallX` 数据集里的动作预测、目标框定位、动作触发信号判断三类任务，串成一个统一训练体系，并补齐对应的开放环评估与闭环部署能力。

### 4.2 训练/推理形态变化

提交前后最明显的变化是任务结构被拆细了。

旧思路更像：

```text
VLA action + 一个 VLM 辅助分支
```

新思路更像：

```text
WallX LeRobot 数据
  -> VLA 分支：预测动作序列
  -> VLM bbox 分支：输出目标框文本
  -> VLM signal 分支：判断是继续预测动作，还是已经可以输出目标框
```

闭环推理时，这个结构又变成：

```text
obs
  -> signal VLM
       -> 如果输出 <|pred_action|> ：继续走 action policy
       -> 如果输出 bbox           ：跳过 action，直接把 bbox/停止信号返回客户端
```

这个“先判定是否该继续动作，再决定是否调用动作模型”的逻辑，是本次提交最有辨识度的设计点之一。

## 5. 数据层改动分析

### 5.1 新增 `wallx_cotrain_datasets.py`

这是整次改动最关键的新文件之一。它把 `WallX` 的 LeRobot 数据拆成三种训练视角：

#### `WallXVlaDataset`

职责：

- 从 LeRobot 数据中读取图像、状态、动作轨迹。
- 按配置决定是否把动作转换到 ego frame。
- 按配置决定是否改成 delta action。
- 读取归一化统计量，对 action/state 做归一化。
- 从任务文本中解析 `Catch` / `Put` 目标，生成动作预测 prompt。
- 支持拼接历史关键帧图像。

直接证据：

- `starVLA/dataloader/wallx_cotrain_datasets.py` 中 `WallXVlaDataset`
- 使用 `action_prompt_template_grasp_true` / `action_prompt_template_grasp_false`
- `_convert_action_to_ego`、`_truncate_with_keyframe`、`_load_norm_stats`

目的判断：

- 让 `WallX` 原始数据能直接被 `QwenPI` 的动作分支消费。
- 因为动作预测不是简单 copy 数据，而是包含坐标系转换、动作截断和归一化，所以这里本质上是把 `WallX` 数据“翻译”成 StarVLA 的训练格式。

#### `WallXVlmBboxDataset`

职责：

- 从同一份 LeRobot 数据中取图像和 bbox。
- 把 bbox 缩放到训练使用的图像尺寸。
- 生成严格格式的文本答案：`<point>[x1, y1, x2, y2]</point>`
- 当 bbox 无效时重采样。

目的判断：

- 用文本生成的方式训练 VLM 输出目标框，而不是做一个独立检测头。
- 这和后面的 `QWen2_5.py` / `QWen3.py` label 构造调整是配套的。

#### `WallXVlmSignalDataset`

职责：

- 训练一个“当前是否已经接近目标、是否该执行动作”的判断分支。
- 监督信号有两种：
  - 输出 `<|pred_action|>`：表示还没到位，应该继续动作预测
  - 输出 bbox：表示已经可以定位目标并停止继续动作
- 当 `pred_signal` 是 stop，但 bbox 无效时，样本会被丢弃并重采样。

目的判断：

- 这是为闭环控制里的“动作/停止切换”服务的。
- 从代码看，它不是单纯的分类任务，而是“分类 + 条件 bbox 输出”的混合式文本监督。

### 5.2 数据集公共能力

`_WallXLeRobotBase` 说明这次不只是堆了几个 dataset，而是做了统一抽象。

公共能力包括：

- 基于 `LeRobotDatasetMetadata` 获取 episode 数量和 fps
- 支持 `episode_start` / `num_episodes`
- 对 LeRobot 非零起点 episode 子集做本地 frame index 映射
- 支持历史关键帧提取

目的判断：

- 这是为了让 `WallX` 数据既能服务训练，也能服务按 episode 切分的评估。
- 其中“非零 episode 起点时自己做 frame 映射”的逻辑，说明作者遇到过 LeRobot 原生 episode 子集能力不完全满足需求的问题。

### 5.3 `dataloader/__init__.py` 的扩展

`starVLA/dataloader/__init__.py` 新增了这些 dataset 路由：

- `vlm_sft_qwen3_datasets`
- `wallx_vla_dataset`
- `wallx_vlm_dataset`
- `wallx_vlm_signal_dataset`

目的判断：

- 把 WallX 数据和 Qwen3 风格 VLM SFT 数据，正式纳入 StarVLA 的统一 dataloader 工厂，而不是靠临时脚本单独绕开框架。

### 5.4 外部 VLM 数据接入

`starVLA/dataloader/qwenvl_llavajson/qwen_data_config.py` 新增：

- `gqa_local`
- `visual_genome_local`

`starVLA/dataloader/vlm_sft_qwen3_datasets.py` 则提供了对应的数据读取、图像处理、对话模板展开和 collator。

目的判断：

- 为 `train_starvla_cotrain_multi_vlm.py` 提供外部视觉语言监督来源。
- 也就是说，多源 VLM 混训不是只在配置上声明，而是底层 dataloader 也配齐了。

## 6. 训练层改动分析

### 6.1 `train_starvla_cotrain.py`：从双分支扩到三分支

这个文件的变化是本次训练主线的中心。

直接能确认的新能力：

- `prepare_data()` 现在同时构造：
  - `vla_train_dataloader`
  - `vlm_bbox_train_dataloader`
  - `vlm_signal_train_dataloader`
  - 对应 eval dataloader
- 加入 `datasets.split.enable` 的连续 episode 切分能力
- `VLAMTrainer` 新增：
  - `loss_scale_vlm_total`
  - `loss_scale_vlm_bbox`
  - `loss_scale_vlm_signal`
- `_train_step()` 同时计算：
  - `action_loss`
  - `vlm_bbox_loss_raw`
  - `vlm_signal_loss_raw`
  - 再按权重组合成 `vlm_loss`
- `eval_action_model()` 也同步评估 bbox/signal 两路 VLM loss

可以把训练逻辑概括为：

```text
一次 step
  1. VLA batch -> action_loss
  2. bbox batch -> vlm_bbox_loss
  3. signal batch -> vlm_signal_loss
  4. 按权重组合，再一起反传
```

目的判断：

- 这说明 `WallX` 不再只是“动作预测带一点 VLM 辅助”，而是显式训练一个更完整的多任务系统。
- 同时加入 eval split，说明作者想把“训练 episode”和“评估 episode”在数据层明确隔开，而不是继续只看 train set 指标。

### 6.2 `train_starvla_cotrain.py` 中一个很重要的事实

主配置 `starVLA/config/training/starvla_cotrain_wallx_qwenpi.yaml` 里：

```yaml
loss_scale:
  vla: 1.0
  vlm: 1.0
  vlm_bbox: 0.2
  vlm_signal: 0.0
```

这意味着：

- 代码已经支持 signal loss 分支
- 但主配置默认把 `vlm_signal` 权重设成了 `0.0`
- 配置里虽然还有 `loss_scale.vla`，但在这批新增/修改的训练脚本中没有看到它被实际读取，当前 action loss 看起来是直接使用原值反传

能直接确认的是：能力已经接进来，但默认主配置没有真正让 signal loss 参与总损失。

这里的真实意图不确定，可能是：

- signal 分支能力先接通、先验证流程，暂时不训练
- 或者只想先在推理/评估里使用 signal prompt
- 或者作者当时还在试验 signal loss 的权重

这三种解释里，代码本身无法唯一证明哪一种是作者真实目的，所以只能标为“不确定”。

### 6.3 `train_starvla_cotrain_multi_vlm.py`：多源 VLM 混训

这个文件新增的核心能力是：

- 保留 WallX 的 VLA 分支
- 把 VLM 分支升级为多来源
- 支持每步按权重随机抽取一个 VLM source
- 可选在多卡之间同步 source 选择
- 为某个指定 source 单独做 eval

从配置 `starvla_cotrain_wallx_qwenpi_multi_vlm.yaml` 看，当前启用的 source 是：

- `dzb`：WallX 自身 bbox 数据
- `gqa`
- `vg`

并且配置了权重：

- `dzb: 0.7`
- `gqa: 0.15`
- `vg: 0.15`

目的判断：

- 这是典型的“机器人任务数据 + 通用视觉语言数据”联合训练思路。
- 目标应该是增强 VLM 的通用定位/理解能力，同时保留 WallX 场景专属性。

关于为什么是 `0.7 / 0.15 / 0.15` 这个比例，代码里没有解释，属于“不确定”。

### 6.4 `train_starvlm_cotrain_vlm.py`：VLM-only 对齐入口

这个新增文件的目的很清楚：

- 做一个只训练 VLM 分支的入口
- 但数据处理方式、WallX list-style batch 处理方式、episode split 方式，都尽量和 cotrain 入口保持一致

目的判断：

- 方便单独验证 VLM 数据和监督方式
- 也方便在不训练 action model 的情况下，先把 bbox/VLM 能力训出来

### 6.5 `train_starvla_cotrain_backup.py`

这个文件名本身就说明它是备份脚本。

从代码内容看，它更像当前主训练脚本的前一阶段版本，特点是：

- 有 episode split
- 有 VLA + 单 VLM 分支
- 但没有引入 `vlm_signal` 这条第三分支

目的判断：

- 大概率是为了在主脚本继续演进时保留一个“较稳定的旧版共训入口”。
- 这属于合理推断，但因为文件里没有注释明确写“为何保留”，所以只能说“看起来是回滚/对照用备份”，不能说得更死。

## 7. 模型层改动分析

### 7.1 `QWen2_5.py` 和 `QWen3.py`：VLM label 构造方式改变

这两个文件的改动方向高度一致，说明这是一次明确的接口调整。

旧逻辑：

- 在有 `solutions` 时，通过 action token 范围去找监督起点
- 这更像是面向原先特殊 token 监督的方案

新逻辑：

- 如果传入 `solutions`，就构造完整对话
- 找到 `"<|im_start|>assistant\n"` 后面的 assistant 内容区间
- 只保留 assistant 答案 span 作为 labels
- 其余 token 统一置为 `IGNORE_INDEX`

目的判断：

- 让 VLM 分支可以稳定监督普通文本答案，而不是依赖特定 action token 范围
- 对本次新增的两种 WallX VLM 任务尤其必要：
  - bbox 文本：`<point>[...]</point>`
  - signal 文本：`<|pred_action|>` 或 bbox 文本

换句话说，如果不改这里，前面的 `WallXVlmBboxDataset` 和 `WallXVlmSignalDataset` 很难以当前形式直接训练。

### 7.2 `QwenPI.py`：恢复 `repeated_diffusion_steps` 配置驱动

这里的变化很小，但值得单独记：

- 旧代码把 `repeated_diffusion_steps` 强制写死成 `2`
- 新代码把这行硬编码注释掉，重新使用配置值

直接能确认的是：行为从“强制固定 2”回到了“读配置”。

目的判断：

- 很可能是把之前为“大动作模型/大 action FM”做的临时实验性硬编码撤回。
- 旁边的注释 `NO repeat for big action FM` 暗示过曾有这方面试验，但作者最终为什么恢复配置驱动，代码本身不能完全证明。

## 8. 推理与评估层改动分析

### 8.1 `scripts/eval_open_loop_wallx.py`

这是一个新的开放环评估脚本，按当前代码，它会在选定帧上评估三件事：

1. `signal` 分支输出的是继续动作还是输出 bbox
2. 在 signal 允许的情况下，评估 action 预测误差
3. 独立评估 legacy bbox 任务

输出内容包括：

- 每帧日志
- action 曲线图
- bbox 可视化叠图
- `open_loop_records.json`
- `summary.json`
- 文本报告

目的判断：

- 这是为 `WallX` 新多任务结构提供离线分析工具。
- 以前如果只有训练 loss，很难看出 signal 分支是否真的在“该停的时候停、该动的时候动”。
- 现在这个脚本把 signal gating 逻辑显式纳入评估。

### 8.2 `scripts/serve_wallx_closed_loop.py`

这是另一个关键新增文件，它把模型包装成兼容 `wall-x` 客户端的 websocket 服务。

可以直接确认的设计点：

- 协议上兼容 `wall_x/serving/client_for_sim.py` 一类客户端
- 服务启动时先发送 metadata
- 客户端发 observation dict
- 服务返回 action dict，并附带可选 VQA/signal 信息
- 支持 `msgpack` / `msgpack_numpy`

更重要的是它的内部策略类 `StarVLAWallXPolicy` 是双分支的：

- 线程 A：bbox debug
- 线程 B：signal -> action

并且支持：

- `checkpoint_bbox` / `checkpoint_policy` 分开加载
- `device_bbox` / `device_policy` 分开运行
- 当 signal 输出 bbox 时，按配置可直接跳过动作预测
- 把每帧的 debug 信息写到 `frame_debug_root`

目的判断：

- 这说明作者已经不满足于单卡、单模型、单线程的简单部署方式，而是在为实际闭环仿真调试做工程化包装。
- “bbox 调试”和“policy 决策”拆线程、拆 checkpoint、拆 device，这很像是在解决推理延迟或互相干扰的问题。

但这里要注意：

- 代码体现了工程取向
- 真正的性能瓶颈是不是作者唯一动机，代码无法单独证明，所以只能说“很像是在服务真实闭环调试/部署需求”

### 8.3 服务脚本的两个备份文件

#### `serve_wallx_closed_loop_bk.py`

从代码结构看，这个版本比最终版更早，特征是：

- 仍是 wall-x websocket 服务
- 已有 `vqa_enabled`
- 已有 frame debug 目录
- 但还没有最终版那种明显的双模型、双 checkpoint、双设备、双线程拆分

目的判断：

- 更像“单模型版 / 早期版”的闭环服务实现

#### `serve_wallx_closed_loop.py.bak_before_frame_debug`

文件名已经直接提示它是“frame debug 之前”的备份。

从代码和与最终版的 diff 体量看：

- 这个版本比最终版简单很多
- 没有最终版完整的 per-frame debug 保存逻辑
- 也没有最终版完整的双 checkpoint / 双 device 设计

目的判断：

- 这是闭环服务进一步工程化之前的中间快照

这里有一点要特别说明：

- 文件名说的是 “before_frame_debug”
- 但最终版相对它的改动不只 frame debug，还包括双分支/双模型等增强
- 所以只能说它是“frame debug 之前的一个阶段性版本”，不能把差异只归因于 frame debug

## 9. 配置层改动分析

### 9.1 WallX 专项训练配置

新增的几个配置文件大致可以这样理解：

- `starvla_cotrain_wallx_qwenpi.yaml`
  - 主要 WallX 共训配置
  - 包含 `vla_data`、`vlm_data`、`vlm_signal_data`
- `starvla_cotrain_wallx_qwenpi_multi_vlm.yaml`
  - 在主配置基础上引入多源 VLM 混训
- `starvla_cotrain_wallx_qwenpi_buckup.yaml`
- `starvla_cotrain_wallx_qwenpi_buckup_old.yaml`
  - 命名显示它们是备份配置

这些配置说明：

- WallX 已经被视为一个独立训练场景，而不是临时复用通用配置。

### 9.2 DeepSpeed 配置调整

`ds_config.yaml` 和 `zero3.yaml` 都新增了 `activation_checkpointing`，并且 `zero3.yaml` 明确：

- `fp16.enabled = false`
- `bf16.enabled = true`

目的判断：

- 这通常是为了降低显存压力，并配合大模型/多分支训练。
- 结合本次新增的多 dataloader、多任务和 Qwen VL 分支，这个改动是合理配套项。

### 9.3 `.gitignore`

新增忽略：

- `debug/*`
- `assets/*`
- `results/*`

目的判断：

- 新脚本会产出调试图、可视化图、结果文件、checkpoint 结果目录
- 这次提交明显开始把“运行结果很多”的开发方式常态化，所以需要把这些目录从 Git 里排除

## 10. 这次改动想解决什么问题

综合代码看，我认为这次提交主要在解决下面几类问题。

### 10.1 让 WallX 不只是能训练 action，而是能训练“何时继续动作、何时停止并定位”

这是最核心的问题定义。

证据：

- `vlm_signal_data`
- `WallXVlmSignalDataset`
- `eval_open_loop_wallx.py` 里的 signal-gated action 评估
- `serve_wallx_closed_loop.py` 里的 signal -> action 控制流

### 10.2 让 WallX 的 VLM 监督真正可训练

证据：

- bbox/signal 都是文本输出
- `QWen2_5.py` / `QWen3.py` 改成 assistant span masking

如果不做这个改动，VLM 文本监督很可能不稳或不正确。

### 10.3 让训练和评估基于 episode 做更清晰的数据划分

证据：

- 连续 episode split
- train/eval dataloader 分开构建

这说明作者想避免把评估继续混在训练 episode 里。

### 10.4 让闭环调试更接近真实客户端

证据：

- websocket 服务
- msgpack 协议兼容
- per-frame debug 目录
- dual model / dual device

这说明工作重点已经从“先把模型训出来”延伸到“如何落到模拟器闭环测试”。

## 11. 开发过程痕迹与演进判断

从文件命名和备份文件看，比较像这样的演进顺序：

1. 先做基础 WallX 闭环服务和基础共训版本
2. 再加入 episode split、外部 VLM 数据、VLM-only 入口
3. 再引入 signal 分支
4. 最后增强闭环服务的双分支和 frame debug 能力

这里要强调：

- 这是根据文件内容和命名做的“演进顺序判断”
- 不是提交历史直接写出来的事实
- 因此这一节属于“有依据的推断”，不是完全确定的事实

## 12. 明确能确认的结论

下面这些结论我认为可以直接成立：

1. 本次提交把 `WallX` 训练/评估/闭环服务链路完整接进了 `StarVLA`。
2. 主训练入口已经从 `VLA + 单 VLM` 扩展到 `VLA + bbox VLM + signal VLM`。
3. 多源 VLM 混训能力已经落地，而且明确接入了本地 `GQA` 和 `Visual Genome`。
4. Qwen VLM 输入构造被改成面向 assistant 答案 span 的监督方式，这对 bbox/signal 文本监督是必要配套。
5. 新增了面向 `wall-x` 客户端的 websocket 闭环服务，并支持更强的调试信息保存。
6. 新增了开放环评估脚本，用于把 signal、bbox、action 三条链路一起分析。
7. 提交中没有看到测试文件或 README 级别的正式说明更新；也就是说，这次改动更多体现在代码和配置层，而不是文档层。

## 13. 不确定项

下面这些点我不能从代码里百分百确认，所以明确标成“不确定”：

1. 主配置里为什么把 `vlm_signal` loss 设成 `0.0`。
2. `GQA` 和 `Visual Genome` 的混训权重为什么选成 `0.15 / 0.15`。
3. `QwenPI.py` 里撤销 `repeated_diffusion_steps = 2` 的直接触发原因。
4. `backup` / `buckup` 文件是否只是开发备份，还是仍被某些实际脚本调用。
5. 闭环服务拆成双模型双设备，主要目标究竟是吞吐、时延、调试隔离，还是只是便于实验；代码只能看出“有这个工程倾向”，不能唯一确定主因。

## 14. 一句话总结

这次提交的本质，是把 `StarVLA + QwenPI` 从“能在 WallX 数据上做基本共训”推进到了“能围绕 WallX 做多任务训练、离线评估、闭环部署和调试”的阶段。
