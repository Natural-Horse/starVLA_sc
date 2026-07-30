# `train_starvla_cotrain.py` 训练流程详解

> 说明：
> 1. 本文分析的是较早的 **非 router** `train_starvla_cotrain.py` 路径，不是当前主用的 router 训练入口。
> 2. 当前实际工作流、数据集转换、离线评估和 bbox 浏览，请优先看 [`wallx_router_current_workflow.md`](./wallx_router_current_workflow.md)。
> 3. 本文早期是在旧根目录 `/beijing-c/wallx_workspace` 下撰写；当前实际仓库根目录请对应理解为 `/diff/wallx_workspace`。

## 1. 适用范围

本文针对下面这条训练命令在仓库 `/diff/wallx_workspace/starVLA` 中的实际执行流程做代码级解释：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/accelerate launch \
--config_file starVLA/config/deepseeds/deepspeed_zero3.yaml \
--num_processes 4 \
--main_process_port 29523 \
starVLA/training/train_starvla_cotrain.py \
--config_yaml starVLA/config/training/starvla_cotrain_wallx_qwenpi.yaml \
--datasets.vla_data.data_mix wallx_dzb
```

本文重点解释：

1. 这条命令从启动到训练结束，代码实际经过了哪些文件和函数。
2. `WallX` 这套数据是如何被构造成 `VLA / VLM bbox / VLM signal` 三路训练输入的。
3. `QwenPI` 模型内部怎样把 VLM 和动作头接起来。
4. 一个 step 内到底算了哪些 loss、怎么反传、怎么保存。
5. 这条命令里有哪些参数是“真的影响当前训练流程”的，哪些只是配置里有、但当前入口里没有真正用到。

本文只基于当前仓库代码分析，不推测作者仓库外的运行习惯。不确定的地方会明确说明。

## 2. 先看这条命令的每一段在做什么

### 2.1 `CUDA_VISIBLE_DEVICES=0,1,2,3`

作用：

- 限制当前训练进程只看见 4 张 GPU。
- 后续 `accelerate launch --num_processes 4` 会在这 4 张可见卡上各起一个进程。

### 2.2 `accelerate launch`

作用：

- 由 Hugging Face Accelerate 负责启动多进程训练。
- 脚本 `train_starvla_cotrain.py` 本身没有手动 `torch.multiprocessing.spawn(...)`，多卡进程管理来自这里。

### 2.3 `--config_file starVLA/config/deepseeds/deepspeed_zero3.yaml`

这不是训练 YAML，而是 **Accelerate 的启动配置**。

文件内容很短：

- [`starVLA/config/deepseeds/deepspeed_zero3.yaml`](/diff/wallx_workspace/starVLA/starVLA/config/deepseeds/deepspeed_zero3.yaml)

它声明了：

- `distributed_type: DEEPSPEED`
- `deepspeed_config_file: ./starVLA/config/deepseeds/zero3.yaml`

也就是说：

- `accelerate launch` 会先读取 `deepspeed_zero3.yaml`
- 再间接使用 [`starVLA/config/deepseeds/zero3.yaml`](/diff/wallx_workspace/starVLA/starVLA/config/deepseeds/zero3.yaml)
- 当前命令的 ZeRO-3、bf16、activation checkpointing 这些细节，来自 `zero3.yaml`

### 2.4 `--num_processes 4`

作用：

- 启动 4 个训练进程。
- 在当前命令下，通常就是 4 卡 4 进程。

### 2.5 `--main_process_port 29523`

作用：

- 给多进程通信指定主端口。
- 主要用于分布式初始化，避免跟别的作业端口冲突。

### 2.6 `starVLA/training/train_starvla_cotrain.py`

这是实际被执行的训练脚本：

- [`train_starvla_cotrain.py`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py)

### 2.7 `--config_yaml starVLA/config/training/starvla_cotrain_wallx_qwenpi.yaml`

这是业务训练配置：

- [`starvla_cotrain_wallx_qwenpi.yaml`](/diff/wallx_workspace/starVLA/starVLA/config/training/starvla_cotrain_wallx_qwenpi.yaml)

它决定：

- 用哪个 framework
- 用哪个基础 VLM
- 动作头结构
- 数据集路径和样本格式
- 学习率、保存频率、eval 频率等

### 2.8 `--datasets.vla_data.data_mix wallx_dzb`

这一段**对当前 WallX 入口几乎没有实际影响**。

原因是：

1. 在 `train_starvla_cotrain.py` 里，这个字段只被读取后用于日志打印：
   - [`train_starvla_cotrain.py:214`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L214)
   - [`train_starvla_cotrain.py:215`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L215)
2. 当前 `WallX` 路径实际使用的是 `wallx_vla_dataset`：
   - [`starvla_cotrain_wallx_qwenpi.yaml:51`](/diff/wallx_workspace/starVLA/starVLA/config/training/starvla_cotrain_wallx_qwenpi.yaml#L51)
3. `wallx_vla_dataset` 对应的实现 [`wallx_cotrain_datasets.py`](/diff/wallx_workspace/starVLA/starVLA/dataloader/wallx_cotrain_datasets.py) 里没有使用 `data_mix`

`data_mix` 真正有业务含义的是旧的 `lerobot_datasets.py` 路径，不是当前这条 WallX 路径。

所以对这条命令来说：

- `--datasets.vla_data.data_mix wallx_dzb` 当前更像是“保留的兼容参数/日志字段”
- 不是 WallX 数据构建的核心开关

## 3. 启动后的总流程图

整体流程可以先压缩成下面这张图：

```text
accelerate launch
  -> 读取 accelerate + deepspeed 配置
  -> 启动 4 个 Python 进程
  -> 执行 train_starvla_cotrain.py
       -> 解析 CLI 覆盖参数
       -> 加载 YAML 并 merge
       -> wrap_config(cfg)
       -> setup_directories()
       -> build_framework(cfg) -> QwenPI
            -> get_vlm_model() -> Qwen3-VL 接口
            -> get_action_model() -> Layerwise Flow Matching Action Head
       -> prepare_data()
            -> 构建 VLA dataloader
            -> 构建 VLM bbox dataloader
            -> 构建 VLM signal dataloader
            -> 可选按 episode 切 train/eval
       -> setup_optimizer_and_scheduler()
       -> VLAMTrainer(...)
       -> trainer.prepare_training()
            -> set_seed
            -> 可选加载预训练 checkpoint
            -> 可选冻结模块
            -> accelerator.prepare(...)
            -> init wandb / checkpoint dir
       -> trainer.train()
            -> 反复取三路 batch
            -> action_loss
            -> vlm_bbox_loss
            -> vlm_signal_loss
            -> backward + step + log + eval + save
       -> 保存 final_model
```

## 4. 代码入口：`train_starvla_cotrain.py`

### 4.1 CLI 参数如何进入配置

入口在：

- [`train_starvla_cotrain.py:919`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L919)

执行顺序是：

1. `argparse` 只先拿到 `--config_yaml`
2. 其余参数通过 `parse_known_args()` 收集到 `clipargs`
3. 调用 `normalize_dotlist_args(clipargs)` 把类似：

```bash
--datasets.vla_data.data_mix wallx_dzb
```

转成：

```text
datasets.vla_data.data_mix=wallx_dzb
```

对应函数在：

- [`trainer_tools.py:25`](/diff/wallx_workspace/starVLA/starVLA/training/trainer_utils/trainer_tools.py#L25)

然后：

4. `OmegaConf.load(args.config_yaml)` 读取主 YAML
5. `OmegaConf.from_dotlist(dotlist)` 构造 CLI 覆盖配置
6. `OmegaConf.merge(cfg, cli_cfg)` 合并

也就是说，这条命令里的 CLI 覆盖会直接覆盖 YAML 中对应字段。

### 4.2 `Accelerator` 和 `DeepSpeedPlugin` 是怎么创建的

脚本顶部直接执行：

- [`train_starvla_cotrain.py:72`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L72)
- [`train_starvla_cotrain.py:73`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L73)

```python
deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
```

注意这里有一个重要事实：

- 脚本里没有把 ZeRO stage、bf16、gradient accumulation 这些细节手动传给 `DeepSpeedPlugin(...)`
- 这些运行时分布式细节主要依赖 `accelerate launch --config_file ...` 提供的外部启动配置

所以这条命令里：

- `train_starvla_cotrain.py` 负责“训练逻辑”
- `accelerate launch + deepspeed_zero3.yaml + zero3.yaml` 负责“分布式运行时”

## 5. 主配置 `starvla_cotrain_wallx_qwenpi.yaml` 怎么影响训练

配置文件：

- [`starvla_cotrain_wallx_qwenpi.yaml`](/diff/wallx_workspace/starVLA/starVLA/config/training/starvla_cotrain_wallx_qwenpi.yaml)

核心部分可以分成 4 组：

### 5.1 `framework`

决定模型结构：

- `framework.name: QwenPI`
- `framework.qwenvl.base_vlm: .../Qwen3-VL-4B-Instruct`
- `framework.action_model.*`: 动作头的维度、时序长度、扩散参数

### 5.2 `datasets`

决定三路数据：

- `datasets.vla_data`
- `datasets.vlm_data`
- `datasets.vlm_signal_data`

并且打开了：

```yaml
split:
  enable: true
  train_ratio: 0.9
  mode: contiguous
```

这意味着当前训练会按 episode 做 90%/10% 的连续切分。

### 5.3 `trainer`

决定训练超参：

- `max_train_steps: 15000`
- `num_warmup_steps: 500`
- `save_interval: 1000`
- `eval_interval: 50`
- 多组学习率：
  - `base`
  - `qwen_vl_interface`
  - `action_model`

### 5.4 DeepSpeed 实际使用的混精和 ZeRO

当前命令真正使用的是：

- [`deepspeed_zero3.yaml`](/diff/wallx_workspace/starVLA/starVLA/config/deepseeds/deepspeed_zero3.yaml)
- [`zero3.yaml`](/diff/wallx_workspace/starVLA/starVLA/config/deepseeds/zero3.yaml)

其中 `zero3.yaml` 明确写了：

- `zero_optimization.stage = 3`
- `bf16.enabled = true`
- `fp16.enabled = false`
- 打开 `activation_checkpointing`

所以当前命令的运行时形态是：

- DeepSpeed ZeRO-3
- bf16
- activation checkpointing

## 6. `main(cfg)` 的执行顺序

`main` 在：

- [`train_starvla_cotrain.py:873`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L873)

顺序非常清晰：

1. `cfg = wrap_config(cfg)`
2. `output_dir = setup_directories(cfg)`
3. `vla = build_framework(cfg)`
4. `prepare_data(...)`
5. `setup_optimizer_and_scheduler(...)`
6. 构造 `VLAMTrainer`
7. `trainer.prepare_training()`
8. `trainer.train()`
9. 最后 `dist.barrier()` + `dist.destroy_process_group()`

### 6.1 `wrap_config(cfg)` 做了什么

函数在：

- [`config_tracker.py:476`](/diff/wallx_workspace/starVLA/starVLA/training/trainer_utils/config_tracker.py#L476)

它把原始 OmegaConf 包装成 `AccessTrackedConfig`。

作用：

- 记录训练过程中到底访问了哪些配置字段
- 在 checkpoint 保存时，把“实际访问过的配置”导出到 `config.yaml`

保存逻辑在：

- [`config_tracker.py:434`](/diff/wallx_workspace/starVLA/starVLA/training/trainer_utils/config_tracker.py#L434)
- [`train_starvla_cotrain.py:484`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L484)

### 6.2 `setup_directories(cfg)`

函数在：

- [`train_starvla_cotrain.py:87`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L87)

它会把：

```text
cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
```

当前配置下就是：

```text
results/Checkpoints/qwenpi_wallx_cotrain_vlm_nosignal_data_samelr
```

并创建：

- `output_dir`
- `output_dir/checkpoints`

## 7. 模型构建：`build_framework(cfg)` -> `QwenPI`

### 7.1 框架选择

`build_framework` 在：

- [`framework/__init__.py:33`](/diff/wallx_workspace/starVLA/starVLA/model/framework/__init__.py#L33)

配置里写的是：

```yaml
framework:
  name: QwenPI
```

因此最终会实例化：

- [`QwenPI.py`](/diff/wallx_workspace/starVLA/starVLA/model/framework/QwenPI.py)

### 7.2 `QwenPI` 的两个核心子模块

`Qwen_PI.__init__()` 在：

- [`QwenPI.py:51`](/diff/wallx_workspace/starVLA/starVLA/model/framework/QwenPI.py#L51)

它做的核心事情只有两件：

1. `self.qwen_vl_interface = get_vlm_model(config=self.config)`
2. `self.action_model = get_action_model(config=self.config)`

也就是把整个模型拆成：

```text
QwenPI
  = VLM 编码器（Qwen3-VL interface）
  + 动作头（Layerwise Flow Matching Action Head）
```

### 7.3 当前命令会选中哪个 VLM

`get_vlm_model` 在：

- [`model/modules/vlm/__init__.py:1`](/diff/wallx_workspace/starVLA/starVLA/model/modules/vlm/__init__.py#L1)

因为 `base_vlm` 路径里包含 `Qwen3-VL`，所以当前命令会走：

- [`QWen3.py`](/diff/wallx_workspace/starVLA/starVLA/model/modules/vlm/QWen3.py)

对应类：

- [`QWen3.py:34`](/diff/wallx_workspace/starVLA/starVLA/model/modules/vlm/QWen3.py#L34)

这个模块负责：

1. 从 Hugging Face 路径加载 `Qwen3VLForConditionalGeneration`
2. 创建 `AutoProcessor`
3. 把原始 `images + instruction (+ solution)` 组装成 Qwen3-VL 输入
4. 在 VLM 监督训练时，为 assistant span 构造 labels

### 7.4 当前命令会选中哪个动作头

`QwenPI` 直接从：

- [`LayerwiseFM_ActionHeader.py`](/diff/wallx_workspace/starVLA/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py)

导入 `get_action_model`

而 `get_action_model` 最终返回：

- [`LayerwiseFM_ActionHeader.py:395`](/diff/wallx_workspace/starVLA/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py#L395)

即：

```python
return LayerwiseFlowmatchingActionHead(global_config=config)
```

所以当前命令的动作预测头是：

- `LayerwiseFlowmatchingActionHead`

## 8. `QwenPI` 前向到底怎么计算 `action_loss`

`QwenPI.forward(...)` 在：

- [`QwenPI.py:80`](/diff/wallx_workspace/starVLA/starVLA/model/framework/QwenPI.py#L80)

### 8.1 输入样本格式

它期望 `examples` 是一个 `list[dict]`，每个样本至少包含：

- `image`
- `lang`
- `action`

可选：

- `state`

### 8.2 先走 Qwen3-VL

代码先把：

- 图像列表 `batch_images`
- 文本 `instructions`

送入：

- [`QwenPI.py:103`](/diff/wallx_workspace/starVLA/starVLA/model/framework/QwenPI.py#L103)

也就是：

```python
qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(...)
qwenvl_outputs = self.qwen_vl_interface(...)
```

并要求：

- `output_hidden_states=True`

所以这里不是只取最终 logits，而是保留所有隐藏层输出。

### 8.3 取最后若干层 hidden states 作为动作头条件

代码会读取：

- [`QwenPI.py:111`](/diff/wallx_workspace/starVLA/starVLA/model/framework/QwenPI.py#L111)

```python
all_hidden = qwenvl_outputs.hidden_states
expected_layers = len(self.action_model.model.transformer_blocks)
vl_embs_list = list(all_hidden[-expected_layers:])
```

也就是说：

- 动作头不是只看最后一层 hidden state
- 而是拿 VLM 最后若干层 hidden states
- 然后和动作头内部的每一层 transformer block 做 layer-wise 对齐

这也是 `LayerwiseFlowmatchingActionHead` 这个名字里 `Layerwise` 的来源。

### 8.4 动作标签如何切出来

代码在：

- [`QwenPI.py:120`](/diff/wallx_workspace/starVLA/starVLA/model/framework/QwenPI.py#L120)

会把输入样本里的 `action` 转成 tensor，然后只取最后：

```python
self.future_action_window_size + 1
```

个 step 作为训练目标。

当前配置中：

- `future_action_window_size: 15`

所以这里的动作目标长度是：

- `16`

### 8.5 `repeated_diffusion_steps`

这里会读取：

- [`QwenPI.py:125`](/diff/wallx_workspace/starVLA/starVLA/model/framework/QwenPI.py#L125)

来自：

```yaml
trainer:
  repeated_diffusion_steps: 4
```

它会把：

- `actions_target`
- `vl_embs_list`
- 可选 `state`

都重复 `4` 次后再喂给动作头。

代码里没有进一步解释“为什么要重复”，但从实现看，它是在扩充 flow matching 的训练样本数。

### 8.6 动作头内部怎么出 loss

动作头 `LayerwiseFlowmatchingActionHead.forward(...)` 在：

- [`LayerwiseFM_ActionHeader.py:272`](/diff/wallx_workspace/starVLA/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py#L272)

它的逻辑可以简化成：

1. 对真实动作加噪，生成 `noisy_trajectory`
2. 构造目标速度 `velocity = actions - noise`
3. 用 `ActionEncoder` 编码当前 noisy action
4. 拼上 `future_tokens` 和可选 `state_features`
5. 对每一层 transformer block，使用对应层的 `vl_embs_list[layer_idx]` 做 cross-attention
6. 通过 `action_decoder` 预测速度
7. 用 `MSE(pred_velocity, target_velocity)` 作为 loss

最终返回的是一个标量：

- `action_loss`

## 9. 三路数据是怎么构建出来的

## 9.1 统一入口：`prepare_data(cfg, accelerator, output_dir)`

函数在：

- [`train_starvla_cotrain.py:204`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L204)

当前入口会同时构建 3 路训练数据：

1. `vla_train_dataloader`
2. `vlm_bbox_train_dataloader`
3. `vlm_signal_train_dataloader`

如果开启切分，还会同时构建 3 路 eval dataloader。

### 9.2 episode split 逻辑

配置中：

```yaml
datasets:
  split:
    enable: true
    train_ratio: 0.9
    mode: contiguous
```

对应逻辑在：

- [`train_starvla_cotrain.py:134`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L134)

它会：

1. 分别读取 `vla_data`、`vlm_data`、`vlm_signal_data` 的总 episode 数
2. 校验三者总 episode 数必须一致
3. 计算：
   - train = 前 90%
   - eval = 后 10%
4. 把 `episode_start` / `num_episodes` 写回 train_cfg 和 eval_cfg

因此当前命令下，训练和评估不是随机 split，而是：

- **按 episode 顺序连续切分**

### 9.3 dataloader 工厂：`build_dataloader`

入口在：

- [`dataloader/__init__.py:34`](/diff/wallx_workspace/starVLA/starVLA/dataloader/__init__.py#L34)

当前配置里：

- `vla_data.dataset_py = wallx_vla_dataset`
- `vlm_data.dataset_py = wallx_vlm_dataset`
- `vlm_signal_data.dataset_py = wallx_vlm_signal_dataset`

所以最终会走到：

- [`dataloader/__init__.py:66`](/diff/wallx_workspace/starVLA/starVLA/dataloader/__init__.py#L66)
- [`dataloader/__init__.py:80`](/diff/wallx_workspace/starVLA/starVLA/dataloader/__init__.py#L80)
- [`dataloader/__init__.py:98`](/diff/wallx_workspace/starVLA/starVLA/dataloader/__init__.py#L98)

## 10. `WallX` 三种数据集的样本长什么样

实现文件：

- [`wallx_cotrain_datasets.py`](/diff/wallx_workspace/starVLA/starVLA/dataloader/wallx_cotrain_datasets.py)

### 10.1 共同基类 `_WallXLeRobotBase`

类在：

- [`wallx_cotrain_datasets.py:144`](/diff/wallx_workspace/starVLA/starVLA/dataloader/wallx_cotrain_datasets.py#L144)

它负责：

1. 用 `LeRobotDatasetMetadata` 读取 `fps` 和 `total_episodes`
2. 构造 `delta_timestamps`
3. 根据 `episode_start/num_episodes` 选子集
4. 在 LeRobot 不能直接处理非零起始 episode 时，自己维护本地 frame index 映射
5. 收集历史关键帧图像

### 10.2 `WallXVlaDataset`

类在：

- [`wallx_cotrain_datasets.py:260`](/diff/wallx_workspace/starVLA/starVLA/dataloader/wallx_cotrain_datasets.py#L260)

单样本输出结构在：

- [`wallx_cotrain_datasets.py:443`](/diff/wallx_workspace/starVLA/starVLA/dataloader/wallx_cotrain_datasets.py#L443)

格式是：

```python
{
  "image": [current_image] + history_images,
  "lang": action_prompt,
  "action": normalized_action_chunk,
  # optional
  "state": normalized_state
}
```

这个类会做几件关键预处理：

1. 把 `video.front` 转成训练尺寸的 PIL 图
2. 根据 keyframe 规则拼历史帧
3. 按配置把 action 转到 ego frame
4. 可选转成 delta action
5. 用 `norm_stats*.json` 做归一化
6. 从任务字符串里解析 `Catch` / `Put`，再结合 `grasp` 生成动作 prompt

### 10.3 `WallXVlmBboxDataset`

类在：

- [`wallx_cotrain_datasets.py:456`](/diff/wallx_workspace/starVLA/starVLA/dataloader/wallx_cotrain_datasets.py#L456)

输出格式：

```python
{
  "image": [current_image] + history_images,
  "lang": bbox_prompt,
  "solution": "<point>[x1, y1, x2, y2]</point>"
}
```

关键点：

- 只接受有效 bbox
- bbox 会按训练图像尺寸缩放
- 如果当前 index 的 bbox 无效，会随机重采样直到拿到有效样本

### 10.4 `WallXVlmSignalDataset`

类在：

- [`wallx_cotrain_datasets.py:543`](/diff/wallx_workspace/starVLA/starVLA/dataloader/wallx_cotrain_datasets.py#L543)

输出格式：

```python
{
  "image": [current_image],
  "lang": signal_prompt,
  "solution": "<|pred_action|>" 或 "<point>[...]</point>",
  "pred_signal": raw_pred_signal
}
```

关键点：

- 如果 `pred_signal` 表示“继续动作”，监督答案就是 `<|pred_action|>`
- 如果 `pred_signal` 表示“停止”，监督答案就是 bbox 文本
- 如果 stop 样本没有合法 bbox，会被丢弃并重采样

### 10.5 collate 方式

三路 collate 都是：

- [`wallx_cotrain_datasets.py:644`](/diff/wallx_workspace/starVLA/starVLA/dataloader/wallx_cotrain_datasets.py#L644)

直接返回原始 `list[dict]`，不在 dataloader 层做 tensorization。

这意味着：

- VLA 样本在 `QwenPI.forward` 里再拆开
- VLM 样本在 trainer 的 `_prepare_vlm_batch()` 里再转成 Qwen 输入

## 11. 优化器和学习率组是怎么构建的

入口在：

- [`train_starvla_cotrain.py:300`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L300)

调用的是：

- [`trainer_tools.py:51`](/diff/wallx_workspace/starVLA/starVLA/training/trainer_utils/trainer_tools.py#L51)

### 11.1 参数分组规则

`build_param_lr_groups(model, cfg)` 会读取：

```yaml
trainer:
  learning_rate:
    action_model: 3e-6
    base: 1e-5
    qwen_vl_interface: 1e-5
```

于是最终 param group 会按模块路径分成：

1. `action_model`
2. `qwen_vl_interface`
3. 其余未被覆盖参数 -> `base`

如果 `freeze_modules` 配了模块路径，对应参数会被排除在 param group 之外。

### 11.2 优化器和调度器

优化器固定是：

- `torch.optim.AdamW`

调度器通过 Transformers 的：

- `get_scheduler(...)`

按配置构造：

- `lr_scheduler_type: cosine_with_min_lr`
- `num_warmup_steps: 500`
- `num_training_steps: 15000`

## 12. `trainer.prepare_training()` 做了什么

方法在：

- [`train_starvla_cotrain.py:363`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L363)

执行顺序：

1. 按 rank 设置随机种子
2. 如果配置了 `trainer.pretrained_checkpoint`，则加载预训练权重
3. 按 `freeze_modules` 冻结子模块
4. 打印可训练参数量
5. 调用 `setup_distributed_training(...)`
6. 初始化 wandb
7. 初始化 checkpoint 目录

### 12.1 分布式包装是怎么做的

真正执行的是：

- [`trainer_tools.py:277`](/diff/wallx_workspace/starVLA/starVLA/training/trainer_utils/trainer_tools.py#L277)

内部只有一句：

```python
prepared_components = accelerator.prepare(*components)
```

也就是说：

- model
- optimizer
- train dataloaders
- eval dataloaders

都统一交给 Accelerate/DeepSpeed 包装。

## 13. 一个训练 step 的完整路径

## 13.1 训练循环入口

在：

- [`train_starvla_cotrain.py:654`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L654)

循环条件不是按 `epochs`，而是：

```python
while self.completed_steps < self.config.trainer.max_train_steps:
```

所以当前这条命令真正控制训练时长的是：

- `max_train_steps: 15000`

不是 `epochs: 100`

### 13.2 每步先取三路 batch

取 batch 在：

- [`train_starvla_cotrain.py:563`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L563)

每一步会同时拿到：

- `batch_vla`
- `batch_vlm_bbox`
- `batch_vlm_signal`

三个 iterator 都独立维护，谁先耗尽就单独 reset。

### 13.3 `action_loss`

先执行：

- [`train_starvla_cotrain.py:793`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L793)

```python
output_dict = self.model.forward(batch_vla)
action_loss = output_dict["action_loss"]
```

然后立刻：

```python
self.accelerator.backward(action_loss)
```

### 13.4 `vlm_bbox_loss` 和 `vlm_signal_loss`

trainer 会先把 `list[dict]` 的 VLM batch 转成 Qwen3-VL 输入：

- [`train_starvla_cotrain.py:596`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L596)

具体是：

```python
qwen_vl_interface.build_qwenvl_inputs(
    images=...,
    instructions=...,
    solutions=...
)
```

这里 `solutions` 会触发 `QWen3.build_qwenvl_inputs(...)` 为 assistant span 构造 labels：

- [`QWen3.py:175`](/diff/wallx_workspace/starVLA/starVLA/model/modules/vlm/QWen3.py#L175)

然后训练脚本分别做两次前向：

- [`train_starvla_cotrain.py:804`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L804)
- [`train_starvla_cotrain.py:811`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L811)

得到：

- `vlm_bbox_loss_raw`
- `vlm_signal_loss_raw`

### 13.5 三个 loss 如何组合

组合逻辑在：

- [`train_starvla_cotrain.py:818`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L818)

形式是：

```text
vlm_bbox_loss_weighted   = vlm_bbox_loss_raw   * loss_scale.vlm_bbox
vlm_signal_loss_weighted = vlm_signal_loss_raw * loss_scale.vlm_signal
vlm_loss_raw             = vlm_bbox_loss_weighted + vlm_signal_loss_weighted
vlm_loss                 = vlm_loss_raw * loss_scale.vlm
```

当前配置中：

```yaml
loss_scale:
  vla: 1.0
  vlm: 1.0
  vlm_bbox: 0.2
  vlm_signal: 0.0
```

所以当前这条命令下，实际效果是：

```text
vlm_loss = 0.2 * vlm_bbox_loss_raw + 0.0 * vlm_signal_loss_raw
```

也就是说：

- signal 分支会被前向计算
- 也会出日志
- 但默认配置下不会对总梯度产生贡献

### 13.6 梯度裁剪、step、scheduler

如果配置了：

```yaml
trainer.gradient_clipping: 5.0
```

就会执行：

- [`train_starvla_cotrain.py:825`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L825)

```python
self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
```

随后：

1. `optimizer.step()`
2. `lr_scheduler.step()`

## 14. eval 是怎么做的

入口在：

- [`train_starvla_cotrain.py:700`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L700)

每隔：

- `eval_interval: 50`

执行一次。

### 14.1 eval 内容

当前 eval 会同时做：

1. `self.model.predict_action(examples=examples)`，得到动作预测
2. 对 bbox VLM batch 前向，拿 `vlm_bbox_loss_eval`
3. 对 signal VLM batch 前向，拿 `vlm_signal_loss_eval`

### 14.2 动作指标

动作评估不是按训练 loss，而是：

- 拿预测动作和 GT 动作做欧氏距离
- 再除以元素个数

实现：

- [`trainer_tools.py:290`](/diff/wallx_workspace/starVLA/starVLA/training/trainer_utils/trainer_tools.py#L290)
- [`train_starvla_cotrain.py:721`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L721)

日志字段叫：

- `mse_score`

严格说这里的实现更接近：

- `L2 norm / numel`

不是标准逐元素 MSE。

## 15. 日志、checkpoint 和最终产物

### 15.1 wandb

只有主进程会：

- [`train_starvla_cotrain.py:435`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L435)

```python
wandb.init(...)
```

### 15.2 过程 checkpoint

每 `1000` step：

- [`train_starvla_cotrain.py:690`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L690)

保存：

- `checkpoints/steps_<n>_pytorch_model.pt`

并写入：

- `summary.jsonl`

以及导出实际访问过的配置：

- `config.yaml`

### 15.3 最终模型

训练结束后会保存：

- `final_model/pytorch_model.pt`

逻辑在：

- [`train_starvla_cotrain.py:848`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py#L848)

## 16. 这条命令实际会生成什么目录

按当前配置，主要会落在：

```text
results/Checkpoints/qwenpi_wallx_cotrain_vlm_nosignal_data_samelr/
```

常见文件包括：

- `checkpoints/steps_1000_pytorch_model.pt`
- `checkpoints/steps_2000_pytorch_model.pt`
- ...
- `summary.jsonl`
- `wandb/`
- `config.yaml`（在 checkpoint 保存时导出）
- `final_model/pytorch_model.pt`

## 17. 命令中的参数和配置，哪些真的生效

这一节专门整理“看起来像会生效，但当前代码里不一定真的用到了”的字段。

### 17.1 明确生效的

下面这些字段在当前入口中是明确生效的：

- `framework.name`
- `framework.qwenvl.base_vlm`
- `framework.action_model.*`
- `datasets.split.*`
- `datasets.vla_data.root/repo_id/image_size/...`
- `datasets.vlm_data.*`
- `datasets.vlm_signal_data.*`
- `trainer.max_train_steps`
- `trainer.num_warmup_steps`
- `trainer.save_interval`
- `trainer.eval_interval`
- `trainer.learning_rate.*`
- `trainer.loss_scale.vlm`
- `trainer.loss_scale.vlm_bbox`
- `trainer.loss_scale.vlm_signal`
- `trainer.gradient_clipping`
- `trainer.save_format`
- `trainer.freeze_modules`
- `trainer.pretrained_checkpoint`（如果手动传）

### 17.2 当前代码里“配置存在，但这条路径没真正用到”的

下面这些字段在当前 WallX 入口里没有看到明确生效：

1. `datasets.vla_data.data_mix`
   - 当前只用于日志打印
   - WallX dataset 构建逻辑不使用它

2. `trainer.epochs`
   - 当前训练循环按 `max_train_steps` 结束

3. `trainer.loss_scale.vla`
   - action loss 直接原值反传，没有看到额外乘这个系数

4. `trainer.max_grad_norm`
   - 当前实际使用的是 `trainer.gradient_clipping`

5. `trainer.warmup_ratio`
   - 当前 scheduler 使用的是 `num_warmup_steps`

6. `trainer.enable_gradient_checkpointing`
   - 当前没有看到训练脚本根据这个字段显式调用 `model.gradient_checkpointing_enable()`
   - 当前真正打开 activation checkpointing 的是 DeepSpeed `zero3.yaml`

7. `trainer.enable_mixed_precision_training`
   - 当前没有看到训练脚本根据这个字段分支控制混精
   - 当前混精来自 `torch.autocast(...)` 和 DeepSpeed `bf16`

8. `trackers: [jsonl, wandb]`
   - 当前没有看到基于 `trackers` 列表做条件分支
   - `wandb` 是直接手动初始化的
   - `summary.jsonl` 也是手工写出的

### 17.3 `trainer.gradient_accumulation_steps` 的特殊说明

这是一个最值得注意的字段。

当前代码里：

- 会把它打印到日志
- 会用来计算 `total_batch_size` 显示值

但没有看到：

- 在 `Accelerator(...)` 构造时显式传入它

而真实训练里使用的是：

```python
with self.accelerator.accumulate(self.model):
```

所以更准确地说：

- **当前实际梯度累积行为由 Accelerate/DeepSpeed 运行时决定**
- 不是 `cfg.trainer.gradient_accumulation_steps` 在脚本中直接硬绑定出来的

在你现在这条命令里它是 `1`，所以不会造成表里不一；
但如果以后想改成别的值，这一点需要额外留意。

## 18. 用当前配置理解这条命令的真实训练行为

把上面都收敛起来，当前命令实际上做的是：

```text
4 卡 ZeRO-3 + bf16 训练
  模型: QwenPI
    - VLM: Qwen3-VL-4B-Instruct
    - Action head: Layerwise Flow Matching

  数据:
    - VLA: WallX LeRobot 动作数据
    - VLM bbox: WallX bbox 文本监督
    - VLM signal: WallX signal 文本监督
    - 按 episode 连续切分 90% train / 10% eval

  每一步:
    - 算 action_loss
    - 算 bbox VLM loss
    - 算 signal VLM loss
    - 但默认只把 bbox VLM loss 计入总损失

  默认损失近似:
    total_loss = action_loss + 0.2 * bbox_vlm_loss
```

更严格地说，代码是分两次 `backward`：

1. 先 `backward(action_loss)`
2. 再 `backward(vlm_loss)`

但因为当前 `vlm_signal` 权重为 `0.0`，所以 signal 分支默认只参与监控，不参与梯度。

## 19. 最后给一个“按文件跳转”的速查表

如果你后面要自己继续看代码，建议按下面顺序看：

1. 训练入口：
   - [`train_starvla_cotrain.py`](/diff/wallx_workspace/starVLA/starVLA/training/train_starvla_cotrain.py)

2. 主配置：
   - [`starvla_cotrain_wallx_qwenpi.yaml`](/diff/wallx_workspace/starVLA/starVLA/config/training/starvla_cotrain_wallx_qwenpi.yaml)

3. 分布式启动配置：
   - [`deepspeed_zero3.yaml`](/diff/wallx_workspace/starVLA/starVLA/config/deepseeds/deepspeed_zero3.yaml)
   - [`zero3.yaml`](/diff/wallx_workspace/starVLA/starVLA/config/deepseeds/zero3.yaml)

4. framework 组装：
   - [`model/framework/__init__.py`](/diff/wallx_workspace/starVLA/starVLA/model/framework/__init__.py)
   - [`model/framework/QwenPI.py`](/diff/wallx_workspace/starVLA/starVLA/model/framework/QwenPI.py)

5. VLM 接口：
   - [`model/modules/vlm/__init__.py`](/diff/wallx_workspace/starVLA/starVLA/model/modules/vlm/__init__.py)
   - [`model/modules/vlm/QWen3.py`](/diff/wallx_workspace/starVLA/starVLA/model/modules/vlm/QWen3.py)

6. 动作头：
   - [`model/modules/action_model/LayerwiseFM_ActionHeader.py`](/diff/wallx_workspace/starVLA/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py)

7. 数据集：
   - [`dataloader/__init__.py`](/diff/wallx_workspace/starVLA/starVLA/dataloader/__init__.py)
   - [`dataloader/wallx_cotrain_datasets.py`](/diff/wallx_workspace/starVLA/starVLA/dataloader/wallx_cotrain_datasets.py)

8. 训练公共工具：
   - [`trainer_utils/trainer_tools.py`](/diff/wallx_workspace/starVLA/starVLA/training/trainer_utils/trainer_tools.py)
   - [`trainer_utils/config_tracker.py`](/diff/wallx_workspace/starVLA/starVLA/training/trainer_utils/config_tracker.py)

## 20. 一句话总结

你这条命令当前真正启动的是一条 `WallX` 专用的三路共训链路：

- `VLA action` 负责动作预测
- `VLM bbox` 负责目标框文本监督
- `VLM signal` 负责继续动作/停止定位判断

但在默认配置下，真正参与总损失的是：

- `action_loss`
- `0.2 * bbox_vlm_loss`

signal 分支虽然会被完整构建、前向和记录日志，但默认不参与梯度更新。
