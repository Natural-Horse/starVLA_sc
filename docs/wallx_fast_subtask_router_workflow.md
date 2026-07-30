# WallX FAST and Subtask Router Workflow

This document records the current FAST / subtask-router path added after the
original unified-router work. For the older non-subtask router, start from
`wallx_router_current_workflow.md`.

## 1. Current Variants

There are now two router answer formats:

```text
token:
  <|pred_action|>
  <|pred_bbox|><point>[x1, y1, x2, y2]</point>

route_subtask:
  <|pred_action|><|subtask|>{subtask_text}<|end_subtask|>
  <|pred_bbox|><point>[x1, y1, x2, y2]</point>
```

FAST training appends action tokens after the action route:

```text
<|pred_action|><|subtask|>{subtask_text}<|end_subtask|><robot_action_...>...
```

Flow training uses the action route prefix as conditioning for the action
expert hidden states instead of generating action tokens.

## 2. Main Files

```text
Subtask YAML:
  starVLA/config/training/starvla_cotrain_wallx_qwenpi_router_subtask.yaml

Main training entry:
  starVLA/training/train_starvla_cotrain_router.py

Router dataset:
  starVLA/dataloader/wallx_cotrain_datasets.py

QwenPI route / action APIs:
  starVLA/model/framework/QwenPI.py

FAST action tokenizer utilities:
  starVLA/model/modules/action_model/fast_ActionHeader.py
  starVLA/model/modules/action_model/tools/train_fast_tokenizer_wallx.py

Qwen special-token builder:
  starVLA/model/modules/vlm/tools/add_qwen_special_tokens/build_qwen_action_model.py

Online server:
  scripts/serve_wallx_router_policy.py

Offline eval:
  scripts/eval_router_offline_wallx.py
```

## 3. Model Bases

The normal FAST router base contains:

```text
/diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct-ActionRouter
```

It adds:

```text
<robot_action_0> ... <robot_action_2047>
<|pred_action|>
<|pred_bbox|>
```

The subtask-aware base contains:

```text
/diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct-ActionRouterSubtask
```

It adds:

```text
<robot_action_0> ... <robot_action_2047>
<|pred_action|>
<|pred_bbox|>
<|subtask|>
<|end_subtask|>
```

Build command:

```bash
cd /diff/wallx_workspace/starVLA

source_model_id=/diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct
target_model_id=/diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct-ActionRouterSubtask
fast_token_list=starVLA/model/modules/vlm/tools/add_qwen_special_tokens/fast_tokens.txt

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/python \
  starVLA/model/modules/vlm/tools/add_qwen_special_tokens/build_qwen_action_model.py \
  --model-id "${source_model_id}" \
  --fast-tokens-file "${fast_token_list}" \
  --save-dir "${target_model_id}" \
  --router-token "<|pred_action|>" \
  --router-token "<|pred_bbox|>" \
  --router-token "<|subtask|>" \
  --router-token "<|end_subtask|>" \
  --init-strategy normal \
  --device-map auto \
  --overwrite
```

## 4. Dataset Requirements

The non-subtask router uses:

```text
/diff/wallx_workspace/wallx_data_ckp/datasets/dzb/lerobot_ego_data
```

The subtask router uses:

```text
/diff/wallx_workspace/xyx1/lerobot_ego_data_subtask
```

The subtask dataset must have a parquet column:

```text
subtask_text
```

When `action_route_format: route_subtask` is enabled, action-route samples with
empty `subtask_text` are rejected and resampled. Bbox samples do not require
`subtask_text`.

## 5. Training Modes

### 5.1 Stage 1: FAST VLM Pretraining with Subtask Route

This trains Qwen VLM to emit route + subtask + FAST action tokens. The flow
action expert is frozen.

```bash
cd /diff/wallx_workspace/starVLA

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=0,1 \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero3.yaml \
  --num_processes 2 \
  --main_process_port 29511 \
  starVLA/training/train_starvla_cotrain_router.py \
  --config_yaml starVLA/config/training/starvla_cotrain_wallx_qwenpi_router_subtask.yaml \
  --run_id qwenpi_wallx_fast_default_h24_bbox_subtask_new_data \
  --framework.qwenvl.base_vlm /diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct-ActionRouterSubtask \
  --framework.router.action_supervision fast_token_ce \
  --framework.router.action_route_format route_subtask \
  --framework.action_tokenizer.path /diff/wallx_workspace/wall-x/fast-tokenizer \
  --framework.action_tokenizer.token_count 2048 \
  --framework.qwenvl.freeze false \
  --framework.action_model.freeze true \
  --datasets.router_data.action_horizon 24 \
  --framework.action_model.action_horizon 24 \
  --framework.action_model.future_action_window_size 23 \
  --datasets.router_data.per_device_batch_size 8
```

### 5.2 Stage 2: Flow from FAST VLM

This loads the stage-1 VLM, then trains the flow action expert. With
`action_loss_grad_to_vlm=true`, flow action loss also updates the VLM hidden
states.

```bash
cd /diff/wallx_workspace/starVLA

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=0,1 \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero3.yaml \
  --num_processes 2 \
  --main_process_port 29525 \
  starVLA/training/train_starvla_cotrain_router.py \
  --config_yaml starVLA/config/training/starvla_cotrain_wallx_qwenpi_router_subtask.yaml \
  --run_id qwenpi_wallx_flow_from_fast_default_h24_bbox_subtask_new_data_no_detach_action_grad \
  --framework.qwenvl.base_vlm /diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct-ActionRouterSubtask \
  --framework.router.action_supervision flow_matching \
  --framework.router.action_route_format route_subtask \
  --framework.router.action_loss_grad_to_vlm true \
  --trainer.pretrained_checkpoint /diff/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_fast_default_h24_bbox_subtask_new_data/checkpoints/steps_14000_pytorch_model.pt \
  --trainer.reload_modules qwen_vl_interface \
  --framework.qwenvl.freeze false \
  --framework.action_model.freeze false \
  --datasets.router_data.action_horizon 24 \
  --framework.action_model.action_horizon 24 \
  --framework.action_model.future_action_window_size 23 \
  --datasets.router_data.per_device_batch_size 4
```

### 5.3 Direct Flow without FAST Pretraining

This trains route-subtask VLM CE and flow matching directly from the
subtask-aware base.

```bash
cd /diff/wallx_workspace/starVLA

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=2,3 \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero3.yaml \
  --num_processes 2 \
  --main_process_port 29123 \
  starVLA/training/train_starvla_cotrain_router.py \
  --config_yaml starVLA/config/training/starvla_cotrain_wallx_qwenpi_router_subtask.yaml \
  --run_id qwenpi_wallx_flow_no_fast_h24_bbox_subtask_new_data \
  --framework.qwenvl.base_vlm /diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct-ActionRouterSubtask \
  --framework.router.action_supervision flow_matching \
  --framework.router.action_route_format route_subtask \
  --framework.router.action_loss_grad_to_vlm true \
  --framework.qwenvl.freeze false \
  --framework.action_model.freeze false \
  --datasets.router_data.action_horizon 24 \
  --framework.action_model.action_horizon 24 \
  --framework.action_model.future_action_window_size 23 \
  --datasets.router_data.per_device_batch_size 4
```

## 6. Training Logs

`vlm_loss_raw` is the full assistant-answer CE. For subtask / FAST training it
is no longer only the first route token. The trainer also logs read-only
breakdowns:

```text
router_route_token_ce
router_action_subtask_ce
router_action_route_prefix_ce
router_fast_action_token_ce
router_bbox_text_ce
router_vlm_token_ce
router_token_accuracy
batch_route_action_fraction
batch_route_bbox_fraction
```

`router_token_accuracy` still measures only the first route-token decision.

## 7. Serving

Subtask flow server:

```bash
cd /diff/wallx_workspace/starVLA

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/python \
  scripts/serve_wallx_router_policy.py \
  --config_yaml /diff/wallx_workspace/starVLA/starVLA/config/training/starvla_cotrain_wallx_qwenpi_router_subtask.yaml \
  --checkpoint /diff/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_flow_no_fast_h24_bbox_subtask/final_model/pytorch_model.pt \
  --base_vlm /diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct-ActionRouterSubtask \
  --special_tokens_policy strict \
  --require_fast_action_tokens \
  --action_decode_mode flow \
  --action_route_format route_subtask \
  --route_max_new_tokens 64 \
  --action_horizon 24 \
  --host 127.0.0.1 \
  --port 8000 \
  --device cuda:0 \
  --unnormalize_actions
```

Subtask FAST server:

```bash
cd /diff/wallx_workspace/starVLA

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/python \
  scripts/serve_wallx_router_policy.py \
  --config_yaml /diff/wallx_workspace/starVLA/starVLA/config/training/starvla_cotrain_wallx_qwenpi_router_subtask.yaml \
  --checkpoint /diff/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_fast_default_h24_bbox_subtask_new_data/checkpoints/steps_14000_pytorch_model.pt \
  --base_vlm /diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct-ActionRouterSubtask \
  --special_tokens_policy strict \
  --require_fast_action_tokens \
  --action_decode_mode fast \
  --action_route_format route_subtask \
  --action_horizon 24 \
  --fast_tokenizer_path /diff/wallx_workspace/wall-x/fast-tokenizer \
  --fast_token_count 2048 \
  --fast_max_new_tokens 256 \
  --fast_decode_attempts 3 \
  --fast_debug_dir /tmp/starvla_fast_debug \
  --route_max_new_tokens 64 \
  --host 127.0.0.1 \
  --port 8000 \
  --device cuda:0 \
  --unnormalize_actions
```

When action route is selected, the server response includes:

```python
response["route"]["subtask_text"]
response["route"]["action_route_solution"]
```

The server log also prints:

```text
Predicted action subtask request_id=... subtask='...' route_solution='...'
```

## 8. Client Notes

Use one IsaacDataCollect client for both token and subtask routers. The client
should send structured fields and let the server build the YAML prompt:

```text
STARVLA_TASK_PROMPT=""
STARVLA_INSTRUCTION="..."
STARVLA_GRASP_TARGET_NAME="..."
STARVLA_PLACE_TARGET_NAME="..."
STARVLA_GRASP_SCENE_OBJECT_NAME="..."
STARVLA_OPERATION=grasp
```

Do not set `STARVLA_TASK_PROMPT` for subtask models unless the full prompt
exactly matches the training YAML. A prompt override bypasses server-side YAML
prompt construction.

Interactive controls after each executed horizon:

```text
Enter: request next action
1: hide the grasp scene object
2: show the grasp scene object
```

## 9. Train / Inference Alignment

The route-subtask flow is structurally aligned:

```text
training prompt:
  YAML prompt + images/history

training action assistant target:
  <|pred_action|><|subtask|>{GT subtask_text}<|end_subtask|>

inference action prefix:
  <|pred_action|><|subtask|>{predicted subtask_text}<|end_subtask|>
```

The unavoidable gap is teacher forcing:

```text
training action expert sees GT subtask
inference action expert sees predicted subtask
```

Online history also has a data-distribution gap: training history comes from
dataset keyframes, while closed-loop history comes from big-turn keyframes
saved by the client.
