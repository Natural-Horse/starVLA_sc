# WallX Unified Router Training Branch

## 1. Branch Scope

Branch:

```text
feature/wallx-unified-router-prompt
```

This branch adds a new WallX training paradigm on top of the previous WallX cotrain work.
The previous WallX docs remain separate:

- `docs/29f97a9_to_7d522b1_wallx_change_analysis.md`
- `docs/train_starvla_cotrain_wallx_command_flow.md`

This document only describes the new unified-router branch.

For the latest practical workflow, dataset conversion notes, offline evaluation
behavior, and the sampled bbox web viewer, also see:

- `docs/wallx_router_current_workflow.md`

## 2. Goal

The old WallX cotrain path trains three dataloaders/tasks in parallel:

```text
VLA action
VLM bbox
VLM signal
```

This branch changes the training format to one prompt and one router-style assistant answer:

```text
image + unified prompt
  -> VLM first chooses a route token
       <|pred_action|> : call action expert
       <|pred_bbox|>   : continue autoregressive bbox text generation
```

The intended inference behavior is:

```text
obs + unified prompt
  -> VLM generates route token
  -> if <|pred_action|>: action expert predicts continuous actions
  -> if <|pred_bbox|>: VLM continues generating <point>[x1, y1, x2, y2]</point>
```

## 3. Main Files

New or changed files:

- `starVLA/config/training/starvla_cotrain_wallx_qwenpi_router.yaml`
- `starVLA/training/train_starvla_cotrain_router.py`
- `starVLA/dataloader/wallx_cotrain_datasets.py`
- `starVLA/dataloader/__init__.py`
- `starVLA/model/framework/QwenPI.py`
- `starVLA/model/modules/vlm/QWen3.py`
- `starVLA/model/modules/vlm/QWen2_5.py`

The old training scripts are intentionally left in place.

## 4. Dataset Format

The new dataset class is:

```text
WallXRouterDataset
```

It is registered through:

```yaml
datasets:
  router_data:
    dataset_py: wallx_router_dataset
```

Every sample uses the same prompt template:

```text
{instruction}
You are performing a drone navigation task. The first image is the current front view;
any following images are previous keyframes for context. Target: {target_name}.
Operation: {operation}. Decide what to do next. If you are not close enough to the target,
output exactly {pred_action_token}. If you are close enough to the target, output
{pred_bbox_token}<point>[x1, y1, x2, y2]</point> using coordinates in the current front view.
Do not output any other text.
```

For `pred_signal == <pred_action>`, the dataset returns:

```python
{
    "image": images,
    "lang": unified_prompt,
    "route": "action",
    "route_token": "<|pred_action|>",
    "solution": "<|pred_action|>",
    "action": action_array,
}
```

For `pred_signal == <stop>`, the dataset returns:

```python
{
    "image": images,
    "lang": unified_prompt,
    "route": "bbox",
    "route_token": "<|pred_bbox|>",
    "solution": "<|pred_bbox|><point>[x1, y1, x2, y2]</point>",
    "bbox_solution": "<point>[x1, y1, x2, y2]</point>",
}
```

Invalid bbox stop samples are resampled.

## 5. Training Pipeline

Launch example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero3.yaml \
  --num_processes 4 \
  --main_process_port 29523 \
  starVLA/training/train_starvla_cotrain_router.py \
  --config_yaml starVLA/config/training/starvla_cotrain_wallx_qwenpi_router.yaml
```

High-level flow:

```text
accelerate launch
  -> load router YAML
  -> build QwenPI
  -> build WallXRouterDataset
  -> optional contiguous episode split
  -> VLARouterTrainer.prepare_training()
  -> train loop
```

One training step:

```text
batch = list[router samples]

1. Build Qwen chat inputs with solutions.
   - action route solution: <|pred_action|>
   - bbox route solution: <|pred_bbox|><point>[...]</point>

2. Run Qwen once for the whole batch:
   qwen_output = qwen_vl_interface(..., output_hidden_states=True)

3. Compute VLM autoregressive loss:
   vlm_loss = CE over assistant answer span

4. Select only route == action samples.

5. Reuse the same Qwen hidden states from step 2.

6. Feed selected hidden states into action expert:
   action_loss = action_model(hidden_states_for_action_samples, action_labels)

7. Backward:
   total_loss = loss_scale.vlm * vlm_loss + loss_scale.action * action_loss

8. Optimizer step, scheduler step, log, eval, save.
```

The first implementation intentionally avoids a second Qwen forward during training.
VLM CE loss and action expert loss share one Qwen forward.

## 6. Losses

Current config:

```yaml
trainer:
  loss_scale:
    vlm: 1.0
    action: 1.0
```

So the training objective is:

```text
total_loss = vlm_loss + action_loss
```

`vlm_loss` trains both route-token selection and bbox continuation:

```text
action samples: <|pred_action|>
bbox samples:   <|pred_bbox|><point>[...]</point>
```

`action_loss` only applies to action-route samples.

## 7. Router Tokens

The config defines:

```yaml
pred_action_token: "<|pred_action|>"
pred_bbox_token: "<|pred_bbox|>"
```

`QWen3.py` and `QWen2_5.py` collect these tokens from `datasets.router_data` and add missing tokens as tokenizer special tokens.

This matters because the router design assumes the first generated decision is one route token, not a long sequence of ordinary sub-tokens.

## 8. QwenPI Router-Aware Action Interface

`QwenPI.py` adds:

```python
action_loss_from_hidden_states(hidden_states, examples, indices=None)
```

This computes action expert loss from already-computed Qwen hidden states.

The router trainer uses it like this:

```text
whole batch Qwen forward
  -> hidden_states for all samples
  -> select action sample indices
  -> action_loss_from_hidden_states(...)
```

This keeps training efficient and keeps action expert conditioning aligned with the assistant route answer.

`QwenPI.py` also adds:

```python
forward_action_with_route(...)
predict_action_with_route_token(...)
predict_route(...)
predict_router(...)
```

These are initial inference helpers for the route-token policy.

## 9. Inference Routing

Basic inference flow:

```text
predict_route(examples)
  -> image + unified prompt + assistant_start
  -> one Qwen forward
  -> read first-token logits
  -> compare only <|pred_action|> and <|pred_bbox|>
```

If the first-token router chooses:

```text
<|pred_action|>
```

then:

```text
predict_action_with_route_token(...)
  -> image + unified prompt + assistant <|pred_action|>
  -> Qwen hidden states
  -> action expert
```

If the generated text starts with:

```text
<|pred_bbox|>
```

then `predict_route(..., continue_bbox=True)` appends the selected `<|pred_bbox|>` token to the prompt and continues autoregressive generation for the bbox text:

```text
image + unified prompt + assistant <|pred_bbox|>
  -> continue generating <point>[x1, y1, x2, y2]</point>
```

The first-token router returns route diagnostics:

- `route_action_prob`
- `route_bbox_prob`
- `route_confidence`
- `raw_first_token_text`

If the configured route tokens are not single tokenizer ids, the code falls back to the older generate-and-parse routing path.

The current first-token implementation may still rerun Qwen for the action branch after routing. A later optimization can reuse KV cache:

```text
prefix forward with use_cache=True
  -> choose route token
  -> one-token forward with past_key_values
  -> append route-token hidden state
  -> action expert
```

That optimization is not required for the first training implementation.

## 10. Logging

The router trainer logs:

- `vlm_loss_raw`
- `vlm_loss`
- `action_dit_loss_raw`
- `action_dit_loss`
- `batch_route_action_count`
- `batch_route_bbox_count`
- `batch_route_action_fraction`
- `batch_route_bbox_fraction`
- `router_token_accuracy`
- `router_pred_action_count`
- `router_pred_bbox_count`
- `router_pred_other_count`

These are meant to show whether the model is actually learning to emit:

```text
<|pred_action|>
<|pred_bbox|>
```

as the first route decision.

## 11. Validation Done

Static checks already passed:

```bash
python -m py_compile \
  starVLA/training/train_starvla_cotrain_router.py \
  starVLA/dataloader/wallx_cotrain_datasets.py \
  starVLA/model/framework/QwenPI.py \
  starVLA/model/modules/vlm/QWen3.py \
  starVLA/model/modules/vlm/QWen2_5.py

git diff --check
```

YAML parsing was also checked with OmegaConf.

## 12. Recommended Smoke Test

Before a long run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero3.yaml \
  --num_processes 4 \
  --main_process_port 29523 \
  starVLA/training/train_starvla_cotrain_router.py \
  --config_yaml starVLA/config/training/starvla_cotrain_wallx_qwenpi_router.yaml \
  --trainer.max_train_steps 10 \
  --trainer.save_interval 10 \
  --trainer.eval_interval 5
```
