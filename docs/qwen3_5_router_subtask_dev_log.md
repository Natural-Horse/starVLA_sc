# Qwen3.5 Router-Subtask Development Log

Date: 2026-06-03

This log records the Qwen3.5 support work for the WallX FAST / subtask-router
training path.

## Goal

Make the existing QwenPI router-subtask training flow work with local Qwen3.5
vision-language checkpoints, especially:

```text
/diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3.5-4B
```

The target generated base is:

```text
/diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3.5-4B-ActionRouterSubtask
```

## Code Changes

### Qwen3.5 VLM wrapper

File:

```text
starVLA/model/modules/vlm/QWen3_5.py
```

Changes:

- Added config-safe lookup so router configs without `datasets.vla_data` work.
- Added fallback from `datasets.vla_data` to `datasets.router_data` for prompt config.
- Aligned Qwen3.5 input packing with the Qwen3 wrapper:
  - render chat template as text first
  - process text/images/videos through `AutoProcessor`
- Changed supervised labels to keep the full assistant answer span instead of
  starting only at the first FAST action token.
- Skipped the empty Qwen3.5 thinking prefix:

```text
<think>

</think>

```

This keeps the first supervised token aligned with the route token:

```text
<|pred_action|>
<|pred_bbox|>
```

### QwenPI dynamic VLM dimensions

File:

```text
starVLA/model/framework/QwenPI.py
```

Changes:

- Replaced the hard-coded VLM layer count `36` with dynamic config lookup.
- Read hidden size and layer count from `model.config` / `model.config.text_config`.
- Preserved Qwen3-VL-4B behavior because it still resolves to:

```text
hidden_size = 2560
num_hidden_layers = 36
```

Observed local model dimensions:

```text
Qwen3-VL-4B: hidden_size=2560, num_hidden_layers=36
Qwen3-VL-2B: hidden_size=2048, num_hidden_layers=28
Qwen3.5-4B:  hidden_size=2560, num_hidden_layers=32
Qwen3.5-9B:  hidden_size=4096, num_hidden_layers=32
```

## Generated Qwen3.5 ActionRouterSubtask Base

Command:

```bash
cd /diff/wallx_workspace/starVLA

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/python \
  starVLA/model/modules/vlm/tools/add_qwen_special_tokens/build_qwen_action_model.py \
  --model-id /diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3.5-4B \
  --fast-tokens-file starVLA/model/modules/vlm/tools/add_qwen_special_tokens/fast_tokens.txt \
  --save-dir /diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3.5-4B-ActionRouterSubtask \
  --router-token '<|pred_action|>' \
  --router-token '<|pred_bbox|>' \
  --router-token '<|subtask|>' \
  --router-token '<|end_subtask|>' \
  --init-strategy normal \
  --device-map auto \
  --overwrite
```

Generated metadata:

```text
source_model: /diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3.5-4B
model_type: qwen3_5
tokenizer_len_before: 248077
tokenizer_len_after: 250129
embedding_rows_before: 248320
embedding_rows_after: 250129
added_token_count: 2052
action_token_min: 248077
action_token_max: 250124
```

Router token ids:

```text
<|pred_action|>: 250125
<|pred_bbox|>: 250126
<|subtask|>: 250127
<|end_subtask|>: 250128
```

Single-token validation passed for:

```text
<robot_action_0>
<robot_action_2047>
<|pred_action|>
<|pred_bbox|>
<|subtask|>
<|end_subtask|>
```

## Training Usage

The Qwen3.5-4B FAST router-subtask training command only needs to change the
base VLM path:

```bash
--framework.qwenvl.base_vlm /diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3.5-4B-ActionRouterSubtask
```

Keep the existing router-subtask settings:

```bash
--framework.router.action_supervision fast_token_ce
--framework.router.action_route_format route_subtask
--framework.action_tokenizer.token_count 2048
--framework.qwenvl.freeze false
```

Do not reload an old Qwen3-VL `qwen_vl_interface` checkpoint directly into
Qwen3.5. The backbone architecture and parameter names differ.

## Validation Performed

Commands / checks performed:

```text
python -m py_compile starVLA/model/modules/vlm/QWen3_5.py starVLA/model/framework/QwenPI.py
```

Qwen3.5 label construction was checked with dummy image inputs:

```text
action solution labels decode to:
<|pred_action|><|subtask|>Fly to cup<|end_subtask|><robot_action_0>

bbox solution labels decode to:
<|pred_bbox|><point>[1, 2, 3, 4]</point>
```

QwenPI dynamic dimension lookup was checked with mocked configs:

```text
resolved 2560 36
resolved 2048 28
resolved 4096 32
```

## Performance Note

The Qwen3.5 conversion is slower than Qwen3-VL conversion because:

- the local environment falls back to torch implementation for Qwen3.5 linear
  attention materialization
- `resize_token_embeddings` performs Hugging Face's default mean/covariance
  initialization before the script applies its own configured init strategy
- the generated Qwen3.5-4B model is about 8.5 GB

Future optimization: pass `mean_resizing=False` to `resize_token_embeddings`
inside the special-token builder, because the script already initializes the
new token rows explicitly.
