# Qwen Special Token Addition Script

Quickly add new special tokens to Qwen/Qwen2.5-VL-3B-Instruct (or compatible models) and save them to a locally loadable directory.

## 运行

```bash


source_model_id=/diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct
target_model_id=/diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct-ActionRouter
fast_token_list=starVLA/model/modules/vlm/tools/add_qwen_special_tokens/fast_tokens.txt

python starVLA/model/modules/vlm/tools/add_qwen_special_tokens/build_qwen_action_model.py \
  --model-id ${source_model_id} \
  --fast-tokens-file ${fast_token_list} \
  --save-dir ${target_model_id} \
  --init-strategy normal \
  --overwrite
  
```

The generated ActionRouter model contains:

```text
<robot_action_0> ... <robot_action_2047>
<|pred_action|>
<|pred_bbox|>
```

To build the subtask-aware variant used by action-route subtask training:

```bash
source_model_id=/diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct
target_model_id=/diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct-ActionRouterSubtask
fast_token_list=starVLA/model/modules/vlm/tools/add_qwen_special_tokens/fast_tokens.txt

python starVLA/model/modules/vlm/tools/add_qwen_special_tokens/build_qwen_action_model.py \
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

This variant contains the same FAST/router tokens plus:

```text
<|subtask|>
<|end_subtask|>
```

 
## Arguments
 
- --model-id: HF Hub model ID or an existing local model directory
- --save-dir: Output directory
- --fast-tokens-file
- --init-strategy: avg / normal / zero
- --padding-side: left / right
- --device-map: optional Hugging Face device_map, e.g. auto / cuda / cpu

 
## Results
 
The saved directory contains:
 
- config.json / model.safetensors / tokenizer.*
- added_custom_token_id_map.json (records the mapping from custom tokens to IDs)
- action_token_config.json (records FAST token range and router token ids)

 
 
## Load
 
```python
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
tok = AutoTokenizer.from_pretrained("./qwen_vl_with_spatial", trust_remote_code=True)
model = Qwen3VLForConditionalGeneration.from_pretrained("./qwen_vl_with_spatial", dtype="auto")
print(tok.convert_tokens_to_ids("<robot_action_0>"))
print(tok.convert_tokens_to_ids("<|pred_action|>"))
```
