# Go2 Remote Training Smoke

- Goal: run a real one-step Qwen3-VL + sparse waypoint Flow Matching training smoke test on `zju-server` without disturbing other users' GPU processes.
- Status: passed.

## Initialization

- Replaced the unrelated `Qwen3-VL-4B-Instruct-ActionRouterSubtask` checkpoint with the clean official `Qwen/Qwen3-VL-4B-Instruct` base.
- Official Hugging Face revision: `ebb281ec70b05090aa6165b016eac8ec08e71b17`.
- Remote path: `/hdd4/MaTianran/rtc_starvla/models/base_vlm/Qwen3-VL-4B-Instruct`.
- Verified architecture: `Qwen3VLForConditionalGeneration`, 36 text layers, hidden size 2560.
- Verified shard SHA-256:
  - `model-00001-of-00002.safetensors`: `30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9`
  - `model-00002-of-00002.safetensors`: `046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6`
- The clean tokenizer receives eight newly initialized router/subtask tokens through `special_tokens.policy=auto_add`.
- The 16-layer Flow Matching DiT action head is randomly initialized; no action-head weights are inherited from the unrelated task.

## Resources And Dependencies

- GPUs 0-4 were occupied by other users and were not used.
- Tests used only idle RTX 3090 GPUs 5 and 6 with per-device batch size 1.
- Two 24 GiB cards cannot initialize the full optimizer on GPU; ZeRO-3 failed while allocating a 5.55 GiB gradient partition.
- CPU optimizer offload works after installing `ninja` and `modelscope` in `/hdd4/MaTianran/rtc_starvla/envs/mtr_star`.
- DeepSpeed CPUAdam requires `DS_SKIP_CUDA_CHECK=1` because the host toolkit is CUDA 11.8 while PyTorch was built with CUDA 12.4. The CPUAdam extension compiled and loaded successfully.

## Result

The final deterministic NAV smoke used two episodes, no shuffle, one optimization step, no checkpoint save, and W&B disabled. It completed successfully with:

- `action_batch_size=1`
- `vlm_loss=7.706926345825195`
- `action_dit_loss=3396.640625`
- `action_dim_loss/dim_0=6953.0234375`
- `action_dim_loss/dim_1=1267.736328125`
- `action_dim_loss/dim_2=1969.162109375`
- `model_time=112.21487400704063 s`

The high initial action loss is expected from a randomly initialized Flow Matching head. The test covers clean-base loading, dual-camera input, NAV waypoint targets, router loss, Flow Matching forward/backward, and the optimizer update.

Remote log: `results/SmokeTests/go2_waypoint_clean_qwen3vl_nav_smoke_gpu56_0731.log`.

## W&B

The remote environment is not logged in to W&B yet (`api_key=null`). Log in once with `wandb login`, then launch with `WANDB_MODE=online`. The configured project is `starvla_go2_waypoint`; the run name comes from `run_id`. Useful live metrics include `loss`, `vlm_loss`, `action_dit_loss`, `action_dim_loss/*`, `router_token_accuracy`, route counts/fractions, learning rates, `data_time`, and `model_time`.
