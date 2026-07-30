# StarVLA WallX Docs Index

This directory now contains both:

- **current practical workflow docs**
- **historical analysis docs**

The most useful starting points are listed below.

## Current Practical Workflow

1. [`wallx_router_current_workflow.md`](./wallx_router_current_workflow.md)

   Use this first. It describes the current working setup under:

   - `/diff/wallx_workspace/starVLA`
   - `/diff/wallx_workspace/wallx_data_ckp/datasets/dzb/sampled_ego_v1`
   - `/diff/wallx_workspace/wallx_data_ckp/datasets/dzb/lerobot_ego_data`

   It covers:

   - raw -> sampled -> bbox -> lerobot conversion flow
   - `keyframe / pred_signal / bbox_signal / is_rotate` semantics
   - router training behavior
   - router offline evaluation behavior
   - sampled GT bbox web viewer

2. [`train_router_commands.sh`](./train_router_commands.sh)

   Command notebook for:

   - router training
   - router + SFT training
   - router offline evaluation
   - sampled bbox web viewer

   This file is meant to be copied from, not executed directly.

3. [`wallx_fast_subtask_router_workflow.md`](./wallx_fast_subtask_router_workflow.md)

   Current notes for the newer FAST and subtask-router path. It covers:

   - `ActionRouter` / `ActionRouterSubtask` Qwen base generation
   - FAST stage-1 VLM pretraining
   - stage-2 flow training from a FAST VLM
   - direct flow training without FAST pretraining
   - subtask serving and client alignment notes
   - training loss/log metric interpretation

4. [`wallx_router_subtask_design.md`](./wallx_router_subtask_design.md)

   Design-focused explanation of the router and subtask mechanism. Read this
   when a new engineer or another model needs to understand the implementation
   without prior context. It covers:

   - why route tokens are used instead of a separate classifier
   - why subtask text is carried in the action route prefix
   - training targets for flow and FAST
   - inference control flow and hidden-state alignment
   - expected teacher-forcing gap and practical guardrails

5. [`qwen3_5_router_subtask_dev_log.md`](./qwen3_5_router_subtask_dev_log.md)

   Development log for adding Qwen3.5 support to the FAST / subtask-router
   path. It records:

   - Qwen3.5 wrapper changes
   - dynamic VLM layer/hidden-size handling in QwenPI
   - generated `Qwen3.5-4B-ActionRouterSubtask` base metadata
   - validation checks and training override

## Router Branch Design Notes

6. [`wallx_unified_router_training_branch.md`](./wallx_unified_router_training_branch.md)

   This explains why the unified-router branch was introduced and how the
   prompt / route-token / action-head design works.

## Historical / Background Docs

7. [`train_starvla_cotrain_wallx_command_flow.md`](./train_starvla_cotrain_wallx_command_flow.md)

   Historical deep dive into the older **non-router** `train_starvla_cotrain.py`
   path. Useful when you need to understand the earlier WallX cotrain stack, but
   it is not the current router training entrypoint.

8. [`29f97a9_to_7d522b1_wallx_change_analysis.md`](./29f97a9_to_7d522b1_wallx_change_analysis.md)

   Historical change-analysis document for the commit range that first brought
   WallX into this repo.

## Recommended Reading Order

If you are working on the current WallX router pipeline, read in this order:

1. `wallx_router_current_workflow.md`
2. `wallx_router_subtask_design.md`
3. `wallx_fast_subtask_router_workflow.md`
4. `qwen3_5_router_subtask_dev_log.md` if using Qwen3.5
5. `train_router_commands.sh`
6. `wallx_unified_router_training_branch.md`

Use the two historical docs only when you need background on the pre-router
implementation or the original integration work.
