#!/usr/bin/env bash


return 0 2>/dev/null || exit 0


# ==============================================================================
# Router only - formal training
# ==============================================================================

cd /diff/wallx_workspace/starVLA

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=0,1 \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero3.yaml \
  --num_processes 2 \
  --main_process_port 29523 \
  starVLA/training/train_starvla_cotrain_router.py \
  --config_yaml starVLA/config/training/starvla_cotrain_wallx_qwenpi_router.yaml


# ==============================================================================
# Router + SFT - formal training
#
# Default SFT weights in the YAML:
#   gqa = 0.5
#   vg  = 0.5
#   vsi = 0.0
# ==============================================================================

cd /diff/wallx_workspace/starVLA

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=0,1 \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero3.yaml \
  --num_processes 2 \
  --main_process_port 29523 \
  starVLA/training/train_starvla_cotrain_router.py \
  --config_yaml starVLA/config/training/starvla_cotrain_wallx_qwenpi_router_sft.yaml

# ==============================================================================
# Router offline evaluation - batch mode (single checkpoint)
# ==============================================================================

cd /diff/wallx_workspace/starVLA

CUDA_VISIBLE_DEVICES=0 \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/python \
  /diff/wallx_workspace/starVLA/scripts/eval_router_offline_wallx.py \
  --mode batch \
  --config_yaml /diff/wallx_workspace/starVLA/starVLA/config/training/starvla_cotrain_wallx_qwenpi_router.yaml \
  --checkpoint /diff/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_unified_router_small_liangdu_new/checkpoints/steps_15000_pytorch_model.pt \
  --output_dir /diff/wallx_workspace/starVLA/results/router_offline_eval_batch \
  --episode_range 0 50 \
  --frame_stride 1 \
  --max_frames_per_episode 0 \
  --device cuda:0


# ==============================================================================
# Sampled GT bbox web viewer
# ==============================================================================

cd /diff/wallx_workspace

/diff/wallx_workspace/miniconda3/envs/wallx/bin/python \
  /diff/wallx_workspace/data_process/deal_new/10sampled_bbox_web.py \
  --root /diff/wallx_workspace/wallx_data_ckp/datasets/dzb/sampled_ego_v1 \
  --host 0.0.0.0 \
  --port 8000


# ==============================================================================
# 闭环推理
# ==============================================================================

1. Server 端启动模型服务

cd /diff/wallx_workspace/starVLA
# flow
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1  \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/python \
  scripts/serve_wallx_router_policy.py \
  --config_yaml /diff/wallx_workspace/starVLA/starVLA/config/training/starvla_cotrain_wallx_qwenpi_router.yaml \
  --checkpoint /diff/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_flow_no_fast_h24_no_freeze_vlm_bbox_new_prompt/final_model/pytorch_model.pt \
  --host 127.0.0.1 \
  --port 8000 \
  --device cuda:0 \
  --unnormalize_actions \
  --action_decode_mode flow \
  --action_horizon 24


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

# fast
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/python \
  scripts/serve_wallx_router_policy.py \
  --config_yaml /diff/wallx_workspace/starVLA/starVLA/config/training/starvla_cotrain_wallx_qwenpi_router.yaml \
  --checkpoint /diff/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_fast_default_h24_bbox_new_prompt/checkpoints/steps_10000_pytorch_model.pt \
  --action_decode_mode fast \
  --base_vlm /diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct-ActionRouter \
  --special_tokens_policy strict \
  --require_fast_action_tokens \
  --action_horizon 24 \
  --fast_tokenizer_path /diff/wallx_workspace/wall-x/fast-tokenizer \
  --fast_token_count 2048 \
  --fast_max_new_tokens 256 \
  --fast_decode_attempts 3 \
  --fast_debug_dir /tmp/starvla_fast_debug \
  --host 127.0.0.1 \
  --port 8001 \
  --device cuda:0 \
  --unnormalize_actions

  如果用 custom tokenizer 的 FAST checkpoint，把两处换掉：

- --checkpoint ...qwenpi_wallx_fast_custom_tokenizer_h24/...
- --fast_tokenizer_path /diff/wallx_workspace/wallx_data_ckp/checkpoints/fast_tokenizers/wallx_ego_h24_v2048

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1 \
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
  --port 10000 \
  --device cuda:0 \
  --unnormalize_actions


另开一个 server 端终端检查：

curl -i http://127.0.0.1:8000/healthz

正常应该返回：

HTTP/1.1 200 OK
...
ok

2. Client 端建立 SSH tunnel

在仿真机器上开一个终端，保持不要关：

ssh -N -L 127.0.0.1:18000:127.0.0.1:8000 ubuntu@1.13.198.68

再开另一个 client 终端检查 tunnel：

curl -i http://127.0.0.1:18000/healthz

正常也应该返回 ok。

3. Client 端启动 Isaac Sim + EGO

在仿真机器上：

cd /home/yuxuanxu/桌面/mtr/IsaacSimCodebase/IsaacDataCollect/data_collect

ENABLE_REAL_MESH_COLLISION=0 \
EGO_VIRTUAL_CEIL=12 \
EGO_VIRTUAL_GROUND=-2.8 \
EGO_VISUAL_INFLAT_MAP_HEIGHT=12 \
SCENE_USDA_OVERRIDE=dataset/big_house/yingzhouyuan_delete_float/yingzhouyuan_delete_float.usda \
FLYGRIPPER_EGO_BRANCH=1 \
EGO_OBS_MODE=cloud \
SCENE_GAUSS_ROTATE=0.0,0.0,0.0 \
VISUALFM_DATA_CAMERA_WIDTH=640 \
VISUALFM_DATA_CAMERA_HEIGHT=480 \
VISUALFM_ACTION_MODE=ego_odom_follow \
VISUALFM_EGO_ODOM_TOPIC=/drone_0_visual_slam/odom \
PUB_YAW_ODOM_TOPIC=/drone_0_flygripper/odom \
VISUALFM_EGO_FOLLOW_LOOKAHEAD=0.12 \
VISUALFM_EGO_FOLLOW_TRAIL_WARMUP_SEC=2.5 \
GOAL_SEND_INTERFACE=local_goal \
EGO_GOAL_LOOK_FORWARD=1 \
PUB_YAW_INTERFACE=local_goal \
PLY_MAX_POINTS=2000000 \
PLY_POST_READY_DELAY=5 \
EGO_VISUAL_INFLAT_MAP_WINDOW_Z=2 \
EGO_OBSTACLES_INFLATION=0.2 \
EGO_OBSTACLE_CLEARANCE=0.25 \
EGO_MAX_VEL=0.08 \
EGO_MAX_ACC=0.05 \
EGO_MAX_JER=0.05 \
EGO_MAX_SNA=0.05 \
EGO_YAW_RATE_LIMIT=0.3 \
EGO_YAW_ACCEL_LIMIT=0.1 \
VISUALFM_EGO_YAW_RATE_LIMIT=0.1 \
VISUALFM_EGO_YAW_ACCEL_LIMIT=0.05 \
VISUALFM_ACTION_ACCEL_SCALE=0.5 \
VISUALFM_EGO_FOLLOW_BRAKE_ACCEL=0.08 \
STARVLA_POLICY_EGO_PARAMS="-0.386134 -2.123701 0.348953 0.0 0.0 0.0 0.848173730" \
./launch_starvla_policy_sim.sh --rviz

这里 STARVLA_POLICY_EGO_PARAMS 是：

x y z x_range y_range z_range init_yaw_rad

4. Client 端检查 ROS 话题

另开一个 client 终端：

cd /home/yuxuanxu/桌面/mtr/IsaacSimCodebase/IsaacDataCollect/data_collect
source ros1_env.sh

rostopic hz /flygripper_camera/image/rgb/raw
rostopic hz /drone_0_flygripper/odom

确认 RGB 和 odom 都有消息之后，再启动 policy client。

5. Client 端启动 StarVLA policy client

cd /home/yuxuanxu/桌面/mtr/IsaacSimCodebase/IsaacDataCollect/data_collect
source ros1_env.sh

STARVLA_SERVER_URI=ws://127.0.0.1:18000 \
STARVLA_SERVER_TIMEOUT_S=20 \
STARVLA_RECONNECT_DELAY_S=2 \
STARVLA_RGB_TOPIC=/flygripper_camera/image/rgb/raw \
STARVLA_ODOM_TOPIC=/drone_0_flygripper/odom \
STARVLA_OBS_TIMEOUT_S=2 \
STARVLA_LOCAL_GOAL_TOPIC=/drone_0_ego_planner_node/local_goal \
STARVLA_PUBLISH_UAV_ACTIONS=0 \
STARVLA_TASK_PROMPT="" \
STARVLA_INSTRUCTION="Pick up the drink bottle on the desk on your front left, turn to your back and fly to the kitchen, then turn to your left and place the drink bottle on the
stool." \
STARVLA_TARGET_NAME="drink bottle" \
STARVLA_OPERATION=grasp \
STARVLA_SEND_STATE=0 \
STARVLA_MAX_HISTORY=2 \
STARVLA_HORIZON_POINTS=16 \
STARVLA_GOAL_POS_TOLERANCE_M=0.10 \
STARVLA_GOAL_YAW_TOLERANCE_RAD=0.087266463 \
STARVLA_GOAL_TIMEOUT_S=0 \
STARVLA_GOAL_WAIT_POLL_S=0.10 \
STARVLA_WAIT_USER_BETWEEN_HORIZONS=1 \
STARVLA_KEYFRAME_YAW_THRESHOLD_DEG=35 \
STARVLA_OBS_SAVE_DIR="/home/yuxuanxu/桌面/mtr/IsaacSimCodebase/infer_vi/1" \
STARVLA_DELETE_ON_BBOX=1 \
STARVLA_DELETE_ON_USER_INPUT=1 \
STARVLA_DELETE_ACTION=hide \
STARVLA_DELETE_WAIT=1 \
STARVLA_DELETE_TIMEOUT_S=60 \
STARVLA_SCENE_OBJECT_COMMAND_FILE=/home/yuxuanxu/桌面/mtr/IsaacSimCodebase/IsaacDataCollect/data_collect/.scene_object_command.json \
STARVLA_SCENE_OBJECT_COMMAND_STATE_FILE=/home/yuxuanxu/桌面/mtr/IsaacSimCodebase/IsaacDataCollect/data_collect/.scene_object_command_state.json \
STARVLA_ALLOW_NORMALIZED_ACTION=0 \
STARVLA_DISABLE_WS_PROXY=1 \
STARVLA_GOAL_REPEAT_COUNT=1 \
STARVLA_GOAL_REPEAT_INTERVAL=0.05 \
EGO_GOAL_LOOK_FORWARD=1 \
VISUALFM_GOAL_EVENT_FILE=/home/yuxuanxu/桌面/mtr/IsaacSimCodebase/IsaacDataCollect/data_collect/.flygripper_goal_event.json \
python3 starvla_policy_client.py

cd /home/yuxuanxu/桌面/mtr/IsaacSimCodebase/IsaacDataCollect/data_collect
source ros1_env.sh
./starvla_return_home.sh -0.386134 -2.123701 0.348953 0.848173730

# ==============================================================================
# wandb同步
# ==============================================================================

wandb sync --entity yuxuanxu-xyx-beijing-jiaotong-university \
--project qwenpi_wallx_fast_default_h24_bbox_subtask_new_data \
/diff/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_fast_default_h24_bbox_subtask_new_data/wandb/wandb/latest-run


wandb sync --entity yuxuanxu-xyx-beijing-jiaotong-university \
--project qwenpi_wallx_fast_custom_tokenizer_h24_bbox \
/diff/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_fast_default_h24_bbox_new_prompt/wandb/wandb/latest-run

wandb sync --entity yuxuanxu-xyx-beijing-jiaotong-university \
--project qwenpi_wallx_flow_no_fast_h24_bbox_subtask \
/diff/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_flow_no_fast_h24_bbox_subtask/wandb/wandb/latest-run

# ==============================================================================
# 生成QWen Action
# ==============================================================================

source_model_id=/diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct
target_model_id=/diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct-ActionRouter
fast_token_list=starVLA/model/modules/vlm/tools/add_qwen_special_tokens/fast_tokens.txt

python starVLA/model/modules/vlm/tools/add_qwen_special_tokens/build_qwen_action_model.py \
  --model-id "${source_model_id}" \
  --fast-tokens-file "${fast_token_list}" \
  --save-dir "${target_model_id}" \
  --init-strategy normal \
  --device-map auto \
  --overwrite

# ==============================================================================
# stage 1/2训练
# ==============================================================================

 1. 默认 FAST tokenizer 训练，horizon=24

  这个是用已有的 /diff/wallx_workspace/wall-x/fast-tokenizer，训练 VLM 预测 <robot_action_i>。


# ==============================================================================

  2. 训练自己的 FAST tokenizer，然后做 FAST 训练

  先训练 tokenizer：

  /diff/wallx_workspace/miniconda3/envs/starvla/bin/python \
    starVLA/model/modules/action_model/tools/train_fast_tokenizer_wallx.py \
    --dataset-root /diff/wallx_workspace/wallx_data_ckp/datasets/dzb/lerobot_ego_data \
    --source-tokenizer-dir /diff/wallx_workspace/wall-x/fast-tokenizer \
    --save-dir /diff/wallx_workspace/wallx_data_ckp/checkpoints/fast_tokenizers/wallx_ego_h24_v2048 \
    --action-horizon 24 \
    --action-dim 6 \
    --vocab-size 2048 \
    --train-ratio 0.9 \
    --action-routes-only \
    --action-in-ego \
    --no-use-delta-action \
    --normalize-action \
    --truncate-keyframe-value 2 \
    --pad-tail repeat \
    --overwrite

  然后用这个 tokenizer 做 FAST 训练：

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  WANDB_MODE=offline \
  CUDA_VISIBLE_DEVICES=0,1 \
  /diff/wallx_workspace/miniconda3/envs/starvla/bin/accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero3.yaml \
    --num_processes 2 \
    --main_process_port 29524 \
    starVLA/training/train_starvla_cotrain_router.py \
    --config_yaml starVLA/config/training/starvla_cotrain_wallx_qwenpi_router.yaml \
    --run_id qwenpi_wallx_fast_custom_tokenizer_h24_bbox \
    --framework.qwenvl.base_vlm /diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct-ActionRouter \
    --framework.qwenvl.special_tokens.policy strict \
    --framework.qwenvl.special_tokens.require_fast_action_tokens true \
    --framework.router.action_supervision fast_token_ce \
    --framework.action_tokenizer.path /diff/wallx_workspace/wallx_data_ckp/checkpoints/fast_tokenizers/wallx_ego_h24_v2048 \
    --framework.action_tokenizer.token_count 2048 \
    --framework.qwenvl.freeze false \
    --framework.action_model.freeze true \
    --datasets.router_data.action_horizon 24 \
    --framework.action_model.action_horizon 24 \
    --framework.action_model.future_action_window_size 23 \
    --datasets.router_data.per_device_batch_size 12

  注意：这里 vocab-size=2048，所以不用重新生成 Qwen ActionRouter。
  如果你之后改成 4096 / 8192，就必须重新生成 Qwen ActionRouter，把 <robot_action_0>... 的数量也扩到对应大小。

# ==============================================================================

  3. 原始 flow matching 训练，不做 FAST，不冻结 VLM

  这个就是你原来的逻辑：router CE + flow matching action expert，VLM 和 action expert 都训练。

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=0,1 \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero3.yaml \
  --num_processes 2 \
  --main_process_port 29525 \
  starVLA/training/train_starvla_cotrain_router.py \
  --config_yaml starVLA/config/training/starvla_cotrain_wallx_qwenpi_router.yaml \
  --run_id qwenpi_wallx_flow_no_fast_h24_no_freeze_vlm_bbox_new_prompt \
  --framework.qwenvl.base_vlm /diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct \
  --framework.qwenvl.special_tokens.policy auto_add \
  --framework.qwenvl.special_tokens.require_fast_action_tokens false \
  --framework.router.action_supervision flow_matching \
  --framework.qwenvl.freeze false \
  --framework.action_model.freeze false \
  --datasets.router_data.action_horizon 24 \
  --framework.action_model.action_horizon 24 \
  --framework.action_model.future_action_window_size 23 \
  --datasets.router_data.per_device_batch_size 4

# ==============================================================================

# 在FAST的基础上跑flow

CUDA_VISIBLE_DEVICES=0,1 \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero3.yaml \
  --num_processes 2 \
  --main_process_port 29534 \
  starVLA/training/train_starvla_cotrain_router.py \
  --config_yaml starVLA/config/training/starvla_cotrain_wallx_qwenpi_router.yaml \
  --run_id qwenpi_wallx_stage2_flow_from_fast_custom_tokenizer_h24_bbox_detach_action_grad \
  --framework.qwenvl.base_vlm /diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct-ActionRouter \
  --framework.qwenvl.special_tokens.policy strict \
  --framework.qwenvl.special_tokens.require_fast_action_tokens true \
  --framework.router.action_supervision flow_matching \
  --framework.router.action_loss_grad_to_vlm false \
  --framework.action_tokenizer.path /diff/wallx_workspace/wallx_data_ckp/checkpoints/fast_tokenizers/wallx_ego_h24_v2048 \
  --framework.action_tokenizer.token_count 2048 \
  --trainer.pretrained_checkpoint /diff/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_fast_custom_tokenizer_h24_bbox/checkpoints/steps_10000_pytorch_model.pt \
  --trainer.reload_modules qwen_vl_interface \
  --framework.qwenvl.freeze false \
  --framework.action_model.freeze false \
  --datasets.router_data.action_horizon 24 \
  --framework.action_model.action_horizon 24 \
  --framework.action_model.future_action_window_size 23 \
  --datasets.router_data.per_device_batch_size 4

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=0,2 \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero3.yaml \
  --num_processes 2 \
  --main_process_port 29577 \
  starVLA/training/train_starvla_cotrain_router.py \
  --config_yaml starVLA/config/training/starvla_cotrain_wallx_qwenpi_router.yaml \
  --run_id qwenpi_wallx_fast_default_h24_bbox_new_prompt_no_detach_action_grad \
  --framework.qwenvl.base_vlm /diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct-ActionRouter \
  --framework.qwenvl.special_tokens.policy strict \
  --framework.qwenvl.special_tokens.require_fast_action_tokens true \
  --framework.router.action_supervision flow_matching \
  --framework.router.action_loss_grad_to_vlm true \
  --framework.action_tokenizer.path /diff/wallx_workspace/wall-x/fast-tokenizer \
  --framework.action_tokenizer.token_count 2048 \
  --trainer.pretrained_checkpoint /diff/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_fast_default_h24_bbox_new_prompt/checkpoints/steps_10000_pytorch_model.pt \
  --trainer.reload_modules qwen_vl_interface \
  --framework.qwenvl.freeze false \
  --framework.action_model.freeze false \
  --datasets.router_data.action_horizon 24 \
  --framework.action_model.action_horizon 24 \
  --framework.action_model.future_action_window_size 23 \
  --datasets.router_data.per_device_batch_size 4


# ===============================================
# FAST + VLM SFT co-train

  cd /diff/wallx_workspace/starVLA

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=2,3 \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero3.yaml \
  --num_processes 2 \
  --main_process_port 29523 \
  starVLA/training/train_starvla_cotrain_router.py \
  --config_yaml starVLA/config/training/starvla_cotrain_wallx_qwenpi_router_sft.yaml \
  --run_id qwenpi_wallx_fast_default_h24_bbox_sft_liangdu \
  --framework.qwenvl.base_vlm /diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct-ActionRouter \
  --framework.qwenvl.special_tokens.policy strict \
  --framework.qwenvl.special_tokens.require_fast_action_tokens true \
  --framework.router.action_supervision fast_token_ce \
  --framework.action_tokenizer.path /diff/wallx_workspace/wall-x/fast-tokenizer \
  --framework.action_tokenizer.token_count 2048 \
  --framework.qwenvl.freeze false \
  --framework.action_model.freeze true \
  --datasets.router_data.action_horizon 24 \
  --framework.action_model.action_horizon 24 \
  --framework.action_model.future_action_window_size 23 \
  --datasets.router_data.per_device_batch_size 12

# ===============================================
# subtask

  Stage 1：FAST + subtask router 预训练 VLM

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

  Stage 2：基于 FAST VLM 继续训练 flow，不截断 action loss 到 VLM 的梯度

  把 steps_XXXX_pytorch_model.pt 换成 stage 1 的实际 checkpoint。

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


无 FAST 预训练：直接跑 flow + subtask router

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

/diff/wallx_workspace/miniconda3/envs/starvla/bin/python \
  /diff/wallx_workspace/starVLA/scripts/plot_wandb_metric.py \
  /diff/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_flow_no_fast_h24_no_freeze_vlm_bbox/wandb/wandb/latest-run \
  vlm_loss_raw \
  --no-log-scale

/diff/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_fast_custom_tokenizer_h24_bbox/wandb/wandb/latest-run

/diff/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_fast_custom_tokenizer_h24/wandb/wandb/latest-run

/diff/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_flow_no_fast_h24_no_freeze_vlm_bbox/wandb/wandb/latest-run