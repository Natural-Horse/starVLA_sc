# StarVLA Isaac Router Serving

This path is for closed-loop IsaacDataCollect control without the legacy wall-x
client protocol.

## Server

Run on the machine with the StarVLA checkpoint and GPU:

```bash
cd /diff/wallx_workspace/starVLA
CUDA_VISIBLE_DEVICES=0 \
/diff/wallx_workspace/miniconda3/envs/starvla/bin/python \
  scripts/serve_wallx_router_policy.py \
  --config_yaml /diff/wallx_workspace/starVLA/starVLA/config/training/starvla_cotrain_wallx_qwenpi_router.yaml \
  --checkpoint /path/to/steps_N_pytorch_model.pt \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda:0 \
  --unnormalize_actions
```

For a route-subtask checkpoint, use the subtask YAML and base:

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

The server sends metadata on connect, then accepts msgpack dictionaries:

```python
{
    "protocol": "starvla_router_v1",
    "request_id": 1,
    "timestamp": 0.0,
    "task": {
        "instruction": "...",
        "target_name": "target object",
        "operation": "grasp",
        "grasp_done": False,
        # Optional: "prompt": "full router prompt"
    },
    "obs": {
        "image": np.ndarray,      # uint8 HWC RGB
        "history": [np.ndarray], # previous uint8 HWC RGB frames, oldest to newest
        "state": np.ndarray,     # [x, y, z, roll, pitch, yaw], optional
        "state_frame": "world_rpy",
    },
}
```

Action responses use the model's training semantics. For the current router
config this is an unnormalized ego-frame offset horizon:

```python
{
    "protocol": "starvla_router_v1",
    "route": {
        "route": "action",
        "...": "...",
        # route_subtask models also include:
        "subtask_text": "Fly to ...",
        "action_route_solution": "<|pred_action|><|subtask|>...<|end_subtask|>",
    },
    "action": {
        "frame": "ego_delta",
        "normalized": False,
        "horizon": np.ndarray, # [T, 6], [dx, dy, dz, droll, dpitch, dyaw]
        "first": np.ndarray,   # [6]
    },
}
```

If the router chooses bbox, no navigation action is returned:

```python
{
    "route": {"route": "bbox", "...": "..."},
    "action_skipped": True,
    "bbox": {
        "xyxy_model": np.ndarray,
        "xyxy_image": np.ndarray,
    },
}
```

## IsaacDataCollect Client

Run on the simulation machine after ROS, ros1_bridge, EGO, and Isaac camera/odom
topics are alive:

```bash
cd /diff/wallx_workspace/IsaacDataCollect/data_collect
source ros1_env.sh
STARVLA_SERVER_URI=ws://SERVER_IP:8000 \
STARVLA_INSTRUCTION="Catch: object. Put: target." \
STARVLA_GRASP_TARGET_NAME="target object" \
STARVLA_PLACE_TARGET_NAME="placement destination" \
STARVLA_GRASP_SCENE_OBJECT_NAME="target_scene_object_name" \
STARVLA_OPERATION=grasp \
STARVLA_SEND_STATE=0 \
STARVLA_HORIZON_POINTS=1 \
python3 starvla_policy_client.py
```

Leave `STARVLA_TASK_PROMPT` empty for both normal-router and subtask-router
models. The server should construct the exact prompt from the YAML so training
and inference stay aligned.

Default subscriptions:

- RGB: `/flygripper_camera/image/rgb/raw`
- Odom: `/drone_0_flygripper/odom`

Default publication:

- EGO local goal: `/drone_0_ego_planner_node/local_goal`

The client converts `ego_delta` actions back to world-frame EGO local goals,
downsamples each returned action horizon to `STARVLA_HORIZON_POINTS`, executes
those waypoints in order, waits until the final selected waypoint is reached,
then requests the next observation/action.

Useful client controls:

- `STARVLA_HORIZON_POINTS`: number of waypoints sampled from each action horizon. `1` selects the last horizon point; `2` selects the middle and last points. If the value is greater than the horizon length, every horizon point is used.
- `STARVLA_SEND_STATE`: send `[x,y,z,roll,pitch,yaw]` to the server. Default: `0`, matching the current router training config `include_state: false`.
- `STARVLA_GOAL_POS_TOLERANCE_M`: position tolerance for considering a waypoint reached. Default: `0.10`.
- `STARVLA_GOAL_YAW_TOLERANCE_RAD`: yaw tolerance for considering a waypoint reached. Default: `0.20`.
- `STARVLA_GOAL_TIMEOUT_S`: timeout for each selected waypoint. Default: `0` disabled.
- `STARVLA_WAIT_USER_BETWEEN_HORIZONS=1`: after each horizon is fully executed, wait for Enter before collecting the next observation.
- `STARVLA_KEYFRAME_YAW_THRESHOLD_DEG`: if the first-to-last yaw change inside an action horizon exceeds this threshold, save the request's current image as a history keyframe for the next request. Default: `35`.
- `STARVLA_OBS_SAVE_DIR`: if non-empty, save each request observation under this directory.
- `STARVLA_GRASP_TARGET_NAME`: semantic target used in the grasp-phase prompt.
- `STARVLA_PLACE_TARGET_NAME`: semantic destination used after switching to put phase.
- `STARVLA_GRASP_SCENE_OBJECT_NAME`: actual scene object name used for hide/show/pickup commands.
- `STARVLA_DELETE_ON_BBOX`: if the server returns bbox, send a scene object command for `STARVLA_GRASP_SCENE_OBJECT_NAME`. Default: `1`.
- `STARVLA_DELETE_ACTION`: scene command action after bbox. Default: `hide`; `pickup` uses the grasp pipeline if enabled.

When `STARVLA_WAIT_USER_BETWEEN_HORIZONS=1`, the prompt after each action
horizon supports:

```text
Enter: request next action
1: hide the grasp scene object
2: show the grasp scene object
```
