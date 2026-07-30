# WallX Router and Subtask Design

This document explains the design of the WallX router / subtask training and
serving path. It is written for readers without the original development
context.

## 1. Problem Being Solved

WallX closed-loop control has two different outputs:

1. Continue moving the robot or drone by predicting an action trajectory.
2. Stop navigation and localize the grasp target with a bbox.

The model therefore needs a routing decision before the final output:

```text
observation + instruction
  -> route = action
       -> predict action trajectory
  -> route = bbox
       -> predict target bbox
```

The unified router design makes this routing decision a normal VLM generation
target instead of a separate classifier.

## 2. Core Router Format

The VLM assistant answer always starts with one route token:

```text
<|pred_action|>
<|pred_bbox|>
```

The first generated assistant token is used as the route decision.

For bbox route, the VLM continues generating bbox text:

```text
<|pred_bbox|><point>[x1, y1, x2, y2]</point>
```

For action route, the old non-subtask format was only:

```text
<|pred_action|>
```

The action expert then consumes Qwen hidden states conditioned on that route
token.

## 3. Why Add Subtask to the Action Route

The instruction can describe a long task:

```text
Pick up the bottle, turn around, fly to the kitchen, place it on the desk.
```

At any single frame, the desired immediate behavior is shorter:

```text
Search bottle
Fly to kitchen
Fly to round desk
```

The `subtask_text` column in the newer dataset gives this local instruction.
The subtask-router design asks the VLM to predict that local instruction before
the action expert predicts movement.

The new action route becomes:

```text
<|pred_action|><|subtask|>{subtask_text}<|end_subtask|>
```

This keeps routing and subtask prediction in a single VLM generation path. It
avoids a three-call inference pipeline such as:

```text
predict subtask -> predict route -> predict action / bbox
```

Instead, action inference uses:

```text
predict route token
if action: continue generation until <|end_subtask|>
then predict action from the same route-subtask prefix
```

## 4. Special Tokens

There are two Qwen base variants.

Normal action-router base:

```text
Qwen3-VL-4B-Instruct-ActionRouter
```

It contains:

```text
<robot_action_0> ... <robot_action_2047>
<|pred_action|>
<|pred_bbox|>
```

Subtask-aware base:

```text
Qwen3-VL-4B-Instruct-ActionRouterSubtask
```

It contains:

```text
<robot_action_0> ... <robot_action_2047>
<|pred_action|>
<|pred_bbox|>
<|subtask|>
<|end_subtask|>
```

These tokens are added before training with:

```text
starVLA/model/modules/vlm/tools/add_qwen_special_tokens/build_qwen_action_model.py
```

At runtime, use `special_tokens.policy: strict` for models trained from these
expanded bases. Strict mode prevents accidental tokenizer expansion during
training or serving.

## 5. Dataset Contract

The non-subtask router dataset needs the usual WallX LeRobot fields:

```text
image/video
state
action
grasp
pred_signal
bbox
keyframe
task
```

The subtask router additionally requires:

```text
subtask_text
```

The training config points to:

```text
/diff/wallx_workspace/xyx1/lerobot_ego_data_subtask
```

When `action_route_format: route_subtask` is active:

- action samples with empty `subtask_text` are rejected and resampled
- bbox samples do not need `subtask_text`
- `subtask_text` is not inserted into the user prompt as ground truth
- `subtask_text` is used as the assistant target

That last point matters. If the ground-truth subtask were placed in the user
prompt, the model would not learn to predict it and inference would be
mismatched.

## 6. Prompt Design

The prompt is phase-specific.

Grasp phase asks the model to choose between action and bbox:

```text
GRASP PHASE: FIND AND GRASP THE OBJECT.
The current target is the object to be grasped: {target_name}.
{target_name}, {target_name}, {target_name} should be grasped.
If the object is still far away, output <|pred_action|><|subtask|>SUBTASK_TEXT<|end_subtask|>.
SUBTASK_TEXT should be one short Search or Fly to command.
If the object is close enough for grasping, output <|pred_bbox|><point>[x1, y1, x2, y2]</point>.
Do not output anything else.
```

Place phase disables bbox:

```text
PLACE PHASE: GO TO THE PLACEMENT DESTINATION.
The current target is the placement destination: {target_name}.
Go to {target_name}.
The placement destination is {target_name}, {target_name}, {target_name}.
Always output <|pred_action|><|subtask|>SUBTASK_TEXT<|end_subtask|>.
SUBTASK_TEXT should be one short Search or Fly to command.
Do not output a bounding box. Do not describe the scene. Do not output anything else.
```

FAST mode uses almost the same prompt, but adds that action tokens must follow
immediately after `<|end_subtask|>`.

## 7. Training Targets

Training calls Qwen with:

```text
user:
  image(s) + router prompt

assistant:
  solution string
```

The VLM label mask keeps only the assistant answer span. The prompt tokens are
masked with `IGNORE_INDEX`.

### 7.1 Flow Matching Action Route

For action samples:

```text
assistant target:
  <|pred_action|><|subtask|>{GT subtask_text}<|end_subtask|>
```

The VLM CE trains route and subtask text generation. The flow action expert
uses the hidden states from this same Qwen forward. This means the action
expert is conditioned on:

```text
prompt + <|pred_action|><|subtask|>{GT subtask_text}<|end_subtask|>
```

For bbox samples:

```text
assistant target:
  <|pred_bbox|><point>[x1, y1, x2, y2]</point>
```

### 7.2 FAST Action Route

For action samples:

```text
assistant target:
  <|pred_action|><|subtask|>{GT subtask_text}<|end_subtask|><robot_action_...>...
```

The VLM CE trains route, subtask text, and FAST action token generation. The
flow action expert is frozen and not used for action loss in this mode.

## 8. Main Training Switches

Important config fields:

```yaml
framework:
  router:
    action_supervision: flow_matching | fast_token_ce
    action_route_format: token | route_subtask
    subtask_start_token: "<|subtask|>"
    subtask_end_token: "<|end_subtask|>"
    action_loss_grad_to_vlm: true | false
```

Meaning:

- `action_supervision=flow_matching`: train VLM CE plus flow action expert loss.
- `action_supervision=fast_token_ce`: train VLM CE on FAST tokens; freeze action expert.
- `action_route_format=token`: action route is only `<|pred_action|>`.
- `action_route_format=route_subtask`: action route carries subtask text.
- `action_loss_grad_to_vlm=true`: flow action loss also updates VLM hidden states.
- `action_loss_grad_to_vlm=false`: flow action loss updates only the action expert; VLM still gets route/bbox CE.

## 9. Loss and Metrics

`vlm_loss_raw` is the full assistant-answer CE. In subtask or FAST mode it
contains more than the route-token loss.

Read-only breakdown metrics are logged:

```text
router_route_token_ce
router_action_subtask_ce
router_action_route_prefix_ce
router_fast_action_token_ce
router_bbox_text_ce
router_vlm_token_ce
router_token_accuracy
```

`router_token_accuracy` only measures the first assistant token:

```text
<|pred_action|> vs <|pred_bbox|>
```

It does not measure subtask quality or FAST token quality.

## 10. Inference Flow

The server uses the same YAML prompt templates as training unless the client
sends a full `prompt` override.

Recommended client behavior:

```text
send instruction / target_name / operation
do not send full prompt
```

Server route-subtask action flow:

1. Build prompt from YAML.
2. Run Qwen once to score the first assistant token.
3. Pick `<|pred_action|>` or `<|pred_bbox|>`.
4. If bbox, force the bbox token and generate bbox text.
5. If action and `route_subtask`, force the action token and generate until
   `<|end_subtask|>`.
6. Parse:

```text
<|pred_action|><|subtask|>{predicted subtask}<|end_subtask|>
```

7. For flow, run the action expert conditioned on that parsed route prefix.
8. For FAST, force that parsed route prefix and generate FAST action tokens.

The response includes:

```python
response["route"]["subtask_text"]
response["route"]["action_route_solution"]
```

The server log also prints the predicted subtask.

## 11. Training / Inference Alignment

The intended alignment is:

```text
training action prefix:
  <|pred_action|><|subtask|>{GT subtask_text}<|end_subtask|>

inference action prefix:
  <|pred_action|><|subtask|>{predicted subtask_text}<|end_subtask|>
```

This is structurally aligned but has a normal teacher-forcing gap:

```text
training action expert sees GT subtask
inference action expert sees predicted subtask
```

If predicted subtask is wrong, action quality can degrade. This is expected and
should be evaluated separately from first-token router accuracy.

Other possible distribution differences:

- training history frames come from dataset keyframe sampling
- online history frames come from client-side big-turn keyframe accumulation
- training phase labels come from `grasp`
- online phase comes from client `operation` and switches to `put` after object
  hide / bbox handling

## 12. Why Not Use Separate Subtask / Router / Action Prompts

A three-prompt design would require multiple VLM calls:

```text
predict subtask
predict route
predict action or bbox
```

That adds latency and makes hidden-state conditioning harder to keep aligned.

The chosen design keeps route and subtask in one assistant prefix. The action
expert sees the same kind of prefix during both training and inference.

## 13. Practical Guardrails

- Use `ActionRouterSubtask` base for route-subtask checkpoints.
- Use `strict` special-token policy for expanded bases.
- Keep `STARVLA_TASK_PROMPT=""` in the Isaac client unless intentionally
  debugging a full prompt.
- Use different server ports when running multiple checkpoints at once.
- For subtask serving, set:

```bash
--action_route_format route_subtask
--route_max_new_tokens 64
```

- Check server logs for:

```text
Predicted action subtask ...
```

- If the server warns that action route has no parseable subtask, inspect the
generated route text and the loaded YAML/base mismatch first.
