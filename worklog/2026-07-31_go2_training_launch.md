# Go2 Training Launch

- Goal: inspect representative n200 training samples, verify idle GPU ownership, and launch checkpointed training on idle RTX 3090 GPUs if the samples are sound.
- Status: training active on `zju-server`.

## Data Review

- Exported representative dual-camera samples for `nav_to_pick`, `pick`, `nav_to_place`, `place`, and `done` with `scripts/preview_go2_training_samples.py`.
- Dataset contains 130 episodes and 16,170 sampled frames: NAV 8,928 (55.2%), GRASP 2,434 (15.1%), PLACE 3,768 (23.3%), DONE 1,040 (6.4%).
- No RECOVER samples are present; recovery behavior must not be treated as a trained capability or metric for this dataset.
- Visual observations, route targets, subtask text, and NAV waypoint targets were consistent in the reviewed samples.

## Checkpoint Validation

- One-step NAV training and checkpoint save passed on idle GPUs 5 and 6 with ZeRO-3 CPU optimizer offload.
- Test checkpoint: `results/Checkpoints/go2_waypoint_checkpoint_smoke_gpu56_0731/checkpoints/steps_1_pytorch_model.pt`.
- Checkpoint size: 11,920,479,530 bytes; save overhead was about 87 seconds.

## Active Training

- tmux session: `go2_n200_train`
- run ID: `go2_waypoint_n200_0731_save300`
- GPUs: idle RTX 3090 cards 5 and 6 only; occupied A40 cards were not touched.
- Steps: 1,000; global batch size: 2; W&B mode: offline.
- Evaluation interval: 100 steps; checkpoint interval: 300 steps.
- Estimated average: about 109 seconds/step including amortized evaluation/checkpoint overhead; first checkpoint is expected after about 9.1 hours.
- Log: `results/TrainingLogs/go2_waypoint_n200_0731_save300.log`.
