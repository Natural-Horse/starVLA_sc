# BBox-Conditioned Centroid Estimation

This module predicts `object_body_x/y/z` from deployment-available inputs:

```text
RGB image + visual-grounding bbox + object_name -> drone-body-frame object centroid
```

It is independent from RTC. The bbox preprocessing stage talks to the existing
StarVLA router server (`scripts/serve_wallx_router_policy.py`) using the same
WebSocket/msgpack request format as deployment.

## 1. Annotate BBoxes With VLM Only

The centroid baseline does not need RTC, action decoding, or FAST action tokens.
BBox preprocessing uses only the Qwen-VL router/bbox behavior from the policy
checkpoint. It preserves the same deployment prompt and image formatting, but
does not start the action server.

```bash
cd /hdd4/MaTianran/rtc_starvla/code/starVLA_rtc_dev
source /hdd4/MaTianran/rtc_starvla/envs/mtr_star/bin/activate

CUDA_VISIBLE_DEVICES=5 python centroid_estimation/scripts/annotate_bbox_vlm.py \
  --config_yaml starVLA/config/training/starvla_wallx_rtc_h36_subtas_cuda1.yaml \
  --checkpoint /hdd4/MaTianran/rtc_starvla/models/policies/qwenpi_wallx_fast_default_h36_bbox_subtask_new_data_min/final_model/pytorch_model.pt \
  --base_vlm /hdd4/MaTianran/rtc_starvla/models/base_vlm/Qwen3-VL-4B-Instruct-ActionRouterSubtask \
  --data_root /hdd4/MaTianran/rtc_starvla/data/previous_data_sample50 \
  --metadata_csv /hdd4/MaTianran/rtc_starvla/data/previous_data_sample50/metadata.csv \
  --output_json /hdd4/MaTianran/rtc_starvla/data/previous_data_sample50/bbox_annotations.json \
  --device cuda:0 \
  --route_mode first_token
```

By default this uses the same deployment decision logic:

```text
image + router prompt -> first token route
<|pred_action|>       -> bbox_valid=false
<|pred_bbox|>         -> continue generation and parse <point>[x1,y1,x2,y2]</point>
```

The saved bbox is scaled back to original image coordinates, equivalent to the
server's `response["bbox"]["xyxy_image"]`.

## 2. Prepare Splits

```bash
python centroid_estimation/scripts/prepare_splits.py \
  --data_root /hdd4/MaTianran/rtc_starvla/data/previous_data_sample50 \
  --metadata_csv /hdd4/MaTianran/rtc_starvla/data/previous_data_sample50/metadata.csv \
  --bbox_file /hdd4/MaTianran/rtc_starvla/data/previous_data_sample50/bbox_annotations.json \
  --output_json /hdd4/MaTianran/rtc_starvla/data/previous_data_sample50/splits_intra_traj_half_even.json \
  --split_mode intra_trajectory_half \
  --train_frame_parity even \
  --seed 42
```

## 3. Train Ridge-Residual Model

```bash
python centroid_estimation/scripts/train_centroid.py \
  --data_root /hdd4/MaTianran/rtc_starvla/data/previous_data_sample50 \
  --metadata_csv /hdd4/MaTianran/rtc_starvla/data/previous_data_sample50/metadata.csv \
  --bbox_file /hdd4/MaTianran/rtc_starvla/data/previous_data_sample50/bbox_annotations.json \
  --split_json /hdd4/MaTianran/rtc_starvla/data/previous_data_sample50/splits_intra_traj_half_even.json \
  --output_dir /hdd4/MaTianran/rtc_starvla/models/centroid/bbox_resnet18_ridge_intra_even \
  --target_mode residual_bbox_ridge \
  --epochs 50 \
  --batch_size 64 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --device cuda
```

## 4. Evaluate

```bash
python centroid_estimation/scripts/eval_centroid.py \
  --checkpoint /hdd4/MaTianran/rtc_starvla/models/centroid/bbox_resnet18_ridge_intra_even/best_model.pt \
  --data_root /hdd4/MaTianran/rtc_starvla/data/previous_data_sample50 \
  --metadata_csv /hdd4/MaTianran/rtc_starvla/data/previous_data_sample50/metadata.csv \
  --bbox_file /hdd4/MaTianran/rtc_starvla/data/previous_data_sample50/bbox_annotations.json \
  --split_json /hdd4/MaTianran/rtc_starvla/data/previous_data_sample50/splits_intra_traj_half_even.json \
  --output_json /hdd4/MaTianran/rtc_starvla/models/centroid/bbox_resnet18_ridge_intra_even/eval_test.json \
  --device cuda
```

## 5. Single Prediction

```bash
python centroid_estimation/scripts/predict_centroid.py \
  --checkpoint /hdd4/MaTianran/rtc_starvla/models/centroid/bbox_resnet18_ridge_intra_even/best_model.pt \
  --image /path/to/rgb/000000.jpg \
  --bbox_xyxy x1 y1 x2 y2 \
  --object_name beer_mug \
  --device cuda
```
