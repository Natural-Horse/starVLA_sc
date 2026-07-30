# 基于 bbox 的质心估计模型结构

最后更新：2026-07-03。

## 目标

输入部署时可获得的信息：

```text
RGB 图像 + 目标 bbox + object_name
```

输出目标物体在无人机机体系下的三维质心：

```text
[object_body_x, object_body_y, object_body_z]
```

当前训练只使用 VLM grounding 产生有效 bbox 的帧。

## 数据

原始训练读取：

```text
/hdd4/MaTianran/rtc_starvla/data/previous_data_sample50
```

已导出的有效 bbox 数据集：

```text
/hdd4/MaTianran/rtc_starvla/data/previous_data_valid_bbox
```

每条样本包含：

- 640x480 RGB 图像；
- `bbox_xyxy`，原图像素坐标；
- bbox crop；
- `bbox_norm = [x1/640, y1/480, x2/640, y2/480]`；
- `object_name` 对应的 object id；
- 监督目标 `object_body_x/y/z`。

## 输入张量

```text
full_image: [3, 224, 224]，ImageNet normalization
crop_image: [3, 224, 224]，由 bbox 从原图裁剪后缩放
bbox_norm:  [4]
object_id:  int64
```

## 神经网络结构

实现位置：

```text
centroid_estimation/models.py
```

当前默认模型：

```text
ResidualCentroidBBoxRegressor(
  backbone=resnet18,
  pretrained=false,
  shared_backbone=false,
  object_embed_dim=32,
  bbox_embed_dim=64,
  hidden_dim=512,
  output_dim=3
)
```

特征路径：

```text
full_image -> ResNet18 -> 512 维全图特征
crop_image -> ResNet18 -> 512 维局部特征
bbox_norm  -> MLP(4 -> 64 -> 64) -> bbox 特征
object_id  -> Embedding(num_objects -> 32) -> 物体类别特征
拼接       -> MLP(512+512+64+32 -> 512 -> 256 -> 3)
```

当前使用全图和 crop 两套独立 ResNet18。

## Ridge anchor

当前训练模式：

```text
target_mode = residual_bbox_ridge
ridge_alpha = 1.0
```

Ridge 只在训练集拟合，使用部署时也可获得的特征：

```text
bbox_center_x = (x1 + x2) / 2 / 640
bbox_center_y = (y1 + y2) / 2 / 480
bbox_width    = (x2 - x1) / 640
bbox_height   = (y2 - y1) / 480
bbox_area     = bbox_width * bbox_height
bbox_aspect   = bbox_width / max(bbox_height, eps)
object_onehot
```

Ridge 先预测一个 `anchor_xyz`，神经网络预测 residual：

```text
residual_gt = target_xyz - anchor_xyz
pred_xyz = anchor_xyz + residual_pred
```

residual 的均值和标准差只从训练集计算，并保存进 checkpoint。

## 训练参数

本次正式训练：

```text
split: trajectory_half_seed42
target_mode: residual_bbox_ridge
backbone: resnet18
epochs: 50
batch_size: 64
lr: 1e-4
weight_decay: 1e-4
num_workers: 4
GPU: CUDA_VISIBLE_DEVICES=5
```

输出目录：

```text
/hdd4/MaTianran/rtc_starvla/models/centroid/bbox_resnet18_ridge_traj_half
```

## checkpoint 内容

```text
best_model.pt
  model_state_dict
  object_to_id
  target_mode
  target_norm
  anchor.type
  anchor.state
  model_args
```

## 当前测试结果

测试集：trajectory-half split 的 958 个有效 bbox 样本。

```text
mean L2 error:        0.015160867
median L2 error:      0.009281754
mae_x:                0.012059134
mae_y:                0.005902265
mae_z:                0.001677167
anchor-only mean L2:  0.050959248
```

独立评测文件：

```text
/hdd4/MaTianran/rtc_starvla/models/centroid/bbox_resnet18_ridge_traj_half/eval_test.json
```

结论：当前模型在测试集上明显优于只用 Ridge anchor 的结果，但还只跑了 trajectory-half 的单次实验，后续应补 intra-trajectory split 和 direct/class-mean 对照。
