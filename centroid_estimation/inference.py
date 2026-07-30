from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .anchors import ClassMeanAnchor, IdentityAnchor, RidgeAnchor
from .models import build_model


def load_checkpoint(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    target_mode = checkpoint["target_mode"]
    model_args = dict(checkpoint["model_args"])
    model_args["pretrained"] = False
    model = build_model(target_mode, **model_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    anchor_payload = checkpoint.get("anchor", {})
    anchor_type = anchor_payload.get("type", target_mode)
    anchor_state = anchor_payload.get("state", {})
    if anchor_type == "direct":
        anchor = IdentityAnchor()
    elif anchor_type == "residual_class_mean":
        anchor = ClassMeanAnchor.from_state_dict(anchor_state)
    elif anchor_type == "residual_bbox_ridge":
        anchor = RidgeAnchor.from_state_dict(anchor_state)
    else:
        raise ValueError(f"Unsupported anchor type: {anchor_type}")
    return model, anchor, checkpoint


def undo_norm(values: np.ndarray, norm: dict[str, list[float]]) -> np.ndarray:
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    return values * std + mean


def predict_batch(
    model: torch.nn.Module,
    anchor: Any,
    checkpoint: dict[str, Any],
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    full = batch["full_image"].to(device)
    crop = batch["crop_image"].to(device)
    bbox = batch["bbox_norm"].to(device)
    obj = batch["object_id"].to(device)
    with torch.inference_mode():
        out_norm = model(full, crop, bbox, obj).detach().cpu().numpy()
    pred_delta_or_xyz = undo_norm(out_norm, checkpoint["target_norm"])
    target_mode = checkpoint["target_mode"]
    anchors = []
    for idx in range(len(batch["key"])):
        record = {
            "object_name": batch["object_name"][idx],
            "bbox_norm": batch["bbox_norm"][idx].detach().cpu().numpy(),
        }
        anchors.append(anchor.predict_record(record))
    anchor_xyz = np.stack(anchors, axis=0).astype(np.float32)
    if target_mode == "direct":
        return pred_delta_or_xyz.astype(np.float32), np.zeros_like(pred_delta_or_xyz, dtype=np.float32)
    return (anchor_xyz + pred_delta_or_xyz).astype(np.float32), anchor_xyz

