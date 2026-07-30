from __future__ import annotations

import torch
from torch import nn


def _build_resnet_backbone(name: str = "resnet18", pretrained: bool = False) -> tuple[nn.Module, int]:
    try:
        from torchvision import models
    except Exception as exc:
        raise RuntimeError("torchvision is required for centroid ResNet backbones.") from exc

    if name == "resnet18":
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        except AttributeError:
            model = models.resnet18(pretrained=pretrained)
        dim = int(model.fc.in_features)
    elif name == "resnet34":
        try:
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            model = models.resnet34(weights=weights)
        except AttributeError:
            model = models.resnet34(pretrained=pretrained)
        dim = int(model.fc.in_features)
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    model.fc = nn.Identity()
    return model, dim


class CentroidBBoxRegressorBase(nn.Module):
    def __init__(
        self,
        *,
        num_objects: int,
        backbone: str = "resnet18",
        pretrained: bool = False,
        object_embed_dim: int = 32,
        bbox_embed_dim: int = 64,
        hidden_dim: int = 512,
        shared_backbone: bool = False,
        output_dim: int = 3,
    ) -> None:
        super().__init__()
        self.shared_backbone = bool(shared_backbone)
        self.full_backbone, feat_dim = _build_resnet_backbone(backbone, pretrained=pretrained)
        if self.shared_backbone:
            self.crop_backbone = self.full_backbone
        else:
            self.crop_backbone, crop_feat_dim = _build_resnet_backbone(backbone, pretrained=pretrained)
            if crop_feat_dim != feat_dim:
                raise RuntimeError("Backbone feature dimensions differ unexpectedly.")
        self.object_embedding = nn.Embedding(int(num_objects), int(object_embed_dim))
        self.bbox_mlp = nn.Sequential(
            nn.Linear(4, bbox_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bbox_embed_dim, bbox_embed_dim),
            nn.ReLU(inplace=True),
        )
        head_in = feat_dim * 2 + object_embed_dim + bbox_embed_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(
        self,
        full_image: torch.Tensor,
        crop_image: torch.Tensor,
        bbox_norm: torch.Tensor,
        object_id: torch.Tensor,
    ) -> torch.Tensor:
        full_feat = self.full_backbone(full_image)
        crop_feat = self.crop_backbone(crop_image)
        bbox_feat = self.bbox_mlp(bbox_norm.float())
        object_feat = self.object_embedding(object_id.long())
        return self.head(torch.cat([full_feat, crop_feat, bbox_feat, object_feat], dim=-1))


class DirectCentroidBBoxRegressor(CentroidBBoxRegressorBase):
    """Directly predicts normalized object_body_x/y/z targets."""


class ResidualCentroidBBoxRegressor(CentroidBBoxRegressorBase):
    """Predicts normalized residual_xyz; caller adds the anchor in raw space."""


def build_model(target_mode: str, **kwargs) -> nn.Module:
    if target_mode == "direct":
        return DirectCentroidBBoxRegressor(**kwargs)
    if target_mode in {"residual_class_mean", "residual_bbox_ridge"}:
        return ResidualCentroidBBoxRegressor(**kwargs)
    raise ValueError(f"Unsupported target_mode: {target_mode}")
