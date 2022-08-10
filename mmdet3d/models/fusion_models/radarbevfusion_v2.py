from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F
import sys
import numpy as np

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform
)
from mmdet3d.ops import Voxelization
from mmdet3d.models import FUSIONMODELS

from .base import Base3DFusionModel

__all__ = ["RadarBEVFusion_v2"]


@FUSIONMODELS.register_module()
class RadarBEVFusion_v2(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        radar_head: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("radar") is not None:
            self.encoders["radar"] = nn.ModuleDict(
                {
                    "voxelize": Voxelization(**encoders["radar"]["voxelize"]),
                    "radar_voxel_encoder": build_backbone(encoders["radar"]['radar_voxel_encoder']),
                    "radar_middle_encoder": build_backbone(encoders["radar"]['radar_middle_encoder'])
                }
            )
            #self.voxelize_reduce = encoders["radar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        #self.heads = nn.ModuleDict()
        # for name in heads:
        #    if heads[name] is not None:
        #        self.heads[name] = build_head(heads[name])
        if heads is not None:
            self.heads = build_head(heads)
        else:
            self.heads = None

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        img_aug_matrix,
        lidar_aug_matrix,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            img_aug_matrix,
            lidar_aug_matrix
        )
        return x

    def extract_radar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)  # [N, C]
        batch_size = coords[-1, 0] + 1
        voxel_features = self.encoders["radar"]["radar_voxel_encoder"](feats, sizes, coords)  # [P, C]
        x = self.encoders["radar"]["radar_middle_encoder"](voxel_features, coords, batch_size)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            f, c, n = self.encoders["radar"]["voxelize"](res)
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        sizes = torch.cat(sizes, dim=0)

        # if self.voxelize_reduce:
        #    feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
        feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        for sensor in self.encoders:
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    img_aug_matrix,
                    lidar_aug_matrix,
                )
            elif sensor == "radar":
                feature = self.extract_radar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        # add radar-guide
        radar_features = features[1]

        if self.training:
            outputs = {}
            pred_dict = self.heads(x, metas)
            losses = self.heads.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)

            for name, val in losses.items():
                if val.requires_grad:
                    outputs[f"loss/{'type'}/{name}"] = val * self.loss_scale['type']
                else:
                    outputs[f"stats/{'type'}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            pred_dict = self.heads(x, radar_features, metas)
            bboxes = self.heads.get_bboxes(pred_dict, metas)
            for k, (boxes, scores, labels) in enumerate(bboxes):
                outputs[k].update(
                    {
                        "boxes_3d": boxes.to("cpu"),
                        "scores_3d": scores.cpu(),
                        "labels_3d": labels.cpu(),
                    }
                )

            return outputs
