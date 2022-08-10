_base_ = [
    '../_base_/datasets/bev-radar-nus.py',
    '../_base_/default_runtime.py'
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
image_size = [256, 704]
voxel_size = [0.1, 0.1, 8.0]

model = dict(
    type='RadarBEVFusion',
    encoders=dict(
        camera=dict(
            backbone=dict(
                type='SwinTransformer',
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                patch_norm=True,
                out_indices=[1, 2, 3],
                with_cp=False,
                convert_weights=True,
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth')),
            neck=dict(
                type='GeneralizedLSSFPN',
                in_channels=[192, 384, 768],
                out_channels=256,
                start_level=0,
                num_outs=3,
                norm_cfg=dict(
                    type='BN2d',
                    requires_grad=True),
                act_cfg=dict(
                    type='ReLU',
                    inplace=True),
                upsample_cfg=dict(
                    mode='bilinear',
                    align_corners=False)),
            vtransform=dict(
                type='LSSTransform',
                image_size=image_size,
                in_channels=256,
                out_channels=80,
                feature_size=[image_size[0] // 8, image_size[1] // 8],
                xbound=[-51.2, 51.2, 0.4],
                ybound=[-51.2, 51.2, 0.4],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[1.0, 60.0, 0.5],
                downsample=2)),
        radar=dict(
            voxelize=dict(
                max_num_points=10,
                point_cloud_range=point_cloud_range,
                voxel_size=[0.8, 0.8, 8.0],
                max_voxels=[30000, 40000]),  # [90000, 120000]),
            radar_voxel_encoder=dict(
                type='PillarFeatureNet',
                in_channels=8,
                feat_channels=[32],
                with_distance=False,
                voxel_size=(0.8, 0.8, 8.0),
                norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                point_cloud_range=point_cloud_range),
            radar_middle_encoder=dict(
                type='PointPillarsScatter', in_channels=32, output_shape=(128, 128)))),
    fuser=dict(
        type='RadarConvFuser',
        in_channels=[80, 32],
        out_channels=128,
        deconv_blocks=3),
    decoder=dict(
        backbone=dict(
            type='SECOND',
            in_channels=128,
            out_channels=[128, 256],
            layer_nums=[5, 5],
            layer_strides=[1, 2],
            norm_cfg=dict(
                type='BN',
                eps=1.0e-3,
                momentum=0.01),
            conv_cfg=dict(
                type='Conv2d',
                bias=False)),
        neck=dict(
            type='SECONDFPN',
            in_channels=[128, 256],
            out_channels=[128, 128],
            upsample_strides=[1, 2],
            norm_cfg=dict(
                type='BN',
                eps=1.0e-3,
                momentum=0.01),
            upsample_cfg=dict(
                type='deconv',
                bias=False),
            use_conv_for_no_stride=True)),
    heads=dict(
        type='TransFusionHead',
        num_proposals=200,
        auxiliary=True,
        in_channels=256,
        hidden_channel=128,
        num_classes=10,
        num_decoder_layers=1,
        num_heads=8,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        common_heads=dict(
            center=[2, 2],
            height=[1, 2],
            dim=[3, 2],
            rot=[2, 2],
            vel=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=10),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_heatmap=dict(
            type='GaussianFocalLoss',
            reduction='mean',
            loss_weight=1.0),
        loss_bbox=dict(
            type='L1Loss',
            reduction='mean',
            loss_weight=0.25),
        train_cfg=dict(
            dataset='nuScenes',
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 1],
            voxel_size=voxel_size,
            out_size_factor=8,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(
                    type='BboxOverlaps3D',
                    coordinate='lidar',
                ),
                cls_cost=dict(
                    type='FocalLossCost',
                    gamma=2.0,
                    alpha=0.25,
                    weight=0.15),
                reg_cost=dict(
                    type='BBoxBEVL1Cost',
                    weight=0.25),
                iou_cost=dict(
                    type='IoU3DCost',
                    weight=0.25))),
        test_cfg=dict(
            dataset='nuScenes',
            grid_size=[1024, 1024, 1],
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            nms_type=None
        )))

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)
total_epochs = 20
runner = dict(type='CustomEpochBasedRunner', max_epochs=20)
evaluation = dict(interval=1)
