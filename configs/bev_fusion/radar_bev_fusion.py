_base_ = [
    '../_base_/datasets/bev-radar-nus.py',
    '../_base_/default_runtime.py'
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
image_size = [256, 704]
voxel_size = [0.1, 0.1, 0.2]

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
            type='GeneralizedResNet',
            in_channels=128,
            blocks=([2, 128, 2], [2, 256, 2], [2, 512, 1])),
        neck=dict(
            type='LSSFPN',
            in_indices=[-1, 0],
            in_channels=[512, 128],
            out_channels=256,
            scale_factor=2)),
    heads=dict(
        type='CenterHead',
        in_channels=256,
        train_cfg=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 1],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]),
        test_cfg=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type=['circle', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'],
            nms_scale=[
                [1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0],
                [1.0, 1.0],
                [2.5, 4.0]],
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2),
        tasks=[['car'], ['truck', 'construction_vehicle'], ['bus', 'trailer'], [
            'barrier'], ['motorcycle', 'bicycle'], ['pedestrian', 'traffic_cone']],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True))

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='cyclic',
    target_ratio=(5, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
total_epochs = 20
runner = dict(type='CustomEpochBasedRunner', max_epochs=20)
evaluation = dict(interval=1)
