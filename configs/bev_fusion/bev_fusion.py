_base_ = [
    '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.075, 0.075, 0.2]
image_size = [256, 704]

# For nuScenes we usually do 10-class detection
class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
               ]
map_classes = [
    'drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'divider'
]
dataset_type = 'NuScenesDataset'
data_root = '/data/workspace/zhaoyuan/bev/mmdetection3d/data/nuscenes/'
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/nuscenes/': 's3://nuscenes/nuscenes/',
#         'data/nuscenes/': 's3://nuscenes/nuscenes/'
#     }))
max_epochs = 20
# augment2d=dict(
#    resize=[[0.48, 0.48], [0.48, 0.48]],
#    rotate=[-0.3925, 0.3925],
#    gridmask=dict(
#        prob=0.0,
#        fixed_prob=True))

db_sampler = dict(
    dataset_root=data_root,
    info_path=data_root + "nuscenes_dbinfos_train.pkl",
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(car=5, truck=5, bus=5, trailer=5,
                 construction_vehicle=5, traffic_cone=5, barrier=5, motorcycle=5, bicycle=5, pedestrian=5)),
    sample_groups=dict(car=2, truck=3, construction_vehicle=7, bus=4, traile=6,
                       barrier=2, motorcycle=6, bicycle=6, pedestrian=2, traffic_cone=2),
    classes=class_names,
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        reduce_beams=32))

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        reduce_beams=32),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        reduce_beams=32,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(
        type='ObjectPaste',
        stop_epoch=-1,
        db_sampler=db_sampler,
        sample_2d=True),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.38, 0.55],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    dict(
        type='GlobalRotScaleTrans',
        resize_lim=[0.95, 1.05],
        rot_lim=[-0.3925 * 2, 0.3925 * 2],
        trans_lim=[0.5],
        is_train=True),
    dict(
        type='LoadBEVSegmentation',
        dataset_root=data_root,
        xbound=[-50.0, 50.0, 0.5],
        ybound=[-50.0, 50.0, 0.5],
        classes=map_classes),
    dict(type='RandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(
        type='ImageNormalize',
        mean=[103.530, 116.280, 123.675],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='GridMask',
        use_h=True,
        use_w=True,
        max_epoch=max_epochs,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=1,
        fixed_prob=False),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', classes=class_names),
    dict(type='Collect3D', keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_masks_bev'], meta_keys=[
         'camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera', 'lidar2image', 'img_aug_matrix', 'lidar_aug_matrix'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        reduce_beams=32),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        reduce_beams=32,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(
        type='GlobalRotScaleTrans',
        resize_lim=[1.0, 1.0],
        rot_lim=[0.0, 0.0],
        trans_lim=0.0,
        is_train=False),
    dict(
        type='LoadBEVSegmentation',
        dataset_root=data_root,
        xbound=[-50.0, 50.0, 0.5],
        ybound=[-50.0, 50.0, 0.5],
        classes=map_classes),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ImageNormalize',
        mean=[103.530, 116.280, 123.675],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='DefaultFormatBundle3D',
        classes=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_masks_bev'], meta_keys=[
         'camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera', 'lidar2image', 'img_aug_matrix', 'lidar_aug_matrix'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            dataset_root=data_root,
            ann_file=data_root + 'nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            object_classes=class_names,
            map_classes=map_classes,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        dataset_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        object_classes=class_names,
        map_classes=map_classes,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        dataset_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        object_classes=class_names,
        map_classes=map_classes,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))

model = dict(
    type='BEVFusion',
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
                in_channels=[192, 384, 768],  # [512, 1024, 2048],
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
                type='DepthLSSTransform',
                in_channels=256,
                out_channels=80,
                image_size=image_size,
                feature_size=[image_size[0] // 8, image_size[1] // 8],
                xbound=[-54.0, 54.0, 0.3],
                ybound=[-54.0, 54.0, 0.3],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[1.0, 60.0, 0.5],
                downsample=2)),
        lidar=dict(
            voxelize=dict(
                max_num_points=10,
                point_cloud_range=point_cloud_range,
                voxel_size=voxel_size,
                max_voxels=[90000, 120000]),
            backbone=dict(
                type='SparseEncoder',
                in_channels=5,
                sparse_shape=[1440, 1440, 41],
                output_channels=128,
                order=('conv', 'norm', 'act'),
                encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
                encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [1, 1, 0]), (0, 0)),
                block_type='basicblock'))),
    fuser=dict(
        type='ConvFuser',
        in_channels=[80, 256],
        out_channels=256),
    decoder=dict(
        backbone=dict(
            type='SECOND',
            in_channels=256,
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
            out_channels=[256, 256],
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
        in_channels=512,
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
            grid_size=[1440, 1440, 41],
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
            grid_size=[1440, 1440, 41],
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
evaluation = dict(interval=2)
