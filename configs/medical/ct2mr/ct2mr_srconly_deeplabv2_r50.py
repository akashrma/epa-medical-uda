# Source-only on CT (target MR ignored) with DeepLabV2

_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/deeplabv2_r50-d8.py',
    '../../_base_/datasets/uda_ct_to_mr_256x256.py',
    '../../_base_/uda/dacs.py',
    '../../_base_/schedules/sgd.py',
    '../../_base_/schedules/poly10.py',
]

seed = 0

model = dict(
    decode_head=dict(
        num_classes=5,
        loss_decode=dict(
            type='DiceCELoss',
            ce_weight=0.5,
            dice_weight=0.5,
            loss_weight=1.0,
        ),
    ),
)

uda = dict(source_only=True)

# Evaluate on both source (CT) and target (MR) domains.
img_norm_cfg = dict(
    mean=[104.00698793, 116.66876762, 122.67891434],
    std=[1.0, 1.0, 1.0],
    to_rgb=False)
crop_size = (256, 256)
volume_dataset_type = 'MedNpzVolumeDataset'

source_test_pipeline = [
    dict(
        type='LoadNpzSliceFromFile',
        input_range='-1_1',
        modality='CT',
        flip_axes=(0, 1),
        to_bgr=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
target_test_pipeline = [
    dict(
        type='LoadNpzSliceFromFile',
        input_range='-1_1',
        modality='MR',
        flip_axes=(0, 1),
        to_bgr=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    val=[
        dict(
            type=volume_dataset_type,
            eval_name='source',
            img_list='data/datalist/test_ct.txt',
            slice_axis=2,
            flip_axes=(0, 1),
            test_mode=True,
            pipeline=source_test_pipeline),
        dict(
            type=volume_dataset_type,
            eval_name='target',
            img_list='data/datalist/test_mr.txt',
            slice_axis=2,
            flip_axes=(0, 1),
            test_mode=True,
            pipeline=target_test_pipeline),
    ],
    test=[
        dict(
            type=volume_dataset_type,
            eval_name='source',
            img_list='data/datalist/test_ct.txt',
            slice_axis=2,
            flip_axes=(0, 1),
            test_mode=True,
            pipeline=source_test_pipeline),
        dict(
            type=volume_dataset_type,
            eval_name='target',
            img_list='data/datalist/test_mr.txt',
            slice_axis=2,
            flip_axes=(0, 1),
            test_mode=True,
            pipeline=target_test_pipeline),
    ],
)

optimizer = dict(type='SGD', lr=2.5e-4, momentum=0.9, weight_decay=0.0005)
optimizer_config = None
runner = dict(type='IterBasedRunner', max_iters=50000)
checkpoint_config = dict(by_epoch=False, interval=50000, max_keep_ckpts=1)
evaluation = dict(interval=500, metric='mDice', save_best='mDice')
log_config = dict(
    interval=50,
    img_interval=1000,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])

name = 'ct2mr_srconly_deeplabv2_r50'
exp = 'medical'
name_dataset = 'ct2mr'
name_architecture = 'deeplabv2_r50-d8'
name_encoder = 'r50'
name_decoder = 'dlv2'
name_uda = 'source_only'
name_opt = 'sgd_2.5e-4_poly10_50k'
