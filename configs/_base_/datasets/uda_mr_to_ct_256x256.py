# MR -> CT dataset settings for UDA with .npy lists

dataset_type = 'MedNpyListDataset'
img_norm_cfg = dict(
    mean=[104.00698793, 116.66876762, 122.67891434],
    std=[1.0, 1.0, 1.0],
    to_rgb=False)
crop_size = (256, 256)

source_train_pipeline = [
    dict(type='LoadNpyImageFromFile', input_range='-1_1', to_bgr=True),
    dict(type='LoadNpyAnnotations'),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

target_train_pipeline = [
    dict(type='LoadNpyImageFromFile', input_range='-1_1', to_bgr=True),
    dict(type='LoadNpyAnnotations'),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadNpyImageFromFile', input_range='-1_1', to_bgr=True),
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
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type=dataset_type,
            img_list='data/datalist/train_mr.txt',
            ann_list='data/datalist/train_mr_gt.txt',
            pipeline=source_train_pipeline),
        target=dict(
            type=dataset_type,
            img_list='data/datalist/train_ct.txt',
            ann_list='data/datalist/train_ct_gt.txt',
            pipeline=target_train_pipeline)),
    val=dict(
        type=dataset_type,
        img_list='data/datalist/val_ct.txt',
        ann_list='data/datalist/val_ct_gt.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_list='data/datalist/val_ct.txt',
        ann_list='data/datalist/val_ct_gt.txt',
        pipeline=test_pipeline))
