"""Evaluate the published MR -> CT EPA weights on CT test volumes."""

_base_ = ['./mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py']

data = dict(
    test=dict(
        _delete_=True,
        type='MedNpzVolumeDataset',
        img_list='data/datalist/test_ct.txt',
        slice_axis=2,
        flip_axes=(0, 1),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadNpzSliceFromFile',
                input_range='-1_1',
                modality='CT',
                flip_axes=(0, 1),
                to_bgr=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(
                        type='Normalize',
                        mean=[104.00698793, 116.66876762, 122.67891434],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ]),
        ]))
