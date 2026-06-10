# DeepLabV2 no-FDA target EPA start ablation.

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/deeplabv2_r50-d8.py',
    '../_base_/datasets/uda_ct_to_mr_256x256.py',
    '../_base_/uda/dacs.py',
    '../_base_/schedules/sgd.py',
    '../_base_/schedules/poly10.py',
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

uda = dict(
    mix='none',
    mix_start_iter=1000,
    etf_contrastive_lambda=0.0,
    etf_contrastive_target_lambda=0.0,
    prototype_mode='source_ema',
    dynamic_proto_feature_dim=2048,
    dynamic_proto_momentum=0.99,
    dynamic_proto_min_pixels=16,
    dynamic_proto_source_lambda=0.2,
    dynamic_proto_target_lambda=0.2,
    dynamic_proto_target_start_iter=2000,
    dynamic_proto_temperature=0.05,
    dynamic_proto_target_temperature=0.05,
    dynamic_proto_ignore_background=True,
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

name = 'ct2mr_selftrain_nomix_srcproto_tgtalign_start2000_dicece_deeplabv2_r50_nofda'
exp = 'medical'
name_dataset = 'ct2mr'
name_architecture = 'deeplabv2_r50-d8'
name_encoder = 'r50'
name_decoder = 'dlv2'
name_uda = 'selftrain_nomix_srcproto_tgtalign_start2000_nofda'
name_opt = 'sgd_2.5e-4_poly10_50k'
