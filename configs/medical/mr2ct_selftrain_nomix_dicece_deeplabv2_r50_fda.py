# Self-training (no class mixing) on MR -> CT with DeepLabV2
# + DiceCE segmentation loss
# + FDA source->target style transfer for source images

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/deeplabv2_r50-d8.py',
    '../_base_/datasets/uda_mr_to_ct_256x256.py',
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
    fda_source_to_target=True,
    fda_source_prob=1.0,
    fda_L=0.01,
    fda_in_denorm=True,
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

name = 'mr2ct_selftrain_nomix_dicece_deeplabv2_r50_fda'
exp = 'medical'
name_dataset = 'mr2ct'
name_architecture = 'deeplabv2_r50-d8'
name_encoder = 'r50'
name_decoder = 'dlv2'
name_uda = 'selftrain_nomix_dicece_fda'
name_opt = 'sgd_2.5e-4_poly10_50k'
