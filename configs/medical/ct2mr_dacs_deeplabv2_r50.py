# DACS on CT -> MR with DeepLabV2

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
    decode_head=dict(num_classes=5),
)

optimizer = dict(type='SGD', lr=2.5e-4, momentum=0.9, weight_decay=0.0005)
optimizer_config = None
runner = dict(type='IterBasedRunner', max_iters=50000)
checkpoint_config = dict(by_epoch=False, interval=50000, max_keep_ckpts=1)
evaluation = dict(interval=5000, metric='mDice', save_best='mDice')
log_config = dict(
    interval=50,
    img_interval=1000,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])

name = 'ct2mr_dacs_deeplabv2_r50'
exp = 'medical'
name_dataset = 'ct2mr'
name_architecture = 'deeplabv2_r50-d8'
name_encoder = 'r50'
name_decoder = 'dlv2'
name_uda = 'dacs'
name_opt = 'sgd_2.5e-4_poly10_50k'
