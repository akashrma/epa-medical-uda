# Self-training (no class mixing) on CT -> MR with SegFormer (MiT-B5)
# + source ETF contrastive loss

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/segformer_b5.py',
    '../_base_/datasets/uda_ct_to_mr_256x256.py',
    '../_base_/uda/dacs.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py',
]

seed = 0

model = dict(
    pretrained='pretrained/mit_b5_mmseg.pth',
    decode_head=dict(num_classes=5),
)

uda = dict(
    mix='none',
    mix_start_iter=1000,
    etf_contrastive_lambda=0.2,
    etf_contrastive_temperature=0.05,
)

# SegFormer (MiT) optimizer setup
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
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

name = 'ct2mr_selftrain_nomix_etf_segformer_mitb5'
exp = 'medical'
name_dataset = 'ct2mr'
name_architecture = 'segformer_mitb5'
name_encoder = 'mitb5'
name_decoder = 'segformer'
name_uda = 'selftrain_nomix_etf'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_50k'
