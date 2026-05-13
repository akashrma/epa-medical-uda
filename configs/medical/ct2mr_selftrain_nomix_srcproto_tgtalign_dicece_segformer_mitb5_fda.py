# Self-training (no class mixing) on CT -> MR with SegFormer (MiT-B5)
# + dynamic source EMA prototypes for target semantic alignment
# + DiceCE segmentation loss + FDA source->target style transfer

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
    mix_start_iter=2000,
    # Disable fixed ETF anchors for this variant.
    etf_contrastive_lambda=0.0,
    etf_contrastive_target_lambda=0.0,
    # Dynamic source-anchor prototype alignment.
    prototype_mode='source_ema',
    dynamic_proto_momentum=0.99,
    dynamic_proto_min_pixels=16,
    dynamic_proto_source_lambda=0.2,
    dynamic_proto_target_lambda=0.2,
    dynamic_proto_target_start_iter=2000,
    dynamic_proto_temperature=0.05,
    dynamic_proto_target_temperature=0.05,
    dynamic_proto_ignore_background=True,
    fda_source_to_target=True,
    fda_source_prob=1.0,
    fda_L=0.01,
    fda_in_denorm=True,
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

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

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

name = 'ct2mr_selftrain_nomix_srcproto_tgtalign_dicece_segformer_mitb5_fda'
exp = 'medical'
name_dataset = 'ct2mr'
name_architecture = 'segformer_mitb5'
name_encoder = 'mitb5'
name_decoder = 'segformer'
name_uda = 'selftrain_nomix_srcproto_tgtalign_fda'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_50k'
