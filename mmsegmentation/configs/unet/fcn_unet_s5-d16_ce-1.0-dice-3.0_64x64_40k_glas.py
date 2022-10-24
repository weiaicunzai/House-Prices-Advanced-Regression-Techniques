_base_ = './fcn_unet_s5-d16_64x64_40k_glas.py'
model = dict(
    decode_head=dict(loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
    ]))



data = dict(samples_per_gpu=16 * 2 * 2)
# checkpoint_config = dict(by_epoch=False, interval=4000)
# evaluation = dict(interval=4, metric='mIoU', pre_eval=True)
evaluation = dict(interval=400, metric='glas', pre_eval=False, save_best='mAcc')
# # fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# # fp16 placeholder
fp16 = dict()

log_config = dict(
    # interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MyTensorboardLoggerHook')
    ])


custom_hooks = [
    dict(type='HistParamHook', priority='Low', interval=400)
]

model=dict(
    decode_head=dict(
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)))

# import datetime

# DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
# TIME_NOW = datetime.datetime.now().strftime(DATE_FORMAT)

work_dir = './work_dirs/fcn_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_glas'

checkpoint_config = dict(
    # by_epoch=False, interval=4000
    # hooks=[
        # dict(type='MyCheckpointHook', by_epoch=False, interval=1)
    # ]
    # type='MyCheckpointHook', by_epoch=False, interval=200
    by_epoch=False, interval=400, max_keep_ckpts=5
)