# _base_ = './fcn_unet_s5-d16_64x64_40k_glas.py'
_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py',
    '../_base_/datasets/glas.py',
    '../_base_/default_runtime.py',
    # '../_base_/schedules/schedule_20k.py'
    '../_base_/schedules/schedule_40k.py'
]





#data = dict(samples_per_gpu=16 * 2 * 2)
#data = dict(samples_per_gpu=64)
# model = dict(test_cfg=dict(crop_size=(480, 480), stride=(320, 320)))




# checkpoint_config = dict(by_epoch=False, interval=4000)


# evaluation = dict(interval=4, metric='mIoU', pre_eval=True)

eval_iters = 2000


evaluation = dict(interval=eval_iters, metric='glas', pre_eval=False, save_best='mAcc')




log_config = dict(
    # interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MyTensorboardLoggerHook')
    ])


custom_hooks = [
    dict(type='HistParamHook', priority='Low', interval=eval_iters)
]











work_dir = './work_dirs/fcn_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_glas'
# work_dir = './tmp_del/fcn_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_glas'

checkpoint_config = dict(
    # by_epoch=False, interval=4000
    # hooks=[
        # dict(type='MyCheckpointHook', by_epoch=False, interval=1)
    # ]
    # type='MyCheckpointHook', by_epoch=False, interval=200
    by_epoch=False, interval=eval_iters, max_keep_ckpts=5
)

# img_scale = (522, 775)







bs_scale = 4

# optimizer
# optimizer = dict(type='SGD', lr=0.01 * bs_scale / 2, momentum=0.9, weight_decay=0.0005)
optimizer = dict(type='SGD', lr=0.01 * bs_scale , momentum=0.9, weight_decay=0.0005)
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# optimizer_config = dict()

# learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4 * 16, by_epoch=False)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4 * bs_scale , by_epoch=False, warmup='linear', warmup_iters=1500, warmup_ratio=1e-6,)

data = dict(samples_per_gpu=4 * bs_scale)

model = dict(

    decode_head=dict(
        # in_channels=()
        in_index=(0, 1, 2, 3, 4),
        ignore_index=255,
        in_channels=(64, 128, 256, 512, 1024),
        # input_transforms=''
        input_transform='resize_concat',
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0),
        ],
        # input_transform='resize_concat',
    ),

    # test_cfg=
    # test_cfg=dict(crop_size=(480, 480), stride=(320, 320))
    #test_cfg=dict(crop_size=(480, 480), stride=(128, 128))
    test_cfg=dict(crop_size=(480, 480), stride=(256, 256))
    #test_cfg=dict(crop_size=(256, 256), stride=(128, 128))

)

# model=dict(
#     decode_head=dict(
#         sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)))

# # fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')

# # fp16 placeholder
fp16 = dict()