_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', 
    '../_base_/datasets/glas.py',
    '../_base_/default_runtime.py', 
    # '../_base_/schedules/schedule_20k.py'
    '../_base_/schedules/schedule_40k.py'
]
# model = dict(test_cfg=dict(crop_size=(256, 256), stride=(128, 128)))
evaluation = dict(metric='mIOU')






