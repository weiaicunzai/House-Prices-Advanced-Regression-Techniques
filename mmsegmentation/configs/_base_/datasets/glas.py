# dataset settings
#dataset_type = 'DRIVEDataset'
dataset_type = 'GlaSDataset'
data_root = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/mmseg_glas'
img_norm_cfg = dict(
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    mean=[200.2103937666394, 130.35073328696086, 200.7955948978498], 
    std=[33.296002816472495, 62.533182555417845, 42.21131635702842], 
    to_rgb=True)
# img_scale = (584, 565)
img_scale = (522, 775)
# img_scale = (775, 522)
#crop_size = (64, 64)
#crop_size = (256, 256)
crop_size = (480, 480)
train_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='LoadImageFromMemory'),
    # dict(type='LoadAnnotations'),
    dict(type='LoadAnnotationsFromMemory'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomRotate', prob=0.5, degree=(0, 90), auto_bound=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='LoadImageFromMemory'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        flip=False,
        # flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=16),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    # samples_per_gpu=16 * 2 * 2,
    # samples_per_gpu=16,
    samples_per_gpu=4,
    workers_per_gpu=4,
    # workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        # times=40000,
        times=30000,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            #img_dir='images/training',
            #ann_dir='annotations/training',
            img_dir='train',
            ann_dir='train',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        #img_dir='images/validation',
        #ann_dir='annotations/validation',
        img_dir='test',
        ann_dir='test',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        #img_dir='images/validation',
        #ann_dir='annotations/validation',
        img_dir='test',
        ann_dir='test',
        pipeline=test_pipeline))
