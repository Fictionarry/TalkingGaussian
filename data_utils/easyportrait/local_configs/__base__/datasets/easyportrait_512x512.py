# dataset settings
dataset_type = 'EasyPortraitDataset'
data_root = 'path/to/data/EasyPortrait'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Pad', size=(1920, 1920), pad_val=0, seg_pad_val=255),
    dict(type='Resize', img_scale=(512, 512)),
    
    # We don't use RandomFlip, but need it in the code to fix error: https://github.com/open-mmlab/mmsegmentation/issues/231
    dict(type='RandomFlip', prob=0.0), 
    dict(type='PhotoMetricDistortion', 
         brightness_delta=16, 
         contrast_range=(0.5, 1.0),
         saturation_range=(0.5, 1.0),
         hue_delta=9),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='annotations/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='annotations/test',
        pipeline=test_pipeline))