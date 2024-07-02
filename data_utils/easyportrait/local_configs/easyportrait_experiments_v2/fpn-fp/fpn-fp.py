norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'EasyPortraitFPDataset'
data_root = '/home/jovyan/datasets/wacv_24/'
img_norm_cfg = dict(
    mean=[143.55267075, 132.96705975, 126.94924335],
    std=[60.2625333, 60.32740275, 59.30988645],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.0),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=16,
        contrast_range=(0.5, 1.0),
        saturation_range=(0.5, 1.0),
        hue_delta=5),
    dict(
        type='Normalize',
        mean=[143.55267075, 132.96705975, 126.94924335],
        std=[60.2625333, 60.32740275, 59.30988645],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(384, 384),
        flip=False,
        transforms=[
            dict(
                type='Normalize',
                mean=[143.55267075, 132.96705975, 126.94924335],
                std=[60.2625333, 60.32740275, 59.30988645],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    train=dict(
        type='EasyPortraitFPDataset',
        data_root='/home/jovyan/datasets/wacv_24/',
        classes=('background', 'skin', 'left brow', 'right brow', 'left eye',
                 'right eye', 'lips', 'teeth'),
        img_dir='easyportrait_384/images/train',
        ann_dir='easyportrait_384/annotations_fp/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='RandomFlip', prob=0.0),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=16,
                contrast_range=(0.5, 1.0),
                saturation_range=(0.5, 1.0),
                hue_delta=5),
            dict(
                type='Normalize',
                mean=[143.55267075, 132.96705975, 126.94924335],
                std=[60.2625333, 60.32740275, 59.30988645],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='EasyPortraitFPDataset',
        data_root='/home/jovyan/datasets/wacv_24/',
        classes=('background', 'skin', 'left brow', 'right brow', 'left eye',
                 'right eye', 'lips', 'teeth'),
        img_dir='easyportrait_384/images/val',
        ann_dir='easyportrait_384/annotations_fp/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(384, 384),
                flip=False,
                transforms=[
                    dict(
                        type='Normalize',
                        mean=[143.55267075, 132.96705975, 126.94924335],
                        std=[60.2625333, 60.32740275, 59.30988645],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='EasyPortraitFPDataset',
        data_root='/home/jovyan/datasets/wacv_24/',
        classes=('background', 'skin', 'left brow', 'right brow', 'left eye',
                 'right eye', 'lips', 'teeth'),
        img_dir='easyportrait_384/images/test',
        ann_dir='easyportrait_384/annotations_fp/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(384, 384),
                flip=False,
                transforms=[
                    dict(
                        type='Normalize',
                        mean=[143.55267075, 132.96705975, 126.94924335],
                        std=[60.2625333, 60.32740275, 59.30988645],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    samples_per_gpu=32,
    workers_per_gpu=8)
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
default_hooks = dict(stop=dict(type='EarlyStoppingHook', monitor='mIoU'))
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(by_epoch=True, interval=100)
evaluation = dict(interval=1, metric='mIoU', save_best='mIoU')
work_dir = 'work_dirs/petrova/fpn-fp'
gpu_ids = [0]
auto_resume = False
