norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='BiSeNetV2',
        detail_channels=(64, 64, 128),
        semantic_channels=(16, 32, 64, 128),
        semantic_expansion_ratio=6,
        bga_channels=128,
        out_indices=(0, 1, 2, 3, 4),
        init_cfg=None,
        align_corners=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=0,
        channels=1024,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=16,
            channels=16,
            num_convs=2,
            num_classes=8,
            in_index=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=32,
            channels=64,
            num_convs=2,
            num_classes=8,
            in_index=2,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=64,
            channels=256,
            num_convs=2,
            num_classes=8,
            in_index=3,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=1024,
            num_convs=2,
            num_classes=8,
            in_index=4,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ],
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
optimizer = dict(type='AdamW', lr=0.05, weight_decay=0.0001)
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=0.0,
    by_epoch=True,
    warmup='linear',
    warmup_iters=1000)
default_hooks = dict(stop=dict(type='EarlyStoppingHook', monitor='mIoU'))
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(by_epoch=True, interval=100)
evaluation = dict(interval=1, metric='mIoU', save_best='mIoU')
work_dir = 'work_dirs/petrova/bisenet-fp'
gpu_ids = [0]
auto_resume = False
