model = dict(
    type='SimCLR',
    data_preprocessor=dict(
        mean=(198.878, 167.418, 132.772),
        std=(21.34, 25.105, 26.093),
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=True),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        num_layers=2,
        with_avg_pool=True),
    head=dict(
        type='ContrastiveHead',
        loss=dict(type='mmcls.CrossEntropyLoss'),
        temperature=0.1))
optimizer = dict(type='LARS', lr=0.3, weight_decay=1e-06, momentum=0.9)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='LARS', lr=0.3, weight_decay=1e-06, momentum=0.9),
    paramwise_cfg=dict(
        custom_keys=dict({
            'bn': dict(decay_mult=0, lars_exclude=True),
            'bias': dict(decay_mult=0, lars_exclude=True),
            'downsample.1': dict(decay_mult=0, lars_exclude=True)
        })))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR', T_max=190, by_epoch=True, begin=10, end=200)
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200)
default_scope = 'mmselfsup'
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(
    window_size=10,
    custom_cfg=[dict(data_src='', method='mean', window_size='global')])
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SelfSupVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_level = 'INFO'
load_from = None
resume = False
dataset_type = 'mmcls.CustomDataset'
data_root = 'data/dilbert/'
view_pipeline = [
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=0.5)
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiView',
        num_views=2,
        transforms=[[{
            'type': 'RandomResizedCrop',
            'size': 224,
            'backend': 'pillow'
        }, {
            'type': 'RandomFlip',
            'prob': 0.5
        }, {
            'type':
            'RandomApply',
            'transforms': [{
                'type': 'ColorJitter',
                'brightness': 0.8,
                'contrast': 0.8,
                'saturation': 0.8,
                'hue': 0.2
            }],
            'prob':
            0.8
        }, {
            'type': 'RandomGrayscale',
            'prob': 0.2,
            'keep_channels': True,
            'channel_weights': (0.114, 0.587, 0.2989)
        }, {
            'type': 'RandomGaussianBlur',
            'sigma_min': 0.1,
            'sigma_max': 2.0,
            'prob': 0.5
        }]]),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='mmcls.CustomDataset',
        data_root='data/dilbert/',
        data_prefix=dict(img_path='./'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiView',
                num_views=2,
                transforms=[[{
                    'type': 'RandomResizedCrop',
                    'size': 224,
                    'backend': 'pillow'
                }, {
                    'type': 'RandomFlip',
                    'prob': 0.5
                }, {
                    'type':
                    'RandomApply',
                    'transforms': [{
                        'type': 'ColorJitter',
                        'brightness': 0.8,
                        'contrast': 0.8,
                        'saturation': 0.8,
                        'hue': 0.2
                    }],
                    'prob':
                    0.8
                }, {
                    'type': 'RandomGrayscale',
                    'prob': 0.2,
                    'keep_channels': True,
                    'channel_weights': (0.114, 0.587, 0.2989)
                }, {
                    'type': 'RandomGaussianBlur',
                    'sigma_min': 0.1,
                    'sigma_max': 2.0,
                    'prob': 0.5
                }]]),
            dict(type='PackSelfSupInputs', meta_keys=['img_path'])
        ]))
launcher = 'none'
work_dir = './work_dirs/selfsup/simclr_resnet50_8xb32-coslr-200e_dilbert'
