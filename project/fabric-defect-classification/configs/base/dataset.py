# dataset_settings
dataset_type = 'FabricData'

train_pipeline = [
    dict(type='LoadImageFromFile', key='lq'),
    dict(type='LoadImageFromFile', key='gt'),
    # dict(type='CropBoundingBoxArea', keys=['lq', 'gt']),
    dict(type='RandomFlip', keys=['lq', 'gt'], flip_prob=0.5, direction='horizontal'),
    dict(type='Resize', keys=['lq', 'gt'], size=(400, 400)),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Concat', keys=['lq', 'gt'], output_key='img'),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile', key='lq'),
    dict(type='LoadImageFromFile', key='gt'),
    dict(type='Resize', keys=['lq', 'gt'], size=(400, 400)),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Concat', keys=['lq', 'gt'], output_key='img'),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=14,
    workers_per_gpu=8,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            path='fabric_data_partitions',
            partitions=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        path='fabric_data_partitions',
        partitions=[9],
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        path='fabric_data_partitions',
        partitions=[9],
        pipeline=test_pipeline))

evaluation = dict(interval=10, metric='accuracy')
