_base_ = ['./base/default_runtime.py', './base/dataset.py']

# model settings
model = dict(
    type='FabricDefectClassifier',
    backbone=dict(
        type='ResNet',
        depth=34,
        in_channels=6,
        num_stages=4,
        stem_channels=128,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=15,
        in_channels=512,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=15,
            reduction='mean',
            loss_weight=1.0),
        topk=(1, 5),
        cal_acc=False))
train_cfg = dict(mixup=dict(alpha=0.2, num_classes=15))

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.8,
    momentum=0.9,
    weight_decay=1e-4,
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1e-6,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=270)
