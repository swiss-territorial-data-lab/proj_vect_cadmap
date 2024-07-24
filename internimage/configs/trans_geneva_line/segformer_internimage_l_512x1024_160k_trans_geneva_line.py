# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/geneva_line.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
load_from = '/scratch/izar/shanli/Cadmap/internimage/pretrain-segformer-ce-photo-0.6221/best_mIoU_iter_48000.pth'
auto_resume = True
model = dict(
    backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=160,
        depths=[5, 5, 22, 5],
        groups=[10, 20, 40, 80],
        mlp_ratio=4.,
        drop_path_rate=0.4,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=2.0,
        post_norm=True,
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=load_from)),
    decode_head=dict(num_classes=2, 
                     in_channels=[160, 320, 640, 1280],

                     loss_decode=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                        )),

                    #  loss_decode=dict(type='FocalLoss',
                    #                   class_weight=[0.00001, 0.99999],
                    #                   gamma=2.0,
                    #                   use_sigmoid=True,
                    #                   loss_weight=1.0)),
    test_cfg=dict(mode='whole', threshold=0.5))
optimizer = dict(
    _delete_=True, type='AdamW', lr=5e-6, betas=(0.9, 0.999), weight_decay=0.05,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=37, layer_decay_rate=0.94,
                       depths=[5, 5, 22, 5], offset_lr_scale=1.0,
                       custom_keys={'head': dict(lr_mult=10.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-5,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
runner = dict(type='IterBasedRunner')
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')
# evaluation = dict(type='IoUMetric', interval=50, metric='Borderline', save_best='Borderline')
# fp16 = dict(loss_scale=dict(init_scale=512))