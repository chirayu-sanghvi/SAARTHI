
_base_ = 'D:\Cardd\mmdetection\configs\mask_rcnn\mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py'


data_root = 'data\CarDD_release\CarDD_release\CarDD_COCO' # dataset root

train_batch_size_per_gpu = 4
train_num_workers = 2

max_epochs = 10
stage2_num_epochs = 1
base_lr = 0.00008

metainfo = {
    'classes': ('dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat', ),
    'palette': [
    (220, 20, 60),    # Red for 'dent'
    (0, 255, 0),      # Green for 'scratch'
    (0, 0, 255),      # Blue for 'crack'
    (255, 255, 0),    # Yellow for 'glass shatter'
    (255, 165, 0),    # Orange for 'lamp broken'
    (128, 0, 128)     # Purple for 'tire flat'

    ]
}

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='train2017/'),
        ann_file='annotations\instances_train2017.json'))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='val2017/'),
        ann_file='annotations\instances_val2017.json'))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations\instances_val2017.json')

test_evaluator = val_evaluator

# Modify the model for a single class
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=6),
        mask_head=dict(num_classes=6)
    )
)





# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=10),
    dict(
        # use cosine lr from 10 to 20 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

# train_pipeline_stage2 = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),  # Ensure masks are loaded
#     dict(
#         type='Resize',
#         img_scale=(640, 640),
#         keep_ratio=True
#     ),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(
#         type='Normalize',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         to_rgb=True
#     ),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
# ]


# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=2,  # only keep latest 2 checkpoints
        save_best='auto'
    ),
    logger=dict(type='LoggerHook', interval=5))

custom_hooks = [
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# load COCO pre-trained weight
load_from = 'checkpoints\mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
#visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])


# Set up training configurations
# train_cfg = dict(
#     val_interval=3
# )

# Logging configurations
# visualizer = dict(
#     vis_backends=[dict(type='TensorboardVisBackend')]
# )
