# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),

    # Shear : level(0 ~ 10), direction('horizontal' 또는 'vertical')
    dict(
        type="Shear",
        level=2,
        prob=0.3,
        direction="horizontal",
        max_shear_magnitude=0.3,
        random_negative_prob=0.5,
    ),

    # Rotate : level(0~10), max_rotate_angle(양수->시계방향)
    dict(
        type="Rotate", level=1, prob=0.5, max_rotate_angle=30, random_negative_prob=0.5
    ),

    # Translate : level(0~10), direction('vertical' 가능), min_size(tranlate 후 filtering할 최소 bbox pixel)
    dict(
        type="Translate",
        level=1,
        prob=0.5,
        direction="horizontal",
        max_translate_offset=250.0,
        random_negative_prob=0.5,
        min_size=0,
    ),

    # RandomShift : Shift_ratio(=prob), filter_thr_px(너비, 높이 threshold for filtering)
    dict(type="RandomShift", shift_ratio=0.5, max_shift_px=32, filter_thr_px=1),

    # MinIoURandomCrop : min_ious(tuple-minimum IoU threshold for all intersections with bboxes), min_crop_size(float) -> 자세 설명 notion
    dict(
        type="MinIoURandomCrop",
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.2,
        bbox_clip_border=True,
    ),

    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
