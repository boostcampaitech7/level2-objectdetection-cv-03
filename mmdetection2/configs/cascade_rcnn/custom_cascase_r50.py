_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/recycle_detection.py',
    '../_base_/schedules/schedule_2x_with_adamw.py', '../_base_/default_runtime.py'
]