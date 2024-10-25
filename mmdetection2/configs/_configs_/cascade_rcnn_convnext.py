# config.py
from mmcv import Config
from mmdet.utils import get_device

model_dir = 'cascade_rcnn'
model_name = 'cascade_rcnn_convnext_fpn_1x_coco'
work_dir = f'./work_dirs/{model_name}'

def get_cfg():
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    cfg = Config.fromfile(f'./configs/{model_dir}/{model_name}.py')

    root = '/data/ephemeral/home/dataset/'

    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + 'train.json'  # train json 정보

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json'  # test json 정보

    cfg.data.samples_per_gpu = 2

    cfg.seed = 2024
    cfg.gpu_ids = [0]
    cfg.work_dir = work_dir

    cfg.evaluation = dict(
        interval=1,
        metric='bbox',
        save_best='auto',
        by_epoch=True
    )

    for bbox_head in cfg.model.roi_head.bbox_head:
        bbox_head.num_classes = 10

    cfg.runner.max_epochs = 20

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)

    cfg.log_config = dict(
        interval=50,
        hooks=[
            dict(type='TextLoggerHook'),
            dict(type='TensorboardLoggerHook'),
        ]
    )

    cfg.device = get_device()

    return cfg
