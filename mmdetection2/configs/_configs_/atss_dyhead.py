# config.py
from mmcv import Config
from mmdet.utils import get_device

model_dir = 'dyhead'
model_name = 'atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco'
work_dir = f'./work_dirs/{model_name}'

def get_cfg():
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    cfg = Config.fromfile(f'./configs/{model_dir}/{model_name}.py')

    root = '/data/ephemeral/home/dataset/'

    cfg.data.train.dataset.classes = classes
    cfg.data.train.dataset.img_prefix = root
    cfg.data.train.dataset.ann_file = root + 'train.json'

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json'  # test json 정보

    cfg.seed = 2024
    cfg.gpu_ids = [0]
    cfg.work_dir = work_dir

    cfg.evaluation = dict(
        interval=1,
        metric='bbox',
        save_best='auto',
        by_epoch=True
    )

    cfg.data.samples_per_gpu = 2

    cfg.model.bbox_head.num_classes = 10 

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
