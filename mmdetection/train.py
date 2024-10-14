from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device


model_dir = 'cascade_rcnn'
model_name = 'cascade_rcnn_r101_fpn_1x_coco'
work_dir = f'./work_dirs/{model_name}'

def main() :
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    cfg = Config.fromfile(f'./configs/{model_dir}/{model_name}.py')

    root='/data/ephemeral/home/dataset/'

    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + 'train_split.json' # train json 정보

    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = root
    cfg.data.val.ann_file = root + 'val_split.json' # train json 정보

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json' # test json 정보

    cfg.data.samples_per_gpu = 8

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

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)

    cfg.log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

    cfg.device = get_device()

    datasets = [build_dataset(cfg.data.train)]

    model = build_detector(cfg.model)
    model.init_weights()

    train_detector(model, datasets, cfg, distributed=False, validate=True)

if __name__ == '__main__':
    main()