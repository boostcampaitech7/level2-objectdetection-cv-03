import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN  # Swin 설정 추가에 필요
from timm import create_model
from detectron2.modeling import build_backbone

from swin.swint import add_swint_config

def setup_config(training=True):
    cfg = get_cfg()
    add_swint_config(cfg)

    if training:
        cfg.merge_from_file("/data/ephemeral/home/git/detectron2_base_code/swin/configs/SwinT/faster_rcnn_swint_T_FPN_3x_.yaml")

        cfg.DATASETS.TRAIN = ('coco_trash_train',)
        cfg.DATASETS.TEST = ('coco_trash_val',)

        cfg.DATALOADER.NUM_WORKERS = 2

        cfg.MODEL.WEIGHTS = None #"/data/ephemeral/home/git/detectron2_base_code/swin/faster_rcnn_swint_T.pth"

        cfg.SOLVER.IMS_PER_BATCH = 4
        cfg.SOLVER.BASE_LR = 0.0001
        cfg.SOLVER.MAX_ITER = 26000
        cfg.SOLVER.STEPS = (10000, 20000)
        cfg.SOLVER.GAMMA = 0.1
        cfg.SOLVER.CHECKPOINT_PERIOD = 2000

        cfg.OUTPUT_DIR = '../../detectron2/output'

        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

        cfg.TEST.EVAL_PERIOD = 2000
    else:
        cfg.merge_from_file("/data/ephemeral/home/git/detectron2_base_code/swin/configs/SwinT/faster_rcnn_swint_T_FPN_3x_.yaml")

        cfg.DATASETS.TEST = ('coco_trash_test',)

        cfg.DATALOADER.NUM_WORKERS = 2

        cfg.OUTPUT_DIR = '../../detectron2/output'

        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')

        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3

    return cfg
