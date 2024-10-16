import os
from detectron2 import model_zoo
from detectron2.config import get_cfg

def setup_config(training=True):
    cfg = get_cfg()
    if training:
        cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))

        cfg.DATASETS.TRAIN = ('coco_trash_train',)
        cfg.DATASETS.TEST = ('coco_trash_val',)

        cfg.DATALOADER.NUM_WORKERS = 2

        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')

        cfg.SOLVER.IMS_PER_BATCH = 8
        cfg.SOLVER.BASE_LR = 0.001
        cfg.SOLVER.MAX_ITER = 15000
        cfg.SOLVER.STEPS = (8000, 12000)
        cfg.SOLVER.GAMMA = 0.005
        cfg.SOLVER.CHECKPOINT_PERIOD = 3000

        cfg.OUTPUT_DIR = '../../detectron2/output'

        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

        cfg.TEST.EVAL_PERIOD = 3000
    else:
        cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))

        cfg.DATASETS.TEST = ('coco_trash_test',)

        cfg.DATALOADER.NUM_WORKERS = 2

        cfg.OUTPUT_DIR = '../../detectron2/output'

        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')

        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        
    return cfg
