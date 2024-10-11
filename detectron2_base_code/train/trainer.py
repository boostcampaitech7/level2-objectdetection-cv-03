import os
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader

from common.augmentation import MyMapper

class MyTrainer(DefaultTrainer):
    
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(cfg, mapper=MyMapper, sampler=sampler)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs('../../detectron2/output_eval', exist_ok=True)
            output_folder = '../../detectron2/output_eval'
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
