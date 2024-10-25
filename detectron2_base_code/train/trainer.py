import os
from typing import OrderedDict
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger
logger = setup_logger()

from common.augmentation import MyMapper, MyBaseMapper

class MyTrainer(DefaultTrainer):
    
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(cfg, mapper=MyBaseMapper, sampler=sampler)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs('../../detectron2/output_eval', exist_ok=True)
            output_folder = '../../detectron2/output_eval'
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

class SwinTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        # MyBaseMapper를 mapper로 설정하여 데이터 로더를 구성
        return build_detection_train_loader(cfg, mapper=MyBaseMapper, sampler=sampler)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name=dataset_name,
                             tasks=["bbox"],
                             distributed=True,
                             output_dir=output_folder)
    
    @classmethod
    def build_tta_model(cls, cfg, model):
        return GeneralizedRCNNWithTTA(cfg, model)
    
    @classmethod
    def test_with_TTA(cls, cfg, model):
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = cls.build_tta_model(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res