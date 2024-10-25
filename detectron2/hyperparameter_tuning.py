import os
import copy
import torch
import detectron2
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader

import detectron2.data.transforms as T

import optuna

setup_logger()

# Register Dataset
try:
    register_coco_instances('coco_trash_train', {}, '../../dataset/train.json', '../../dataset/')
except AssertionError:
    pass

try:
    register_coco_instances('coco_trash_test', {}, '../../dataset/test.json', '../../dataset/')
except AssertionError:
    pass

MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                         "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

def MyMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    transform_list = [
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3)
    ]
    
    image, transforms = T.apply_transform_gens(transform_list, image)
    
    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))
    
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop('annotations')
        if obj.get('iscrowd', 0) == 0
    ]
    
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict['instances'] = utils.filter_empty_instances(instances)
    
    return dataset_dict

# Trainer class
class MyTrainer(DefaultTrainer):
    
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(
        cfg, mapper = MyMapper, sampler = sampler
        )
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs('./output_eval', exist_ok=True)
            output_folder = './output_eval'
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

# Objective function for Optuna
# 하이퍼파라미터 설정, 모델 학습이 진행되는 함수
def objective(trial):
    # config 불러오기
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))

    # Dataset 설정
    cfg.DATASETS.TRAIN = ('coco_trash_train',)
    cfg.DATASETS.TEST = ('coco_trash_test',)
    
    # Dataloader 및 기본 설정
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
    
    # 하이퍼파라미터를 Optuna에서 제안받기
    # 배치당 이미지 수를 결정
    cfg.SOLVER.IMS_PER_BATCH = trial.suggest_categorical('ims_per_batch', [4, 8, 16])
    # 학습률 결정 (1e-5 ~ 1e-2) - 학습률이 너무 작으면 학습이 느리고, 너무 크면 불안정함
    cfg.SOLVER.BASE_LR = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    # 학습의 최대 반복 횟수 결정
    cfg.SOLVER.MAX_ITER = 5000 # 고정
    # 학습률 감소를 적용할 시점 결정 (50%와 75% 지점)
    cfg.SOLVER.STEPS = (int(cfg.SOLVER.MAX_ITER * 0.5), int(cfg.SOLVER.MAX_ITER * 0.75))
    # 학습률 감소 비율 결정
    cfg.SOLVER.GAMMA = trial.suggest_float('gamma', 0.1, 0.9)

    # RoI(Region of Interest) 당 학습에 사용할 샘플 수 결정 (64, 128, 256 중 선택)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = trial.suggest_categorical('batch_size_per_image', [64, 128, 256])
    # 학습할 클래스 수 지정
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  # Number of classes in dataset
    
    # Output directory
    cfg.OUTPUT_DIR = f'./output_trial_{trial.number}'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Trainer
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # COCOEvaluator로 성능 평가
    evaluator = COCOEvaluator("coco_trash_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "coco_trash_test")
    metrics = trainer.test(cfg, trainer.model, evaluators=[evaluator])

    # COCOEvaluator의 'bbox/AP'(평균 precision) 값을 최종 objective로 사용
    return metrics['bbox']['AP']

# Optuna study 생성 및 최적화 실행
# 그리드 샘플링 방식
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.GridSampler({
    'ims_per_batch': [4, 8, 16],
    'lr': [1e-5, 1e-4, 1e-3],
    'gamma': [0.1, 0.5, 0.9],
    'batch_size_per_image': [64, 128, 256]
}))
study.optimize(objective, n_trials=20)

print(f"Best hyperparameters: {study.best_params}")
print(f"Best AP: {study.best_value}")