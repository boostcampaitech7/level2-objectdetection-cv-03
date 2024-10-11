import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

def register_datasets():
    # 학습 데이터셋 등록
    try:
        register_coco_instances('coco_trash_train', {}, '../../dataset/train.json', '../../dataset/')
    except AssertionError:
        pass

    # 테스트 데이터셋 등록
    try:
        register_coco_instances('coco_trash_test', {}, '../../dataset/test.json', '../../dataset/')
    except AssertionError:
        pass

    # 클래스 이름 설정
    MetadataCatalog.get('coco_trash_train').thing_classes = [
        "General trash", "Paper", "Paper pack", "Metal", 
        "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"
    ]
