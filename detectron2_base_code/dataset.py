import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

def register_datasets():
    # Dataset 등록
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
