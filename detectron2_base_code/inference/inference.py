import os
import torch
import pandas as pd
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader

from common.augmentation import MyMapper

def run_inference(cfg):
    # Predictor 설정
    predictor = DefaultPredictor(cfg)

    # Test Loader 생성
    test_loader = build_detection_test_loader(cfg, 'coco_trash_test', MyMapper)

    # 추론 및 결과 후처리
    prediction_strings = []
    file_names = []

    for data in tqdm(test_loader):
        prediction_string = ''
        data = data[0]

        image = data['image'].cpu().numpy().transpose(1, 2, 0)

        outputs = predictor(image)['instances']

        targets = outputs.pred_classes.cpu().tolist()
        boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
        scores = outputs.scores.cpu().tolist()

        for target, box, score in zip(targets, boxes, scores):
            prediction_string += f"{target} {score} {box[0]} {box[1]} {box[2]} {box[3]} "

        prediction_strings.append(prediction_string)
        file_names.append(data['file_name'].replace('../../dataset/', ''))

    return prediction_strings, file_names

def save_submission(cfg, prediction_strings, file_names):
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.OUTPUT_DIR, 'submission_det2.csv'), index=None)
