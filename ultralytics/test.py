import argparse
from ultralytics import YOLO
import os
import pandas as pd

def main(opt):
    model = YOLO(opt.model)
    results = model.predict(source=opt.source, conf=opt.conf, imgsz=opt.imgsz, stream=opt.stream)
    voc_results = []
    file_results = []
    root = 'test'

    # 결과 순회
    for result in results:
        voc_format = ''
        boxes = result.boxes  # 예측된 바운딩 박스들
        index = os.path.basename(result.path)
        file_name = os.path.join(root, index)
        for box in boxes:
            # 바운딩 박스 좌표 추출
            xyxy = box.xyxy[0].cpu().numpy()  # 바운딩 박스 좌표 (xmin, ymin, xmax, ymax)
            score = box.conf[0].cpu().numpy()  # 신뢰도
            label = int(box.cls[0].cpu().numpy())  # 클래스 레이블
            # Pascal VOC 형식으로 변환
            voc_format += str(label) + ' ' + str(score) + ' ' + str(xyxy[0]) + ' ' + str(xyxy[1]) + ' ' + str(xyxy[2]) + ' ' + str(xyxy[3]) + ' '

        voc_results.append(voc_format)
        file_results.append(file_name)
    
    submission = pd.DataFrame()
    submission['PredictionString'] = voc_results
    submission['image_id'] = file_results

    save_dir = opt.save_dir
    csv_name = opt.name
    submission.to_csv(os.path.join(save_dir, f'{csv_name}.csv'), index=None)

def parse_args():
    # argparse로 인자 받기
    parser = argparse.ArgumentParser(description="Test YOLOv8 model")
    
    # 필요한 인자 추가
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--source', type=str, default='/data/ephemeral/home/PJU/dataset/test')
    parser.add_argument('--save_dir', type=str, default='/data/ephemeral/home/PJU/yolo_series')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--conf', type=float, default=0.05)
    parser.add_argument('--imgsz', type=int, default=512)
    parser.add_argument('--stream', type=bool, default=True)

    # 인자 파싱
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_args()
    main(opt)