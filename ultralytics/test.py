from ultralytics import YOLO
import os
import pandas as pd

def yolo_to_pascal_voc(results):
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
    
    return voc_results, file_results

model = YOLO('/data/ephemeral/home/PJU/yolo_series/runs/detect/train2/weights/best.pt')

results = model.predict(source='/data/ephemeral/home/PJU/dataset/test', conf=0.05, imgsz=512)

voc_results, file_results = yolo_to_pascal_voc(results)

submission = pd.DataFrame()
submission['PredictionString'] = voc_results
submission['image_id'] = file_results
submission.to_csv('/data/ephemeral/home/PJU/yolo_series/yolov8_test3.csv', index=None)