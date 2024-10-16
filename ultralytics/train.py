from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='/data/ephemeral/home/PJU/yolo_series/data.yaml', epochs=100, patience=10, batch=32, imgsz=512)