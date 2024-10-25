import argparse
from ultralytics import YOLO

def main(opt):
    model = YOLO(opt.model)
    model.train(**vars(opt))

def parse_args():
    # argparse로 인자 받기
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")
    
    # 필요한 인자 추가
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--imgsz', type=int, default=512)
    parser.add_argument('--name', type=str, default='yolov8x')

    # optimizer
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr0', type=float, default=1e-4)
    parser.add_argument('--lrf', type=float, default=1e-2)

    # augment
    parser.add_argument('--augment', type=bool, required=True)
    parser.add_argument('--hsv_h', type=float, default=0.015)
    parser.add_argument('--hsv_s', type=float, default=0.7)
    parser.add_argument('--hsv_v', type=float, default=0.4)
    parser.add_argument('--degrees', type=float, default=0.0)
    parser.add_argument('--translate', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--flipud', type=float, default=0.0)
    parser.add_argument('--fliplr', type=float, default=0.5)
    parser.add_argument('--mosaic', type=float, default=1.0)
    parser.add_argument('--mixup', type=float, default=0.0)
    parser.add_argument('--copy_paste', type=float, default=0.0)
    parser.add_argument('--erasing', type=float, default=0.4)
    parser.add_argument('--crop_fraction', type=float, default=1.0)

    # 인자 파싱
    return parser.parse_args()

if __name__ == "__main__":
    # 인자 파싱
    opt = parse_args()
    
    # 학습 함수 호출
    main(opt)