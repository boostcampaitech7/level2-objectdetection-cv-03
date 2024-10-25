import sys
import argparse
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

def parse_args():
    parser = argparse.ArgumentParser(description="Train object detection model")
    parser.add_argument(
        "--config", 
        default="config_default",  # 기본 설정 파일 이름
        help="Configuration file to use, e.g., config_cascade_rcnn_r101"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 설정 파일의 경로를 동적으로 설정
    config_module = f'configs._configs_.{args.config}'
    try:
        # 동적 임포트: 해당 경로에서 get_cfg 함수를 가져온다
        config = __import__(config_module, fromlist=['get_cfg'])
        cfg = config.get_cfg()
    except ImportError as e:
        print(f"Failed to import config: {config_module}")
        raise e

    # 데이터셋 및 모델 생성
    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(cfg.model)
    model.init_weights()

    # 모델 학습
    train_detector(model, datasets, cfg, distributed=False, validate=True)

if __name__ == '__main__':
    sys.path.append('./')
    main()
