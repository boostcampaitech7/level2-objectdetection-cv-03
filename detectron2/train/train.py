import sys
import os

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.dataset import register_datasets
from common.config import setup_config
from trainer import MyTrainer, SwinTrainer

def main():
    # 데이터셋 등록
    register_datasets()
    
    # 설정 불러오기
    cfg = setup_config(training=True)
    
    # 출력 디렉토리 생성
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 트레이너 초기화 및 학습 시작
    trainer = SwinTrainer(cfg)

    # load pretrained weights
    other_weights = torch.load('/data/ephemeral/home/git/detectron2_base_code/swin/faster_rcnn_swint_T.pth')['model']
    
    # 모델의 state_dict 가져오기
    model_state_dict = trainer.model.state_dict()

    # 필요한 레이어만 필터링
    for key in model_state_dict.keys():
        if key in other_weights:
            # shape가 맞지 않는 레이어는 건너뜁니다.
            if model_state_dict[key].shape == other_weights[key].shape:
                model_state_dict[key] = other_weights[key]
                print(f"Loaded {key} with shape {other_weights[key].shape}")
            else:
                print(f"Skipping {key} due to shape mismatch: "
                      f"checkpoint {other_weights[key].shape}, "
                      f"current model {model_state_dict[key].shape}")

    # 수정된 state_dict를 모델에 로드
    trainer.model.load_state_dict(model_state_dict)
    
    # 학습 시작
    trainer.resume_or_load(resume=False)

    trainer.train()
    """
    # 데이터셋 등록
    register_datasets()
    
    # 설정 불러오기
    cfg = setup_config(training=True)
    
    # 출력 디렉토리 생성
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 트레이너 초기화 및 학습 시작
    trainer = SwinTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    """

if __name__ == "__main__":
    main()
