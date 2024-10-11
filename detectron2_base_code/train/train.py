import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.dataset import register_datasets
from common.config import setup_config
from trainer import MyTrainer

def main():
    # 데이터셋 등록
    register_datasets()
    
    # 설정 불러오기
    cfg = setup_config(training=True)
    
    # 출력 디렉토리 생성
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 트레이너 초기화 및 학습 시작
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()
