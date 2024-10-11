import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.dataset import register_datasets
from common.config import setup_config
from inference import run_inference, save_submission

def main():
    # 데이터셋 등록
    register_datasets()
    
    # 설정 불러오기 (training=False)
    cfg = setup_config(training=False)

    # 출력 디렉토리 생성
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Inference 수행
    prediction_strings, file_names = run_inference(cfg)

    # 결과 저장
    save_submission(cfg, prediction_strings, file_names)

if __name__ == "__main__":
    main()
