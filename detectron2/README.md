폴더 세팅 방법
```
dataset/
|
├── test/
    └── ...
├── train/
    └── ...
├── test.json
└── train.json

detectron2/
|
├── ...
├── output/ # 이 위치에 detectron2 output 저장 (local)
└── ...

git # home 위치에 git 폴더 생성 -> 초기화 -> github repository 연동
|
└── detectron2_base_code/ # 이 위치에서 detectron2 작업
    │
    ├── common/
    │   └── ...
    ├── train/
    │   └── ...
    ├── inference/
    │   └── ...
    └── ...
```

detectron2 실행 시 git/detectron2_base_code/ 에서

학습 실행:
`python train/train.py`

추론 실행:
`python inference/run_inference.py`
