
```
mmdetection/
│
├── configs/
│   └── _configs_/
│       ├── __init__.py          
│       ├── config_default.py     
│       ├── config_cascade_rcnn.py 
│       └── ...                   
│
├── train.py                      # Main script for training
├── README.md                     # This documentation file
└── ...
```

### 사용법
**1. config_default.py 기반으로 config 작성**

**2. train.py 실행**

  '''bash
  python train.py --config your_config
